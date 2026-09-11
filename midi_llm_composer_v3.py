from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "slseanwu/MIDI-LLM_Llama-3.2-1B"
LLAMA_VOCAB_SIZE = 128256
AMT_GPT2_BOS_ID = 55026
SYSTEM_PROMPT = "You are a world-class composer. Please compose some music according to the following description: "
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}
ROLES = ("drums", "bass", "chords", "melody")


def ensure_anticipation() -> None:
    try:
        import anticipation.convert  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Anticipation MIDI decoder is missing. Re-run the latest Colab setup cell.") from exc


def ensure_model(root: Path) -> Path:
    model_dir = root / "work" / "midi_llm_models" / "MIDI-LLM_Llama-3.2-1B"
    weights = model_dir / "model.safetensors"
    if weights.exists() and weights.stat().st_size > 2_500_000_000 and (model_dir / "config.json").exists():
        print("♻️ Reusing persistent MIDI-LLM model:", model_dir)
        return model_dir
    print("⬇️ Downloading MIDI-LLM composer weights to Drive for reuse")
    model_dir.mkdir(parents=True, exist_ok=True)
    cache = Path("/content/musicm8_midi_llm_download_cache")
    cache.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(model_dir), cache_dir=str(cache))
    if not weights.exists():
        raise FileNotFoundError(weights)
    return model_dir


def style_method(style: str) -> str:
    return {
        "uk_garage": "2-step kick/snare pocket, swung sixteenths, shuffled hats, syncopated sub bass, short chord stabs, call-and-response hooks, four/eight-bar phrasing",
        "house": "four-on-the-floor drums, offbeat hats, locked kick and bass, economical chord rhythm, eight-bar tension/release phrasing",
        "techno": "precise four-on-the-floor pulse, repeating bass ostinato, sparse purposeful harmony, evolving motif and phrase-boundary transitions",
        "dnb": "breakbeat-derived kick/snare placement, fast controlled hats, syncopated sub/reese bass, half-time harmonic space and phrase-boundary fills",
        "trap": "half-time backbeat, purposeful kick/808 interaction, controlled hat rolls, sparse minor harmony and a strong top-line motif",
        "hiphop": "laid-back drum pocket, coherent bass, sample-like chord voicings and motif repetition with small phrase variations",
        "ambient": "long harmonic motion, smooth voice-leading, sparse percussion, slowly evolving texture and deliberate silence",
        "electronic": "tight electronic drums, bass/chord interlock, recurring motif, four/eight-bar phrasing, builds and breakdowns",
    }.get(style, "tight electronic drums, coherent bass/chord interlock, recurring motif and clear phrase development")


def section_prompt(idea: str, plan: dict[str, Any], section: dict[str, Any], motif_hint: str | None = None) -> str:
    style = str(plan.get("style", "electronic"))
    bpm = float(plan.get("bpm", 120.0))
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    bars = max(1, int(section.get("bars", 4)))
    name = str(section.get("name", "section"))
    energy = float(section.get("energy", 0.6))
    prog = "-".join(str(int(x)) for x in plan.get("progression_degrees", [1, 6, 3, 7]))
    continuity = f"Develop this existing hook identity: {motif_hint}. " if motif_hint else "Establish a memorable original hook identity. "
    return (
        f"Compose ONLY a {bars}-bar {name} section for an original {style.replace('_', ' ')} song at {bpm:.1f} BPM in {KEY_NAMES[root]} {mode}. "
        f"Mood/idea: {idea}. Energy {energy:.2f}. Harmonic progression degrees {prog}. "
        f"Use authentic production/composition practice: {style_method(style)}. {continuity}"
        "Write a balanced MULTITRACK MIDI section: drums/percussion, bass, actual harmony/chords, and a lead/hook. "
        "Do not let drums consume most of the events. Harmony must move, bass must support harmony and groove, and lead phrases must leave breathing room. "
        "Keep every part on one 4/4 clock. Fills belong at phrase boundaries. Use standard MIDI drums and conventional MIDI instruments. "
        "Do not quote any existing song."
    )


def rescue_prompt(role: str, idea: str, plan: dict[str, Any], section: dict[str, Any], motif_hint: str | None) -> str:
    base = section_prompt(idea, plan, section, motif_hint)
    role_text = {
        "drums": "Prioritise a clean, genre-correct drum performance with a stable pocket and restrained fills.",
        "bass": "Prioritise a coherent bass line that outlines the stated progression and locks to the groove; avoid random chromatic jumps.",
        "chords": "Prioritise real chord/harmony events with clear voice-leading through the stated progression; do not substitute a monophonic riff for harmony.",
        "melody": "Prioritise a singable lead/hook with repeated motif identity, rests and chord-aware phrase endings.",
    }[role]
    return base + " " + role_text + f" Make the {role} role especially clear and musically usable."


def trim_music_tokens(raw_ids: list[int]) -> list[int]:
    shifted = [int(x) - LLAMA_VOCAB_SIZE for x in raw_ids]
    valid: list[int] = []
    for token in shifted:
        if 0 <= token < AMT_GPT2_BOS_ID:
            valid.append(token)
        else:
            break
    valid = valid[: (len(valid) // 3) * 3]
    if len(valid) < 24:
        raise RuntimeError(f"MIDI-LLM returned too few valid music tokens ({len(valid)})")
    return valid


def group_size(inst: pretty_midi.Instrument, tolerance: float = 0.035) -> float:
    notes = sorted(inst.notes, key=lambda n: n.start)
    if not notes:
        return 0.0
    groups: list[int] = []
    start: float | None = None
    count = 0
    for n in notes:
        if start is None or abs(float(n.start) - start) <= tolerance:
            count += 1
            if start is None:
                start = float(n.start)
        else:
            groups.append(count)
            start = float(n.start)
            count = 1
    groups.append(count)
    return float(np.mean(groups))


def role_for(inst: pretty_midi.Instrument) -> str:
    if inst.is_drum:
        return "drums"
    if not inst.notes:
        return "chords"
    pitches = np.asarray([n.pitch for n in inst.notes], dtype=np.float32)
    med = float(np.median(pitches))
    mean_dur = float(np.mean([max(0.01, n.end - n.start) for n in inst.notes]))
    simult = group_size(inst)
    prog = int(inst.program)
    if 32 <= prog <= 39 or med < 50:
        return "bass"
    if simult >= 1.45 or mean_dur > 0.75 or 88 <= prog <= 95:
        return "chords"
    if med >= 58 or 80 <= prog <= 87:
        return "melody"
    return "chords"


def roles_for(pm: pretty_midi.PrettyMIDI) -> list[str]:
    roles = [role_for(i) for i in pm.instruments]
    pitched = [j for j, i in enumerate(pm.instruments) if i.notes and not i.is_drum]
    if "bass" not in roles and pitched:
        j = min(pitched, key=lambda k: np.median([n.pitch for n in pm.instruments[k].notes]))
        roles[j] = "bass"
    if "melody" not in roles and pitched:
        options = [j for j in pitched if roles[j] != "bass"] or pitched
        j = max(options, key=lambda k: np.median([n.pitch for n in pm.instruments[k].notes]) - 4.0 * group_size(pm.instruments[k]))
        roles[j] = "melody"
    if "chords" not in roles and pitched:
        options = [j for j in pitched if roles[j] not in {"bass", "melody"}]
        if options:
            roles[options[0]] = "chords"
    return roles


def scale_pcs(root: int, mode: str) -> list[int]:
    return [int((root + x) % 12) for x in SCALES.get(mode, SCALES["minor"])]


def key_fit(pm: pretty_midi.PrettyMIDI, root: int, mode: str, shift: int = 0) -> float:
    pcs = set(scale_pcs(root, mode))
    good = total = 0.0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            w = max(0.05, min(2.0, n.end - n.start))
            total += w
            good += w if (int(n.pitch) + shift) % 12 in pcs else 0.0
    return good / total if total else 0.0


def best_transpose(pm: pretty_midi.PrettyMIDI, root: int, mode: str) -> tuple[int, float]:
    scored = [(key_fit(pm, root, mode, s) - 0.002 * abs(s), key_fit(pm, root, mode, s), s) for s in range(-6, 7)]
    _, fit, shift = max(scored)
    return int(shift), float(fit)


def role_counts(pm: pretty_midi.PrettyMIDI) -> tuple[list[str], dict[str, int]]:
    roles = roles_for(pm)
    counts = {r: 0 for r in ROLES}
    for inst, role in zip(pm.instruments, roles):
        counts[role] += len(inst.notes)
    return roles, counts


def minimums(section: dict[str, Any], style: str) -> dict[str, int]:
    bars = max(1, int(section.get("bars", 4)))
    name = str(section.get("name", "section")).lower()
    sparse = any(x in name for x in ("intro", "break", "outro"))
    if style == "ambient":
        return {"drums": 0, "bass": max(1, bars // 2), "chords": max(4, bars), "melody": max(1, bars // 2)}
    if sparse:
        return {"drums": max(2, bars), "bass": max(2, bars // 2), "chords": max(4, bars), "melody": max(1, bars // 2)}
    return {"drums": max(8, bars * 3), "bass": max(3, bars), "chords": max(6, bars * 2), "melody": max(2, bars // 2)}


def candidate_metrics(path: Path, plan: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
    pm = pretty_midi.PrettyMIDI(str(path))
    pm.instruments = [i for i in pm.instruments if i.notes]
    roles, counts = role_counts(pm)
    bpm = float(plan.get("bpm", 120.0))
    bars = max(1, int(section.get("bars", 4)))
    target = bars * 4.0 * 60.0 / max(1.0, bpm)
    duration = max(0.001, float(pm.get_end_time()))
    ratio = duration / target
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    shift, fit = best_transpose(pm, root, mode)
    mins = minimums(section, str(plan.get("style", "electronic")))
    strengths = [min(1.0, counts[r] / max(1, mins[r])) for r in ROLES if mins[r] > 0]
    pitched = counts["bass"] + counts["chords"] + counts["melody"]
    drum_ratio = counts["drums"] / max(1, pitched)
    duration_score = math.exp(-0.75 * abs(math.log(max(1e-6, ratio))))
    balance = float(np.mean(strengths)) if strengths else 0.0
    score = 3.0 * balance + 1.2 * fit + 0.7 * duration_score - max(0.0, drum_ratio - 6.0) * 0.12
    failures = [f"{r} {counts[r]}<{mins[r]}" for r in ROLES if mins[r] > 0 and counts[r] < mins[r]]
    production_ready = not failures and fit >= 0.34 and drum_ratio <= 9.0
    return {
        "path": str(path), "score": round(score, 5), "roles": counts, "role_assignments": roles,
        "minimum_role_notes": mins, "duration_s": round(duration, 3), "target_duration_s": round(target, 3),
        "duration_ratio": round(ratio, 4), "global_transpose": shift, "key_fit_after_global_transpose": round(fit, 4),
        "drum_to_pitched_ratio": round(drum_ratio, 3), "production_ready": production_ready, "failures": failures,
    }


def generate_candidate(model: Any, tokenizer: Any, prompt: str, token_budget: int, temperature: float, top_p: float, seed: int):
    from anticipation.convert import events_to_midi
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    enc = tokenizer(SYSTEM_PROMPT + prompt + " ", return_tensors="pt", padding=False)
    ids = enc["input_ids"]
    bos = torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]], dtype=ids.dtype)
    ids = torch.cat([ids, bos], dim=1).to("cuda")
    with torch.no_grad():
        output = model.generate(
            input_ids=ids, do_sample=True, max_new_tokens=max(384, min(2046, int(token_budget))),
            temperature=max(0.35, float(temperature)), top_p=max(0.50, min(1.0, float(top_p))),
            num_return_sequences=1, pad_token_id=tokenizer.pad_token_id,
        )
    raw = output[0, ids.shape[1]:].detach().cpu().tolist()
    tokens = trim_music_tokens(raw)
    return events_to_midi(tokens), len(tokens)


def nearest_pitch(target: int, pcs: set[int], lo: int = 24, hi: int = 96) -> int:
    pool = [p for p in range(lo, hi + 1) if p % 12 in pcs]
    if not pool:
        return int(np.clip(target, lo, hi))
    return min(pool, key=lambda p: (abs(p - target), p))


def chord_pcs_at(local_t: float, plan: dict[str, Any]) -> tuple[set[int], int]:
    bpm = float(plan.get("bpm", 120.0))
    beat = 60.0 / max(1.0, bpm)
    bar = max(0, int(local_t / max(1e-6, beat * 4.0)))
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    scale = scale_pcs(root, mode)
    prog = [max(1, min(7, int(x))) for x in plan.get("progression_degrees", [1, 6, 3, 7])]
    degree = prog[bar % len(prog)] - 1
    chord = {scale[degree % 7], scale[(degree + 2) % 7], scale[(degree + 4) % 7]}
    return chord, scale[degree % 7]


def harmony_lock(pitch: int, role: str, local_t: float, plan: dict[str, Any]) -> int:
    if role == "drums":
        return int(pitch)
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    scale = set(scale_pcs(root, mode))
    chord, chord_root = chord_pcs_at(local_t, plan)
    if role == "bass":
        allowed = {chord_root, (chord_root + 7) % 12}
        return nearest_pitch(int(pitch), allowed, 28, 55)
    if role == "chords":
        return nearest_pitch(int(pitch), chord, 45, 84)
    beat = 60.0 / max(1.0, float(plan.get("bpm", 120.0)))
    strong = abs((local_t / beat) - round(local_t / beat)) < 0.12
    return nearest_pitch(int(pitch), chord if strong else scale, 55, 84)


def pick_role_instruments(pm: pretty_midi.PrettyMIDI, requested: str) -> list[pretty_midi.Instrument]:
    pm.instruments = [i for i in pm.instruments if i.notes]
    roles = roles_for(pm)
    exact = [i for i, r in zip(pm.instruments, roles) if r == requested]
    if exact:
        return exact
    pitched = [i for i in pm.instruments if not i.is_drum and i.notes]
    if requested == "drums":
        return [i for i in pm.instruments if i.is_drum and i.notes]
    if not pitched:
        return []
    if requested == "bass":
        return [min(pitched, key=lambda i: np.median([n.pitch for n in i.notes]))]
    if requested == "chords":
        return [max(pitched, key=lambda i: group_size(i) + 0.6 * np.mean([n.end - n.start for n in i.notes]))]
    return [max(pitched, key=lambda i: np.median([n.pitch for n in i.notes]) - 4.0 * group_size(i))]


def normalize_role_instruments(
    source_instruments: list[pretty_midi.Instrument], role: str, plan: dict[str, Any], section: dict[str, Any],
    offset_s: float, source_duration: float, transpose: int,
) -> list[pretty_midi.Instrument]:
    bpm = float(plan.get("bpm", 120.0))
    bars = max(1, int(section.get("bars", 4)))
    target_duration = bars * 4.0 * 60.0 / max(1.0, bpm)
    section_end = offset_s + target_duration
    time_scale = target_duration / max(0.1, source_duration)
    swing = max(0.0, min(0.20, float(plan.get("groove", {}).get("swing", 0.0))))
    step = 60.0 / max(1.0, bpm) / 4.0
    half = step * 0.5
    out: list[pretty_midi.Instrument] = []
    for k, src in enumerate(source_instruments):
        dst = pretty_midi.Instrument(program=int(src.program), is_drum=(role == "drums"), name=f"{role}_{section.get('name','section')}_{k+1}")
        seen: set[tuple[int, int]] = set()
        for n in sorted(src.notes, key=lambda x: (x.start, x.pitch, x.end)):
            raw_local = float(n.start) * time_scale
            idx = max(0, int(round(raw_local / step)))
            local = idx * step + (step * swing if idx % 2 else 0.0)
            start = offset_s + local
            if start >= section_end - 0.01:
                continue
            if role == "drums":
                if not 35 <= int(n.pitch) <= 81:
                    continue
                end = min(section_end - 0.002, start + min(0.12, step * 0.85))
                pitch = int(n.pitch)
            else:
                raw_dur = max(half, (float(n.end) - float(n.start)) * time_scale)
                dur = max(half, round(raw_dur / half) * half)
                end = min(section_end - 0.002, start + dur)
                pitch = harmony_lock(int(n.pitch) + transpose, role, local, plan)
            key = (pitch, idx)
            if key in seen or end <= start + 0.012:
                continue
            seen.add(key)
            dst.notes.append(pretty_midi.Note(int(np.clip(n.velocity, 1, 127)), int(pitch), start, end))
        if dst.notes:
            out.append(dst)
    return out


def normalize_candidate(path: Path, plan: dict[str, Any], section: dict[str, Any], metrics: dict[str, Any], offset_s: float) -> dict[str, list[pretty_midi.Instrument]]:
    pm = pretty_midi.PrettyMIDI(str(path))
    pm.instruments = [i for i in pm.instruments if i.notes]
    roles = roles_for(pm)
    source_duration = max(0.1, float(pm.get_end_time()))
    transpose = int(metrics.get("global_transpose", 0))
    out = {r: [] for r in ROLES}
    for role in ROLES:
        selected = [i for i, r in zip(pm.instruments, roles) if r == role]
        out[role] = normalize_role_instruments(selected, role, plan, section, offset_s, source_duration, transpose)
    return out


def note_count(instruments: list[pretty_midi.Instrument]) -> int:
    return sum(len(i.notes) for i in instruments)


def write_full_project(project: Path, bpm: float, role_groups: dict[str, list[pretty_midi.Instrument]]) -> None:
    full = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for role in ROLES:
        full.instruments.extend(role_groups[role])
    full.write(str(project / "arrangement.mid"))
    stems = project / "midi_stems"
    shutil.rmtree(stems, ignore_errors=True)
    stems.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        if not role_groups[role]:
            continue
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        for inst in role_groups[role]:
            cp = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            cp.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in inst.notes]
            pm.instruments.append(cp)
        pm.write(str(stems / f"{role}.mid"))


def motif_hint_from(instruments: list[pretty_midi.Instrument], root: int, offset_s: float, bpm: float) -> str | None:
    notes = sorted([n for i in instruments for n in i.notes], key=lambda n: n.start)[:8]
    if not notes:
        return None
    pcs = [(int(n.pitch) - root) % 12 for n in notes]
    beats = [round((n.start - offset_s) / (60.0 / bpm), 2) for n in notes]
    return f"lead relative pitch classes {pcs}, onsets around beats {beats}"


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 resilient learned section composer with targeted AI role rescue and harmony locking.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--idea", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidates", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=2046)
    p.add_argument("--temperature", type=float, default=0.90)
    p.add_argument("--top-p", type=float, default=0.95)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("MIDI-LLM composer requires CUDA")
    ensure_anticipation()
    root = args.root.resolve()
    project = args.out.resolve()
    project.mkdir(parents=True, exist_ok=True)
    failure_path = project / "composition_failure.json"
    if failure_path.exists():
        failure_path.unlink()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    bpm = float(plan.get("bpm", 120.0))
    root_pc = int(plan.get("key_root", 0)) % 12
    sections = [s for s in plan.get("sections", []) if isinstance(s, dict) and int(s.get("bars", 0)) > 0]
    if not sections:
        sections = [{"name": "song", "bars": int(plan.get("bars", 32)), "energy": 0.7}]

    model_dir = ensure_model(root)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, pad_token="<|eot_id|>")
    major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    print("🎼 Loading learned composer:", MODEL_ID)
    print("Precision:", dtype, "| GPU:", torch.cuda.get_device_name(0))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype, low_cpu_mem_usage=True,
        local_files_only=True, trust_remote_code=True,
    ).to("cuda")
    model.eval()

    candidates_root = project / "ai_composer_candidates"
    shutil.rmtree(candidates_root, ignore_errors=True)
    candidates_root.mkdir(parents=True, exist_ok=True)
    whole_roles: dict[str, list[pretty_midi.Instrument]] = {r: [] for r in ROLES}
    section_reports: list[dict[str, Any]] = []
    prompts: list[str] = []
    offset_s = 0.0
    motif_hint: str | None = None

    try:
        print("\n🧠 MIDI-LLM V3 SECTION COMPOSER")
        print("   AI writes each section; Musicm8 then quantizes and harmony-locks it like a DAW producer/editor.")
        for si, section in enumerate(sections):
            name = str(section.get("name", f"section_{si+1}"))
            bars = max(1, int(section.get("bars", 4)))
            sec_dir = candidates_root / f"{si+1:02d}_{name.replace(' ', '_')}"
            sec_dir.mkdir(parents=True, exist_ok=True)
            prompt = section_prompt(args.idea, plan, section, motif_hint)
            prompts.append(prompt)
            token_budget = min(int(args.max_tokens), max(520, bars * 140))
            attempts = max(1, min(3, int(args.candidates)))
            reports: list[dict[str, Any]] = []
            print(f"\n🎹 Section {si+1}/{len(sections)}: {name} ({bars} bars)")
            for ci in range(attempts):
                seed = int(args.seed) + si * 104729 + ci * 7919
                try:
                    midi_obj, token_count = generate_candidate(model, tokenizer, prompt, token_budget, args.temperature, args.top_p, seed)
                    path = sec_dir / f"candidate_{ci+1}.mid"
                    midi_obj.save(str(path))
                    m = candidate_metrics(path, plan, section)
                    m.update({"candidate": ci + 1, "seed": seed, "tokens": token_count, "error": None})
                    reports.append(m)
                    print(f"   c{ci+1}: score={m['score']:.2f} roles={m['roles']} fit={m['key_fit_after_global_transpose']:.2f}")
                except Exception as exc:
                    reports.append({"candidate": ci + 1, "seed": seed, "score": -999.0, "error": str(exc)})
                    print(f"   c{ci+1}: failed ({exc})")
            usable = [r for r in reports if not r.get("error") and Path(str(r.get("path", ""))).exists()]
            if not usable:
                raise RuntimeError(f"No usable MIDI-LLM candidate for section {name}")
            selected = max(usable, key=lambda r: float(r.get("score", -999.0)))
            section_roles = normalize_candidate(Path(str(selected["path"])), plan, section, selected, offset_s)
            mins = minimums(section, str(plan.get("style", "electronic")))
            rescues: dict[str, Any] = {}

            for role in ROLES:
                if mins[role] <= 0 or note_count(section_roles[role]) >= mins[role]:
                    continue
                print(f"   🛠️ AI rescue for weak {role}: {note_count(section_roles[role])}<{mins[role]}")
                rp = rescue_prompt(role, args.idea, plan, section, motif_hint)
                rescue_best: tuple[float, list[pretty_midi.Instrument], dict[str, Any]] | None = None
                for ri in range(2):
                    seed = int(args.seed) + si * 104729 + 40000 + ROLES.index(role) * 5003 + ri * 9973
                    try:
                        midi_obj, token_count = generate_candidate(model, tokenizer, rp, max(460, bars * 120), 0.82, 0.93, seed)
                        path = sec_dir / f"rescue_{role}_{ri+1}.mid"
                        midi_obj.save(str(path))
                        pm = pretty_midi.PrettyMIDI(str(path))
                        chosen = pick_role_instruments(pm, role)
                        if not chosen:
                            continue
                        shift, fit = best_transpose(pm, int(plan.get("key_root", 0)) % 12, str(plan.get("mode", "minor")))
                        normalized = normalize_role_instruments(chosen, role, plan, section, offset_s, max(0.1, pm.get_end_time()), shift)
                        count = note_count(normalized)
                        rescue_score = count + 20.0 * fit
                        info = {"path": str(path), "seed": seed, "tokens": token_count, "events": count, "key_fit": round(fit, 4)}
                        if rescue_best is None or rescue_score > rescue_best[0]:
                            rescue_best = (rescue_score, normalized, info)
                    except Exception as exc:
                        rescues.setdefault(role, {}).setdefault("errors", []).append(str(exc))
                if rescue_best is not None and note_count(rescue_best[1]) > note_count(section_roles[role]):
                    section_roles[role] = rescue_best[1]
                    rescues[role] = rescue_best[2]
                    print(f"      rescued {role}: {note_count(section_roles[role])} events")

            counts = {r: note_count(section_roles[r]) for r in ROLES}
            hard_missing = []
            if str(plan.get("style", "electronic")) != "ambient" and counts["drums"] == 0:
                hard_missing.append("drums")
            if counts["bass"] == 0:
                hard_missing.append("bass")
            if counts["chords"] == 0:
                hard_missing.append("chords")
            if hard_missing:
                raise RuntimeError(f"Section {name} still has no usable {', '.join(hard_missing)} after targeted AI rescue")

            for role in ROLES:
                whole_roles[role].extend(section_roles[role])
            next_hint = motif_hint_from(section_roles["melody"], root_pc, offset_s, bpm)
            motif_hint = next_hint or motif_hint
            section_reports.append({
                "index": si, "name": name, "bars": bars, "prompt": prompt,
                "candidates": reports, "selected_candidate": int(selected["candidate"]),
                "selected": selected, "rescues": rescues, "final_role_notes": counts,
                "minimum_role_notes": mins, "quality_warnings": [f"{r} {counts[r]}<{mins[r]}" for r in ROLES if mins[r] > 0 and counts[r] < mins[r]],
            })
            offset_s += bars * 4.0 * 60.0 / max(1.0, bpm)

        role_totals = {r: note_count(whole_roles[r]) for r in ROLES}
        if role_totals["chords"] < max(12, int(plan.get("bars", 32)) // 2):
            raise RuntimeError(f"Final AI arrangement remains harmony-starved after rescue: {role_totals}")
        if role_totals["bass"] < max(8, int(plan.get("bars", 32)) // 4):
            raise RuntimeError(f"Final AI arrangement remains bass-starved after rescue: {role_totals}")

        write_full_project(project, bpm, whole_roles)
        (project / "ai_composer_prompt.txt").write_text("\n\n--- SECTION ---\n\n".join(prompts) + "\n", encoding="utf-8")
        report = {
            "format": "musicm8-midi-llm-section-composer-v3",
            "composer": MODEL_ID,
            "composition_mode": "AI section composition + targeted AI role rescue + DAW timing/harmony lock",
            "seed": int(args.seed), "sections": section_reports, "role_notes": role_totals,
            "rule_note_composer_fallback": False,
            "post_editing": "shared groove quantize + global/key/chord correction; no hand-written rhythmic part is substituted for the AI performance",
            "arrangement": str(project / "arrangement.mid"), "midi_stems": str(project / "midi_stems"),
        }
        (project / "composition_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n✅ LEARNED SECTION COMPOSITION + AI ROLE RESCUE:", project / "arrangement.mid")
        print("Role notes:", role_totals)
    except Exception as exc:
        failure = {
            "format": "musicm8-composition-failure-v1", "error": str(exc), "seed": int(args.seed),
            "sections_completed": section_reports, "current_role_totals": {r: note_count(whole_roles[r]) for r in ROLES},
            "hint": "The Colab main cell can print this file directly; the outer CalledProcessError is not the root cause.",
        }
        failure_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n❌ COMPOSER FAILURE SUMMARY")
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        raise


if __name__ == "__main__":
    main()
