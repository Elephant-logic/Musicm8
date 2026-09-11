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
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROLES = ("drums", "bass", "chords", "melody")


def ensure_anticipation() -> None:
    try:
        import anticipation.convert  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "The learned composer needs the Anticipation MIDI decoder. Install requirements-ai.txt from the latest Musicm8 checkout."
        ) from exc


def ensure_model(root: Path) -> Path:
    model_dir = root / "work" / "midi_llm_models" / "MIDI-LLM_Llama-3.2-1B"
    weights = model_dir / "model.safetensors"
    if weights.exists() and weights.stat().st_size > 2_500_000_000 and (model_dir / "config.json").exists():
        print("♻️ Reusing persistent MIDI-LLM model:", model_dir)
        return model_dir
    print("⬇️ First run: downloading the MIDI-LLM composer (~3.5 GB) to Drive for reuse")
    model_dir.mkdir(parents=True, exist_ok=True)
    cache = Path("/content/musicm8_midi_llm_download_cache")
    cache.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(model_dir), cache_dir=str(cache))
    if not weights.exists():
        raise FileNotFoundError(f"MIDI-LLM download did not create {weights}")
    return model_dir


def style_method(style: str) -> str:
    methods = {
        "uk_garage": "authentic 2-step kick/snare pocket, swung sixteenths, shuffled hats, syncopated sub bass, short chord stabs, call-and-response hooks and four/eight-bar phrase changes",
        "house": "four-on-the-floor drums, offbeat hats, locked kick/bass relationship, economical chord rhythm, eight-bar phrases and tension/release transitions",
        "techno": "precise four-on-the-floor pulse, repeating bass ostinato, evolving motif, sparse purposeful harmony and eight-bar density changes",
        "dnb": "breakbeat-derived kick/snare placement, fast hats, syncopated sub/reese-style bass writing, half-time harmonic space and phrase-boundary fills",
        "trap": "half-time backbeat, purposeful kick/808 interaction, controlled hat subdivisions and rolls, sparse minor harmony and a strong top-line motif",
        "hiphop": "laid-back drum pocket, coherent bass line, sample-like chord voicings, motif repetition with small variations and four-bar phrase answers",
        "ambient": "long harmonic motion, voice-leading, sparse percussion, slowly evolving texture and melody with deliberate silence",
        "electronic": "tight electronic drums, bass/chord interlock, recurring motif, four/eight-bar phrasing, clear builds, breakdowns and transitions",
    }
    return methods.get(style, methods["electronic"])


def section_prompt(idea: str, plan: dict[str, Any], section: dict[str, Any], motif_hint: str | None) -> str:
    style = str(plan.get("style", "electronic"))
    bpm = float(plan.get("bpm", 120.0))
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    bars = int(section.get("bars", 4))
    name = str(section.get("name", "section"))
    energy = float(section.get("energy", 0.6))
    prog = "-".join(str(int(x)) for x in plan.get("progression_degrees", [1, 6, 3, 7]))
    continuity = (
        f"Develop the song's existing hook identity: {motif_hint}. Do not simply repeat it note-for-note; vary it musically while keeping the same identity. "
        if motif_hint else
        "Establish a memorable original hook/motif that later sections can develop. "
    )
    return (
        f"Compose ONLY the {bars}-bar {name} section of an original {style.replace('_', ' ')} song at {bpm:.1f} BPM in {KEY_NAMES[root]} {mode}. "
        f"Song idea: {idea}. Section energy is {energy:.2f}. Global progression degrees: {prog}. "
        f"Use real arranging practice for this genre: {style_method(style)}. "
        f"{continuity}"
        "This is a MULTITRACK section. Give meaningful material to drums/percussion, bass, harmony/chords and a lead/hook; do not spend nearly all events on drums. "
        "Bass must lock to the kick, harmony must actually move through the section, and lead phrases should answer the harmony rather than fire random notes. "
        "Keep every part on one 4/4 musical clock. Use fills only near phrase/section boundaries. Leave space instead of constant notes. "
        "Use conventional MIDI instruments and the standard drum channel. Do not quote or recreate any existing song."
    )


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
    count = 0
    start: float | None = None
    for n in notes:
        if start is None or abs(float(n.start) - start) <= tolerance:
            count += 1
            if start is None:
                start = float(n.start)
        else:
            groups.append(count)
            count = 1
            start = float(n.start)
    groups.append(count)
    return float(np.mean(groups))


def role_for(inst: pretty_midi.Instrument) -> str:
    if inst.is_drum:
        return "drums"
    pitches = [n.pitch for n in inst.notes]
    if not pitches:
        return "chords"
    med = float(np.median(pitches))
    prog = int(inst.program)
    simult = group_size(inst)
    mean_dur = float(np.mean([max(0.01, n.end - n.start) for n in inst.notes]))
    if 32 <= prog <= 39 or med < 50:
        return "bass"
    if 80 <= prog <= 87:
        return "melody"
    if 88 <= prog <= 103:
        return "chords"
    if simult >= 1.55 or mean_dur > 0.72:
        return "chords"
    if med >= 59:
        return "melody"
    return "chords"


def roles_for(pm: pretty_midi.PrettyMIDI) -> list[str]:
    roles = [role_for(i) for i in pm.instruments]
    pitched = [idx for idx, inst in enumerate(pm.instruments) if inst.notes and not inst.is_drum]
    if "bass" not in roles and pitched:
        idx = min(pitched, key=lambda j: np.median([n.pitch for n in pm.instruments[j].notes]))
        if np.median([n.pitch for n in pm.instruments[idx].notes]) < 62:
            roles[idx] = "bass"
    if "melody" not in roles and pitched:
        options = [j for j in pitched if roles[j] != "bass"] or pitched
        idx = max(options, key=lambda j: np.median([n.pitch for n in pm.instruments[j].notes]) - 5.0 * group_size(pm.instruments[j]))
        roles[idx] = "melody"
    if "chords" not in roles and pitched:
        options = [j for j in pitched if roles[j] not in {"bass", "melody"}]
        if options:
            roles[options[0]] = "chords"
    return roles


def key_fit(pm: pretty_midi.PrettyMIDI, root: int, mode: str, shift: int = 0) -> float:
    pcs = {(root + x) % 12 for x in SCALES.get(mode, SCALES["minor"])}
    total = 0.0
    good = 0.0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            weight = max(0.05, min(2.0, n.end - n.start))
            total += weight
            if (int(n.pitch) + shift) % 12 in pcs:
                good += weight
    return good / total if total else 0.0


def best_global_transpose(pm: pretty_midi.PrettyMIDI, root: int, mode: str) -> tuple[int, float]:
    choices = []
    for shift in range(-6, 7):
        fit = key_fit(pm, root, mode, shift)
        choices.append((fit - 0.002 * abs(shift), fit, shift))
    _, fit, shift = max(choices)
    return int(shift), float(fit)


def section_quality(path: Path, plan: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
    pm = pretty_midi.PrettyMIDI(str(path))
    pm.instruments = [i for i in pm.instruments if i.notes]
    roles = roles_for(pm)
    counts = {r: 0 for r in ROLES}
    for inst, role in zip(pm.instruments, roles):
        counts[role] += len(inst.notes)

    duration = max(0.001, float(pm.get_end_time()))
    bpm = float(plan.get("bpm", 120.0))
    bars = max(1, int(section.get("bars", 4)))
    target = bars * 4.0 * 60.0 / max(1.0, bpm)
    ratio = duration / max(target, 1e-6)
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    shift, fit = best_global_transpose(pm, root, mode)

    name = str(section.get("name", "section")).lower()
    style = str(plan.get("style", "electronic"))
    sparse = any(x in name for x in ("intro", "break", "outro"))
    if style == "ambient":
        minimums = {"drums": 0, "bass": max(2, bars // 2), "chords": max(6, bars * 2), "melody": max(2, bars // 2)}
    elif sparse:
        minimums = {"drums": max(2, bars), "bass": max(3, bars), "chords": max(6, bars * 2), "melody": max(2, bars // 2)}
    else:
        minimums = {"drums": max(12, bars * 4), "bass": max(4, bars), "chords": max(8, bars * 2), "melody": max(4, bars)}

    role_strengths = []
    failures = []
    for role in ROLES:
        need = minimums[role]
        got = counts[role]
        if need > 0:
            role_strengths.append(min(1.0, got / need))
            if got < need:
                failures.append(f"{role} {got}<{need}")

    pitched = counts["bass"] + counts["chords"] + counts["melody"]
    drum_ratio = counts["drums"] / max(1, pitched)
    if drum_ratio > 7.0 and style != "ambient":
        failures.append(f"drums dominate pitched material ({drum_ratio:.1f}:1)")

    duration_score = math.exp(-1.25 * abs(math.log(max(1e-6, ratio))))
    balance = float(np.mean(role_strengths)) if role_strengths else 0.0
    production_ready = bool(not failures and 0.45 <= ratio <= 2.20 and fit >= 0.42)
    score = 2.5 * balance + 1.35 * duration_score + 1.25 * fit - max(0.0, drum_ratio - 4.5) * 0.16
    if production_ready:
        score += 1.5
    else:
        score -= 2.0

    return {
        "path": str(path),
        "score": round(float(score), 5),
        "duration_s": round(duration, 3),
        "target_duration_s": round(target, 3),
        "duration_ratio": round(ratio, 4),
        "roles": counts,
        "role_assignments": roles,
        "minimum_role_notes": minimums,
        "global_transpose": shift,
        "key_fit_after_global_transpose": round(fit, 4),
        "drum_to_pitched_ratio": round(drum_ratio, 3),
        "production_ready": production_ready,
        "failures": failures,
    }


def groove_start(t: float, bpm: float, swing: float) -> float:
    step = 60.0 / max(1.0, bpm) / 4.0
    idx = max(0, int(round(float(t) / step)))
    return idx * step + (step * swing if idx % 2 else 0.0)


def normalize_section(
    src: Path,
    plan: dict[str, Any],
    section: dict[str, Any],
    metrics: dict[str, Any],
    offset_s: float,
) -> tuple[list[pretty_midi.Instrument], dict[str, list[pretty_midi.Instrument]], dict[str, Any], str | None]:
    pm = pretty_midi.PrettyMIDI(str(src))
    pm.instruments = [i for i in pm.instruments if i.notes]
    assignments = roles_for(pm)
    bpm = float(plan.get("bpm", 120.0))
    bars = max(1, int(section.get("bars", 4)))
    target_duration = bars * 4.0 * 60.0 / max(1.0, bpm)
    source_duration = max(0.1, float(pm.get_end_time()))
    time_scale = target_duration / source_duration
    shift = int(metrics.get("global_transpose", 0))
    swing = max(0.0, min(0.20, float(plan.get("groove", {}).get("swing", 0.0))))
    half_step = 60.0 / max(1.0, bpm) / 8.0
    section_end = offset_s + target_duration

    instruments: list[pretty_midi.Instrument] = []
    role_groups: dict[str, list[pretty_midi.Instrument]] = {r: [] for r in ROLES}
    role_index = {r: 0 for r in ROLES}

    for src_inst, role in zip(pm.instruments, assignments):
        role_index[role] += 1
        dst = pretty_midi.Instrument(
            program=int(src_inst.program),
            is_drum=bool(src_inst.is_drum),
            name=f"{role}_{str(section.get('name','section'))}_{role_index[role]}",
        )
        seen: set[tuple[int, int]] = set()
        for n in sorted(src_inst.notes, key=lambda x: (x.start, x.pitch, x.end)):
            local_start = groove_start(float(n.start) * time_scale, bpm, swing)
            start = offset_s + local_start
            if start >= section_end - 0.01:
                continue
            if src_inst.is_drum:
                duration = min(0.12, 60.0 / bpm / 4.0 * 0.85)
            else:
                raw_duration = max(half_step, (float(n.end) - float(n.start)) * time_scale)
                duration = max(half_step, round(raw_duration / half_step) * half_step)
            end = min(section_end - 0.002, start + duration)
            if end <= start + 0.012:
                continue
            pitch = int(n.pitch) if src_inst.is_drum else int(np.clip(int(n.pitch) + shift, 0, 127))
            grid = int(round(local_start / max(1e-9, 60.0 / bpm / 4.0)))
            key = (pitch, grid)
            if key in seen:
                continue
            seen.add(key)
            dst.notes.append(pretty_midi.Note(
                velocity=int(np.clip(n.velocity, 1, 127)),
                pitch=pitch,
                start=start,
                end=end,
            ))
        if dst.notes:
            instruments.append(dst)
            role_groups[role].append(dst)

    for inst in instruments:
        if inst.is_drum:
            continue
        by_pitch: dict[int, list[pretty_midi.Note]] = {}
        for n in inst.notes:
            by_pitch.setdefault(int(n.pitch), []).append(n)
        for notes in by_pitch.values():
            notes.sort(key=lambda x: x.start)
            for a, b in zip(notes, notes[1:]):
                if a.end > b.start:
                    a.end = max(a.start + 0.015, b.start - 0.002)

    root = int(plan.get("key_root", 0)) % 12
    melody_notes = sorted([n for inst in role_groups["melody"] for n in inst.notes], key=lambda n: n.start)
    motif_hint = None
    if melody_notes:
        chosen = melody_notes[: min(8, len(melody_notes))]
        degrees = [int((n.pitch - root) % 12) for n in chosen]
        local_beats = [round((n.start - offset_s) / (60.0 / bpm), 2) for n in chosen]
        motif_hint = f"lead pitch classes relative to the song key {degrees}, entering around beats {local_beats}"

    summary = {
        "source": str(src),
        "offset_s": round(offset_s, 4),
        "target_duration_s": round(target_duration, 4),
        "source_duration_s": round(source_duration, 4),
        "time_scale": round(time_scale, 5),
        "global_transpose_semitones": shift,
        "roles": {role: sum(len(i.notes) for i in group) for role, group in role_groups.items()},
        "motif_hint_for_next_section": motif_hint,
    }
    return instruments, role_groups, summary, motif_hint


def write_full_project(project: Path, bpm: float, all_instruments: list[pretty_midi.Instrument], role_groups: dict[str, list[pretty_midi.Instrument]]) -> None:
    full = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    full.instruments.extend(all_instruments)
    arrangement = project / "arrangement.mid"
    full.write(str(arrangement))

    stems = project / "midi_stems"
    shutil.rmtree(stems, ignore_errors=True)
    stems.mkdir(parents=True, exist_ok=True)
    for role, instruments in role_groups.items():
        if not instruments:
            continue
        one = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        for inst in instruments:
            copy = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            copy.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in inst.notes]
            one.instruments.append(copy)
        one.write(str(stems / f"{role}.mid"))


def generate_candidate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
):
    from anticipation.convert import events_to_midi

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    full_prompt = SYSTEM_PROMPT + prompt + " "
    enc = tokenizer(full_prompt, return_tensors="pt", padding=False)
    input_ids = enc["input_ids"]
    midi_bos = torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]], dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, midi_bos], dim=1).to("cuda")
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            do_sample=True,
            max_new_tokens=max(384, min(2046, int(max_tokens))),
            temperature=max(0.35, float(temperature)),
            top_p=max(0.50, min(1.0, float(top_p))),
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    raw = output[0, input_ids.shape[1]:].detach().cpu().tolist()
    tokens = trim_music_tokens(raw)
    return events_to_midi(tokens), len(tokens)


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 learned section-by-section composer using MIDI-LLM; no rule-note composer fallback.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--idea", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidates", type=int, default=2, help="Preferred candidate attempts per section; weak sections may get one extra rescue attempt.")
    p.add_argument("--max-tokens", type=int, default=2046)
    p.add_argument("--temperature", type=float, default=0.92)
    p.add_argument("--top-p", type=float, default=0.96)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("MIDI-LLM composer requires CUDA in the Musicm8 Colab workflow")
    ensure_anticipation()

    root = args.root.resolve()
    project = args.out.resolve()
    project.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    bpm = float(plan.get("bpm", 120.0))
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
        str(model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    candidates_root = project / "ai_composer_candidates"
    shutil.rmtree(candidates_root, ignore_errors=True)
    candidates_root.mkdir(parents=True, exist_ok=True)

    all_instruments: list[pretty_midi.Instrument] = []
    whole_roles: dict[str, list[pretty_midi.Instrument]] = {r: [] for r in ROLES}
    section_reports: list[dict[str, Any]] = []
    offset_s = 0.0
    motif_hint: str | None = None
    prompts: list[str] = []

    print("\n🧠 MIDI-LLM SECTION COMPOSER")
    print("   Each song section gets its own token budget so drums cannot starve chords/lead out of a 2046-token whole-song generation.")

    for si, section in enumerate(sections):
        name = str(section.get("name", f"section_{si+1}"))
        bars = max(1, int(section.get("bars", 4)))
        prompt = section_prompt(args.idea, plan, section, motif_hint)
        prompts.append(prompt)
        section_dir = candidates_root / f"{si+1:02d}_{name.replace(' ', '_')}"
        section_dir.mkdir(parents=True, exist_ok=True)

        preferred = max(1, min(3, int(args.candidates)))
        max_attempts = min(3, preferred + 1)
        candidates: list[dict[str, Any]] = []
        token_budget = min(int(args.max_tokens), max(700, bars * 190))
        print(f"\n🎹 Section {si+1}/{len(sections)}: {name} ({bars} bars) | token budget {token_budget}")

        for ci in range(max_attempts):
            seed = int(args.seed) + si * 104729 + ci * 7919
            try:
                midi_obj, token_count = generate_candidate(
                    model, tokenizer, prompt, token_budget, args.temperature, args.top_p, seed
                )
                path = section_dir / f"candidate_{ci+1}.mid"
                midi_obj.save(str(path))
                metrics = section_quality(path, plan, section)
                metrics.update({"candidate": ci + 1, "seed": seed, "tokens": token_count, "error": None})
                candidates.append(metrics)
                print(
                    f"   candidate {ci+1}: ready={metrics['production_ready']} score={metrics['score']:.2f} "
                    f"roles={metrics['roles']} fit={metrics['key_fit_after_global_transpose']:.2f}"
                )
                if ci + 1 >= preferred and metrics["production_ready"]:
                    break
            except Exception as exc:
                candidates.append({"candidate": ci + 1, "seed": seed, "score": -999.0, "production_ready": False, "error": str(exc)})
                print(f"   candidate {ci+1}: failed ({exc})")

        ready = [c for c in candidates if c.get("production_ready") and not c.get("error")]
        if not ready:
            details = "; ".join(
                f"c{c.get('candidate')}: {c.get('failures') or c.get('error')}" for c in candidates
            )
            raise RuntimeError(
                f"MIDI-LLM did not produce a balanced {name} section after {len(candidates)} attempts. "
                f"Musicm8 refuses to render another drum-heavy/empty-harmony song. {details}"
            )
        selected = max(ready, key=lambda c: float(c.get("score", -999.0)))
        src = Path(str(selected["path"]))
        shutil.copy2(src, project / f"arrangement_ai_{si+1:02d}_{name.replace(' ', '_')}.mid")
        instruments, role_groups, normalized, next_motif = normalize_section(src, plan, section, selected, offset_s)
        all_instruments.extend(instruments)
        for role in ROLES:
            whole_roles[role].extend(role_groups[role])
        section_reports.append({
            "index": si,
            "name": name,
            "bars": bars,
            "energy": float(section.get("energy", 0.6)),
            "prompt": prompt,
            "candidates": candidates,
            "selected_candidate": int(selected["candidate"]),
            "selected": selected,
            "normalization": normalized,
        })
        motif_hint = next_motif or motif_hint
        offset_s += bars * 4.0 * 60.0 / max(1.0, bpm)

    if not all_instruments:
        raise RuntimeError("Learned section composer created no MIDI instruments")
    write_full_project(project, bpm, all_instruments, whole_roles)
    role_totals = {role: sum(len(i.notes) for i in instruments) for role, instruments in whole_roles.items()}
    if role_totals["chords"] < max(12, int(plan.get("bars", 32)) // 2):
        raise RuntimeError(f"Final learned arrangement is still harmony-starved: {role_totals}")

    (project / "ai_composer_prompt.txt").write_text("\n\n--- SECTION ---\n\n".join(prompts) + "\n", encoding="utf-8")
    report = {
        "format": "musicm8-midi-llm-section-composer-v2",
        "composer": MODEL_ID,
        "composition_mode": "learned section-by-section multitrack composition",
        "reason": "MIDI-LLM's official generation budget is 2046 MIDI tokens. A dense 32-bar whole-song generation can spend nearly all tokens on drums/bass and leave only a handful of chord/lead notes. Musicm8 now gives each musical section its own generation budget, then stitches the AI-composed sections on one BPM/key/grid.",
        "seed": int(args.seed),
        "sections": section_reports,
        "role_notes": role_totals,
        "rule_composer_fallback": False,
        "arrangement": str(project / "arrangement.mid"),
        "midi_stems": str(project / "midi_stems"),
    }
    (project / "composition_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n✅ LEARNED SECTION-BY-SECTION COMPOSITION:", project / "arrangement.mid")
    print("Role notes:", role_totals)
    print("No rule-note composer fallback was used.")


if __name__ == "__main__":
    main()
