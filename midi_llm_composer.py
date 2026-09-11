from __future__ import annotations

import argparse
import json
import math
import os
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


def ensure_anticipation() -> None:
    try:
        import anticipation.convert  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "The learned composer needs the Anticipation MIDI decoder. "
            "Install requirements-ai.txt from the latest Musicm8 checkout."
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


def composition_prompt(idea: str, plan: dict[str, Any]) -> str:
    style = str(plan.get("style", "electronic")).replace("_", " ")
    bpm = float(plan.get("bpm", 120.0))
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    bars = int(plan.get("bars", 32))
    sections = ", ".join(
        f"{int(s.get('bars', 4))}-bar {str(s.get('name', 'section'))}"
        for s in plan.get("sections", [])
    )
    prog = "-".join(str(int(x)) for x in plan.get("progression_degrees", [1, 6, 3, 7]))
    groove = plan.get("groove", {})
    swing = float(groove.get("swing", 0.0))

    style_methods = {
        "uk garage": "authentic 2-step kick/snare pocket, swung sixteenths, shuffled hats, syncopated sub bass, short chord stabs and call-and-response hook writing",
        "house": "four-on-the-floor drums, offbeat hats, locked kick/bass relationship, economical chord rhythm, 8-bar phrases and tension/release transitions",
        "techno": "precise four-on-the-floor pulse, repeating bass ostinato, evolving motif, sparse harmony, 8-bar automation-style density changes and transition fills",
        "dnb": "breakbeat-derived kick/snare placement, fast hats, syncopated sub/reese-style bass writing, half-time harmonic space and 4/8-bar fills",
        "trap": "half-time backbeat, purposeful kick/808 interaction, hat subdivisions/rolls used as fills rather than constant clutter, sparse minor harmony and a strong top-line motif",
        "hiphop": "laid-back drum pocket, coherent bass line, sample-like chord voicings, motif repetition with small variations and 4-bar phrase answers",
        "ambient": "long harmonic motion, voice-leading, sparse percussion, slowly evolving texture and melody with deliberate silence",
        "electronic": "tight electronic drums, bass/chord interlock, recurring motif, 4/8-bar phrasing, clear builds, breakdowns and transitions",
    }
    method = style_methods.get(style, style_methods["electronic"])

    return (
        f"Compose a COMPLETE multitrack {bars}-bar {style} song at {bpm:.1f} BPM in {KEY_NAMES[root]} {mode}. "
        f"Idea/mood: {idea}. Form: {sections or f'{bars}-bar full arrangement'}. "
        f"Planned diatonic progression degrees: {prog}. Groove swing amount: {swing:.2f}. "
        "Write the WHOLE arrangement, not a short loop: drums/percussion, bass, harmony/chords and a recurring lead/hook should develop across the form. "
        f"Use real arranging practice: {method}. "
        "Keep every part rhythmically interlocked to the same meter, use repeated motifs with controlled variation, leave space, use fills only at phrase/section boundaries, "
        "and make the energy change clearly between intro, verse, build, drop/chorus, breakdown and final sections. "
        "Do not quote or recreate any existing song. Produce original music and use conventional MIDI instruments/drum channel where appropriate."
    )


def trim_music_tokens(raw_ids: list[int]) -> list[int]:
    shifted = [int(x) - LLAMA_VOCAB_SIZE for x in raw_ids]
    valid: list[int] = []
    for token in shifted:
        if 0 <= token < AMT_GPT2_BOS_ID:
            valid.append(token)
        else:
            break
    # AMT events are time/duration/instrument-pitch triples.
    valid = valid[: (len(valid) // 3) * 3]
    if len(valid) < 24:
        raise RuntimeError(f"MIDI-LLM returned too few valid music tokens ({len(valid)})")
    return valid


def group_size(inst: pretty_midi.Instrument, tolerance: float = 0.035) -> float:
    notes = sorted(inst.notes, key=lambda n: n.start)
    if not notes:
        return 0.0
    groups: list[int] = []
    current = 0
    start = None
    for n in notes:
        if start is None or abs(n.start - start) <= tolerance:
            current += 1
            if start is None:
                start = n.start
        else:
            groups.append(current)
            current = 1
            start = n.start
    groups.append(current)
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

    # Ensure a low register part is treated as bass if the model used an unusual GM program.
    if "bass" not in roles and pitched:
        idx = min(pitched, key=lambda j: np.median([n.pitch for n in pm.instruments[j].notes]))
        if np.median([n.pitch for n in pm.instruments[idx].notes]) < 62:
            roles[idx] = "bass"

    # Ensure the clearest high/mostly-monophonic line is available as a lead.
    if "melody" not in roles and pitched:
        options = [j for j in pitched if roles[j] != "bass"] or pitched
        idx = max(options, key=lambda j: np.median([n.pitch for n in pm.instruments[j].notes]) - 5.0 * group_size(pm.instruments[j]))
        roles[idx] = "melody"

    # Any remaining pitched material supplies harmony.
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
            w = max(0.05, min(2.0, n.end - n.start))
            total += w
            if (int(n.pitch) + shift) % 12 in pcs:
                good += w
    return good / total if total else 0.0


def best_global_transpose(pm: pretty_midi.PrettyMIDI, root: int, mode: str) -> tuple[int, float]:
    choices = []
    for shift in range(-6, 7):
        fit = key_fit(pm, root, mode, shift)
        choices.append((fit - 0.002 * abs(shift), fit, shift))
    _, fit, shift = max(choices)
    return int(shift), float(fit)


def candidate_metrics(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    pm = pretty_midi.PrettyMIDI(str(path))
    pm.instruments = [i for i in pm.instruments if i.notes]
    duration = float(pm.get_end_time())
    target = int(plan.get("bars", 32)) * 4.0 * 60.0 / max(1.0, float(plan.get("bpm", 120.0)))
    roles = roles_for(pm)
    role_set = set(roles)
    note_count = sum(len(i.notes) for i in pm.instruments)
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    shift, fit = best_global_transpose(pm, root, mode)

    ratio = max(1e-6, duration / max(target, 1e-6))
    duration_score = math.exp(-1.45 * abs(math.log(ratio)))
    coverage = len(role_set & {"drums", "bass", "chords", "melody"}) / 4.0
    density_score = min(1.0, note_count / max(120.0, float(plan.get("bars", 32)) * 8.0))

    # Reward meaningful phrase-density changes rather than one unchanging loop.
    bins = 8
    counts = np.zeros(bins, dtype=np.float32)
    if duration > 0:
        for inst in pm.instruments:
            for n in inst.notes:
                counts[min(bins - 1, int((n.start / duration) * bins))] += 1
    variation = float(np.std(counts) / (np.mean(counts) + 1e-6)) if np.mean(counts) > 0 else 0.0
    development = min(1.0, variation / 0.30)

    style = str(plan.get("style", "electronic"))
    mandatory = {"bass", "chords", "melody"} if style == "ambient" else {"drums", "bass", "chords", "melody"}
    complete = mandatory.issubset(role_set) and note_count >= max(64, int(plan.get("bars", 32)) * 4)
    score = 2.4 * coverage + 1.7 * duration_score + 1.0 * fit + 0.7 * density_score + 0.6 * development
    if not complete:
        score -= 1.8
    return {
        "path": str(path),
        "score": round(score, 5),
        "duration_s": round(duration, 3),
        "target_duration_s": round(target, 3),
        "duration_ratio": round(ratio, 4),
        "notes": note_count,
        "instruments": len(pm.instruments),
        "roles": sorted(role_set),
        "role_assignments": roles,
        "key_fit_after_global_transpose": round(fit, 4),
        "global_transpose": shift,
        "phrase_density_variation": round(variation, 4),
        "complete": complete,
    }


def groove_start(t: float, bpm: float, swing: float) -> float:
    step = 60.0 / bpm / 4.0
    idx = max(0, int(round(float(t) / step)))
    return idx * step + (step * swing if idx % 2 else 0.0)


def normalize_selected(src: Path, project: Path, plan: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    pm = pretty_midi.PrettyMIDI(str(src))
    pm.instruments = [i for i in pm.instruments if i.notes]
    assignments = roles_for(pm)
    bpm = float(plan.get("bpm", 120.0))
    bars = int(plan.get("bars", 32))
    target_duration = bars * 4.0 * 60.0 / max(1.0, bpm)
    source_duration = max(0.1, float(pm.get_end_time()))
    time_scale = target_duration / source_duration
    shift = int(metrics.get("global_transpose", 0))
    swing = max(0.0, min(0.20, float(plan.get("groove", {}).get("swing", 0.0))))
    half_step = 60.0 / bpm / 8.0

    full = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    role_groups: dict[str, list[pretty_midi.Instrument]] = {r: [] for r in ("drums", "bass", "chords", "melody")}
    role_counts = {r: 0 for r in role_groups}
    timing_corrections = 0

    for src_inst, role in zip(pm.instruments, assignments):
        role_counts[role] += 1
        dst = pretty_midi.Instrument(
            program=int(src_inst.program),
            is_drum=bool(src_inst.is_drum),
            name=f"{role}_{role_counts[role]}",
        )
        seen: set[tuple[int, int]] = set()
        for n in sorted(src_inst.notes, key=lambda x: (x.start, x.pitch)):
            scaled_start = float(n.start) * time_scale
            scaled_end = float(n.end) * time_scale
            start = groove_start(scaled_start, bpm, swing)
            duration = max(half_step, scaled_end - scaled_start)
            duration = max(half_step, round(duration / half_step) * half_step)
            end = min(target_duration - 0.002, start + duration)
            if start >= target_duration - 0.01 or end <= start + 0.012:
                continue
            pitch = int(n.pitch) if src_inst.is_drum else int(np.clip(int(n.pitch) + shift, 0, 127))
            grid_idx = int(round(start / max(1e-9, 60.0 / bpm / 4.0)))
            key = (pitch, grid_idx)
            if key in seen:
                continue
            seen.add(key)
            if abs(start - scaled_start) > 0.001:
                timing_corrections += 1
            dst.notes.append(pretty_midi.Note(
                velocity=int(np.clip(n.velocity, 1, 127)),
                pitch=pitch,
                start=start,
                end=end,
            ))
        if dst.notes:
            full.instruments.append(dst)
            role_groups[role].append(dst)

    # Trim same-pitch overlaps to avoid doubled/stuck attacks after quantisation.
    for inst in full.instruments:
        if inst.is_drum:
            continue
        by_pitch: dict[int, list[pretty_midi.Note]] = {}
        for n in inst.notes:
            by_pitch.setdefault(int(n.pitch), []).append(n)
        for notes in by_pitch.values():
            notes.sort(key=lambda x: x.start)
            for a, b in zip(notes, notes[1:]):
                if a.end > b.start:
                    a.end = max(a.start + 0.02, b.start - 0.002)

    if not full.instruments:
        raise RuntimeError("Selected MIDI-LLM candidate became empty after normalisation")

    arrangement = project / "arrangement.mid"
    full.write(str(arrangement))
    stems = project / "midi_stems"
    shutil.rmtree(stems, ignore_errors=True)
    stems.mkdir(parents=True, exist_ok=True)
    for role, instruments in role_groups.items():
        one = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        for inst in instruments:
            copy = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            copy.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in inst.notes]
            one.instruments.append(copy)
        if one.instruments:
            one.write(str(stems / f"{role}.mid"))

    return {
        "arrangement": str(arrangement),
        "target_duration_s": round(target_duration, 4),
        "source_duration_s": round(source_duration, 4),
        "time_scale": round(time_scale, 5),
        "global_transpose_semitones": shift,
        "global_swing": round(swing, 4),
        "timing_corrections": timing_corrections,
        "roles": {role: sum(len(i.notes) for i in instruments) for role, instruments in role_groups.items()},
        "method": "learned whole-track MIDI -> one global transpose -> one shared 1/16 swing grid; intervals/harmony are preserved rather than snapping individual notes to a hand-written progression",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 learned whole-track composer using the official 2026 MIDI-LLM text-to-MIDI model.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--idea", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidates", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=2046)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.98)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("MIDI-LLM composer requires CUDA in the Musicm8 Colab workflow")
    ensure_anticipation()
    from anticipation.convert import events_to_midi

    root = args.root.resolve()
    project = args.out.resolve()
    project.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    prompt = composition_prompt(args.idea, plan)
    (project / "ai_composer_prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    model_dir = ensure_model(root)
    cache_dir = Path("/content/musicm8_midi_llm_hf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
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

    full_prompt = SYSTEM_PROMPT + prompt + " "
    enc = tokenizer(full_prompt, return_tensors="pt", padding=False)
    input_ids = enc["input_ids"]
    midi_bos = torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]], dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, midi_bos], dim=1).to("cuda")

    candidates_dir = project / "ai_composer_candidates"
    shutil.rmtree(candidates_dir, ignore_errors=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []

    print("\n🧠 MIDI-LLM is composing the entire multitrack score — no rule-composer fallback")
    for i in range(max(1, min(4, int(args.candidates)))):
        seed = int(args.seed) + i * 7919
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    max_new_tokens=max(256, min(2046, int(args.max_tokens))),
                    temperature=max(0.35, float(args.temperature)),
                    top_p=max(0.50, min(1.0, float(args.top_p))),
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            raw = output[0, input_ids.shape[1]:].detach().cpu().tolist()
            tokens = trim_music_tokens(raw)
            midi_obj = events_to_midi(tokens)
            path = candidates_dir / f"candidate_{i+1}.mid"
            midi_obj.save(str(path))
            metrics = candidate_metrics(path, plan)
            metrics.update({"candidate": i + 1, "seed": seed, "tokens": len(tokens), "error": None})
            reports.append(metrics)
            print(f"  candidate {i+1}: score={metrics['score']:.3f} roles={metrics['roles']} notes={metrics['notes']} duration={metrics['duration_s']}s")
        except Exception as exc:
            reports.append({"candidate": i + 1, "seed": seed, "score": -999.0, "complete": False, "error": str(exc)})
            print(f"  candidate {i+1}: failed ({exc})")

    valid = [r for r in reports if not r.get("error") and Path(str(r.get("path", ""))).exists()]
    if not valid:
        raise RuntimeError("MIDI-LLM did not produce any usable whole-track candidate. Musicm8 will not silently substitute the old rule composer.")
    selected = max(valid, key=lambda r: float(r.get("score", -999)))
    src = Path(str(selected["path"]))
    shutil.copy2(src, project / "arrangement_ai_raw.mid")
    normalized = normalize_selected(src, project, plan, selected)

    report = {
        "format": "musicm8-midi-llm-composer-v1",
        "composer": "slseanwu/MIDI-LLM_Llama-3.2-1B",
        "model_family": "Llama-3.2-1B adapted for text-to-MIDI",
        "composition_mode": "learned full multitrack score",
        "prompt": prompt,
        "seed": int(args.seed),
        "candidates": reports,
        "selected_candidate": int(selected["candidate"]),
        "selected": selected,
        "normalization": normalized,
        "rule_composer_fallback": False,
        "note": "The notes, rhythms, instruments and multitrack relationships come from the learned MIDI-LLM model. Musicm8 only applies a single global key transposition, duration normalization and one shared groove grid before rendering.",
    }
    (project / "composition_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n✅ LEARNED WHOLE-TRACK COMPOSITION:", project / "arrangement.mid")
    print("Selected candidate:", selected["candidate"], "score:", selected["score"])
    print("Role notes:", normalized["roles"])
    print("Global key transpose:", normalized["global_transpose_semitones"], "semitones")
    print("Shared swing:", normalized["global_swing"])


if __name__ == "__main__":
    main()
