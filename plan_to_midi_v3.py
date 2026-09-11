from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}


def clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return default


def add_note(inst: pretty_midi.Instrument, pitch: int, start: float, end: float, velocity: int) -> None:
    if end <= start:
        return
    inst.notes.append(pretty_midi.Note(
        velocity=int(np.clip(velocity, 1, 127)),
        pitch=int(np.clip(pitch, 0, 127)),
        start=max(0.0, float(start)),
        end=max(float(start) + 0.015, float(end)),
    ))


def scale_pitch(root: int, degree0: int, octave: int, mode: str) -> int:
    scale = SCALES.get(mode, SCALES["minor"])
    oct_add, idx = divmod(int(degree0), 7)
    return 12 * (octave + 1 + oct_add) + int(root) + scale[idx]


def chord_tones(root: int, degree: int, octave: int, mode: str, sevenths: float, ninths: float, energy: float) -> list[int]:
    degree0 = max(0, min(6, int(degree) - 1))
    offsets = [0, 2, 4]
    if sevenths > 0.5 and energy > 0.42:
        offsets.append(6)
    if ninths > 0.5 and energy > 0.62:
        offsets.append(8)
    return [scale_pitch(root, degree0 + x, octave, mode) for x in offsets]


def voiced(chord: list[int], previous: list[int] | None) -> list[int]:
    candidates: list[list[int]] = []
    n = len(chord)
    for inversion in range(min(4, n)):
        c = list(chord)
        for i in range(inversion):
            c[i] += 12
        c = sorted(c)
        for shift in (-12, 0, 12):
            s = [p + shift for p in c]
            if min(s) >= 43 and max(s) <= 82:
                candidates.append(s)
    if not candidates:
        return chord
    if previous is None:
        return min(candidates, key=lambda c: abs(float(np.mean(c)) - 61.0))
    def cost(c: list[int]) -> float:
        a, b = sorted(c), sorted(previous)
        m = min(len(a), len(b))
        return sum(abs(a[i] - b[i]) for i in range(m)) + 2.0 * abs(np.mean(a) - np.mean(b))
    return min(candidates, key=cost)


def section_bars(plan: dict[str, Any]) -> list[dict[str, Any]]:
    total = int(plan.get("bars", 32))
    out: list[dict[str, Any]] = []
    for si, sec in enumerate(plan.get("sections", [])):
        count = max(1, int(sec.get("bars", 1)))
        for bi in range(count):
            out.append({
                "name": str(sec.get("name", "section")).lower(),
                "energy": clamp(sec.get("energy", 0.6), 0.05, 1.0, 0.6),
                "section_index": si,
                "bar_in_section": bi,
                "section_bars": count,
            })
    if not out:
        out = [{"name": "section", "energy": 0.65, "section_index": 0, "bar_in_section": i, "section_bars": total} for i in range(total)]
    while len(out) < total:
        x = dict(out[-1]); x["bar_in_section"] = int(x["bar_in_section"]) + 1; out.append(x)
    return out[:total]


def humanize(t: float, rng: random.Random, ms: float, strength: float = 1.0) -> float:
    if ms <= 0 or strength <= 0:
        return max(0.0, t)
    jitter = rng.gauss(0.0, ms * 0.001 * 0.33 * strength)
    return max(0.0, t + jitter)


def swing_time(step: int, step_len: float, swing: float) -> float:
    return step * step_len + (step_len * swing if step % 2 else 0.0)


DRUM_PATTERNS = {
    "uk_garage": {"kick": [0, 6, 10], "snare": [4, 12], "hat": [2, 5, 7, 10, 14], "ghost": [3, 15]},
    "house": {"kick": [0, 4, 8, 12], "snare": [4, 12], "hat": [2, 6, 10, 14], "ghost": [7, 15]},
    "techno": {"kick": [0, 4, 8, 12], "snare": [4, 12], "hat": [2, 3, 6, 7, 10, 11, 14, 15], "ghost": [5, 13]},
    "dnb": {"kick": [0, 10], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14], "ghost": [7, 15]},
    "trap": {"kick": [0, 7, 11], "snare": [8], "hat": [0, 2, 4, 6, 8, 10, 12, 14], "ghost": [13, 15]},
    "hiphop": {"kick": [0, 6, 10], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14], "ghost": [7, 15]},
    "ambient": {"kick": [0], "snare": [12], "hat": [6, 14], "ghost": []},
    "electronic": {"kick": [0, 8], "snare": [4, 12], "hat": [2, 6, 10, 14], "ghost": [7, 15]},
}


def make_drums(plan: dict[str, Any], bpm: float, rng: random.Random) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    beat, step_len, bar_len = 60.0 / bpm, 60.0 / bpm / 4.0, 60.0 / bpm * 4.0
    style = str(plan.get("style", "electronic"))
    pat = DRUM_PATTERNS.get(style, DRUM_PATTERNS["electronic"])
    groove = plan.get("groove", {})
    prod = plan.get("production", {}).get("arrangement", {})
    swing = clamp(groove.get("swing", 0), 0, 0.35, 0)
    density = clamp(groove.get("drum_density", 0.75), 0.1, 1, 0.75)
    variation = clamp(prod.get("drum_variation", 0.5), 0, 1, 0.5)
    hms = clamp(prod.get("humanize_ms", 4), 0, 24, 4)
    phrase_bars = max(2, int(prod.get("phrase_bars", 4)))

    for bar, meta in enumerate(section_bars(plan)):
        base = bar * bar_len
        energy = float(meta["energy"])
        name = meta["name"]
        intro = name == "intro"
        breakdown = "break" in name
        section_end = meta["bar_in_section"] == meta["section_bars"] - 1
        phrase_end = (bar + 1) % phrase_bars == 0
        local = clamp(density * (0.50 + 0.62 * energy), 0.12, 1.0, density)

        kicks = list(pat["kick"])
        if intro and energy < 0.4:
            kicks = kicks[: max(1, len(kicks) // 2)]
        if breakdown:
            kicks = kicks[:1]
        if phrase_end and energy > 0.70 and style in {"uk_garage", "dnb", "trap", "hiphop"} and rng.random() < variation:
            extra = 14 if 14 not in kicks else 15
            kicks.append(extra)
        for i, s in enumerate(sorted(set(kicks))):
            if i and rng.random() > min(1.0, local + 0.18):
                continue
            t = base + swing_time(s, step_len, swing)
            add_note(inst, 36, t, t + 0.095, 83 + int(31 * energy))

        for s in pat["snare"]:
            if breakdown and energy < 0.5 and s != pat["snare"][-1]:
                continue
            t = humanize(base + swing_time(s, step_len, swing), rng, hms, 0.35)
            add_note(inst, 38, t, t + 0.085, 80 + int(32 * energy))

        for j, s in enumerate(pat["hat"]):
            hat_chance = local * (0.72 if intro else 1.0) * (0.56 if breakdown else 1.0)
            if rng.random() > hat_chance:
                continue
            open_hat = (s in {6, 14}) and energy > 0.68 and rng.random() < 0.35 + 0.35 * variation
            t = humanize(base + swing_time(s, step_len, swing), rng, hms, 0.9)
            vel = 42 + int(30 * energy) + (8 if j % 2 else 0)
            add_note(inst, 46 if open_hat else 42, t, t + (0.11 if open_hat else 0.045), vel)

        if energy > 0.48:
            for s in pat.get("ghost", []):
                if rng.random() < 0.16 + 0.42 * variation * energy:
                    t = humanize(base + swing_time(s, step_len, swing), rng, hms, 1.0)
                    add_note(inst, 38, t, t + 0.04, 32 + int(16 * energy))

        if (phrase_end or section_end) and energy > 0.66 and not breakdown:
            fill_prob = 0.30 + 0.48 * variation
            if rng.random() < fill_prob:
                for j, s in enumerate((13, 14, 15)):
                    t = humanize(base + swing_time(s, step_len, swing), rng, hms, 0.6)
                    add_note(inst, 42 if j < 2 else 38, t, t + 0.045, 54 + 7 * j)
    return inst


BASS_POSITIONS = {
    "uk_garage": [0, 3, 6, 10, 13], "house": [0, 4, 8, 12], "techno": [0, 3, 6, 8, 11, 14],
    "dnb": [0, 3, 7, 10, 13], "trap": [0, 5, 9, 12], "hiphop": [0, 4, 7, 11],
    "ambient": [0, 8], "electronic": [0, 4, 10, 13],
}


def make_bass(plan: dict[str, Any], bpm: float, rng: random.Random) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Synth Bass 1"), name="bass")
    beat, step_len, bar_len = 60.0 / bpm, 60.0 / bpm / 4.0, 60.0 / bpm * 4.0
    style = str(plan.get("style", "electronic"))
    positions = BASS_POSITIONS.get(style, BASS_POSITIONS["electronic"])
    root, mode = int(plan.get("key_root", 0)), str(plan.get("mode", "minor"))
    prog = [int(x) for x in plan.get("progression_degrees", [1, 6, 3, 7])]
    groove = plan.get("groove", {})
    prod = plan.get("production", {}).get("arrangement", {})
    density = clamp(groove.get("bass_density", 0.62), 0.1, 1, 0.62)
    swing = clamp(groove.get("swing", 0), 0, 0.35, 0)
    sync = clamp(prod.get("bass_syncopation", 0.55), 0, 1, 0.55)
    hms = clamp(prod.get("humanize_ms", 4), 0, 24, 4)
    motif_mask = [True] + [rng.random() < 0.80 for _ in positions[1:]]

    for bar, meta in enumerate(section_bars(plan)):
        degree = prog[bar % len(prog)]
        next_degree = prog[(bar + 1) % len(prog)]
        root_pitch = scale_pitch(root, degree - 1, 1, mode)
        fifth = root_pitch + 7
        next_root = scale_pitch(root, next_degree - 1, 1, mode)
        energy = float(meta["energy"])
        name = meta["name"]
        base = bar * bar_len
        chance = clamp(density * (0.50 + 0.70 * energy), 0.12, 1.0, density)
        if name == "intro": chance *= 0.55
        if "break" in name: chance *= 0.42
        for j, s in enumerate(positions):
            if j and (not motif_mask[j] or rng.random() > chance):
                continue
            pitch = root_pitch
            if j in {2, 4} and rng.random() < 0.16 + 0.26 * sync * energy:
                pitch = fifth
            if j == len(positions) - 1 and energy > 0.72 and rng.random() < 0.18 + 0.30 * sync:
                # Approach the next harmony by the nearest scale tone instead of jumping randomly.
                candidates = [scale_pitch(root, degree - 2, 1, mode), root_pitch, fifth, next_root]
                pitch = min(candidates, key=lambda p: abs(p - next_root))
            start = humanize(base + swing_time(s, step_len, swing), rng, hms, 0.35)
            dur_steps = 2 if style in {"uk_garage", "dnb", "trap"} else 3
            end = min(base + bar_len - 0.01, start + step_len * dur_steps * (0.76 + 0.16 * (1 - sync)))
            add_note(inst, pitch, start, end, 74 + int(32 * energy))
    return inst


def chord_slots(style: str, rhythm: str, start: float, bar_len: float, energy: float) -> list[tuple[float, float]]:
    if rhythm == "stabs" or style == "uk_garage":
        return [(start + bar_len * 0.12, start + bar_len * 0.30), (start + bar_len * 0.56, start + bar_len * 0.73)] if energy > 0.52 else [(start + bar_len * 0.10, start + bar_len * 0.46)]
    if rhythm == "offbeat" or style == "house":
        return [(start + bar_len * x, start + bar_len * (x + 0.16)) for x in (0.12, 0.37, 0.62, 0.87)]
    if rhythm == "pulse" or style == "techno":
        return [(start + bar_len * x, start + bar_len * (x + 0.10)) for x in (0.0, 0.25, 0.50, 0.75)]
    if rhythm in {"pads", "laid_back"} or style == "ambient":
        return [(start, start + bar_len * 0.96)]
    if rhythm == "half_time" or style == "trap":
        return [(start, start + bar_len * 0.44), (start + bar_len * 0.55, start + bar_len * 0.92)]
    return [(start, start + bar_len * 0.90)]


def make_chords(plan: dict[str, Any], bpm: float) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Electric Piano 1"), name="chords")
    bar_len = 60.0 / bpm * 4.0
    root, mode = int(plan.get("key_root", 0)), str(plan.get("mode", "minor"))
    prog = [int(x) for x in plan.get("progression_degrees", [1, 6, 3, 7])]
    production = plan.get("production", {})
    harmony = production.get("harmony", {})
    arr = production.get("arrangement", {})
    sevenths = clamp(harmony.get("sevenths", 0.4), 0, 1, 0.4)
    ninths = clamp(harmony.get("ninths", 0.2), 0, 1, 0.2)
    rhythm = str(arr.get("chord_rhythm", "mixed"))
    previous: list[int] | None = None
    for bar, meta in enumerate(section_bars(plan)):
        energy = float(meta["energy"])
        degree = prog[bar % len(prog)]
        chord = voiced(chord_tones(root, degree, 3, mode, sevenths, ninths, energy), previous)
        previous = chord
        base = bar * bar_len
        slots = chord_slots(str(plan.get("style", "electronic")), rhythm, base, bar_len, energy)
        if meta["name"] == "intro" and energy < 0.35:
            slots = slots[:1]
        for a, b in slots:
            vel = 45 + int(31 * energy)
            for p in chord:
                add_note(inst, p, a, b, vel)
    return inst


def nearest_scale(root: int, mode: str, pitch: int) -> int:
    pcs = {(root + x) % 12 for x in SCALES.get(mode, SCALES["minor"])}
    candidates = [p for p in range(max(55, pitch - 4), min(85, pitch + 5)) if p % 12 in pcs]
    return min(candidates, key=lambda p: abs(p - pitch)) if candidates else pitch


def nearest_chord_tone(chord: list[int], pitch: int) -> int:
    cands = []
    for p in chord:
        for shift in (-12, 0, 12):
            q = p + shift
            if 58 <= q <= 82:
                cands.append(q)
    return min(cands, key=lambda q: abs(q - pitch)) if cands else pitch


def make_melody(plan: dict[str, Any], bpm: float, rng: random.Random) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Lead 2 (sawtooth)"), name="melody")
    beat, step_len, bar_len = 60.0 / bpm, 60.0 / bpm / 4.0, 60.0 / bpm * 4.0
    root, mode = int(plan.get("key_root", 0)), str(plan.get("mode", "minor"))
    prog = [int(x) for x in plan.get("progression_degrees", [1, 6, 3, 7])]
    groove = plan.get("groove", {})
    arr = plan.get("production", {}).get("arrangement", {})
    density = clamp(groove.get("melody_density", 0.34), 0.05, 0.95, 0.34)
    swing = clamp(groove.get("swing", 0), 0, 0.35, 0)
    rest = clamp(arr.get("melody_rest", 0.4), 0, 0.85, 0.4)
    hms = clamp(arr.get("humanize_ms", 4), 0, 24, 4)
    phrase_bars = max(2, int(arr.get("phrase_bars", 4)))

    # Two-bar motif with a singable contour. It repeats and adapts to chord tones.
    motif_steps = [0, 2, 5, 7, 10, 12, 14, 16, 18, 21, 24, 26, 28, 30]
    motif_curve = [0, 1, 2, 1, 0, -1, 1, 0, 1, 3, 2, 1, 0, -1]
    base_pitch = 67 + root % 5
    last_pitch = base_pitch

    for bar, meta in enumerate(section_bars(plan)):
        energy = float(meta["energy"])
        name = meta["name"]
        degree = prog[bar % len(prog)]
        chord = [p + 12 for p in chord_tones(root, degree, 3, mode, 0.8, 0.2, energy)[:4]]
        phrase_bar = bar % phrase_bars
        base = bar * bar_len
        section_factor = 0.40 if name == "intro" else 0.48 if "break" in name else 1.0 if name in {"drop", "final", "chorus"} else 0.72
        chance = clamp(density * (0.55 + 0.62 * energy) * section_factor, 0.05, 0.96, density)
        if rng.random() < rest * (0.62 if phrase_bar == 0 else 0.35) and name not in {"drop", "final", "chorus"}:
            continue

        for j, global_step in enumerate(motif_steps):
            motif_bar = global_step // 16
            if motif_bar != bar % 2:
                continue
            pos = global_step % 16
            strong = pos in {0, 4, 8, 12}
            if not strong and rng.random() > chance:
                continue
            target = base_pitch + motif_curve[j]
            target = nearest_scale(root, mode, target)
            if strong:
                target = nearest_chord_tone(chord, target)
            # Keep lead motion singable and avoid huge jumps.
            while target - last_pitch > 7: target -= 12
            while last_pitch - target > 7: target += 12
            target = int(np.clip(target, 60, 79))
            start = humanize(base + swing_time(pos, step_len, swing), rng, hms, 0.30)
            dur = step_len * (1.55 if strong else 0.92)
            if pos >= 14: dur = min(dur, base + bar_len - start - 0.01)
            add_note(inst, target, start, max(start + 0.08, min(base + bar_len - 0.01, start + dur)), 61 + int(29 * energy))
            last_pitch = target

        # Phrase-ending answer tone resolves to a chord tone instead of wandering.
        if (bar + 1) % phrase_bars == 0 and energy > 0.48:
            target = nearest_chord_tone(chord, last_pitch)
            start = base + bar_len * 0.82
            add_note(inst, target, start, base + bar_len * 0.98, 63 + int(24 * energy))
            last_pitch = target
    return inst


def write_project(out: Path, bpm: float, instruments: list[pretty_midi.Instrument], plan: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stems = out / "midi_stems"; stems.mkdir(parents=True, exist_ok=True)
    full = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for inst in instruments:
        full.instruments.append(inst)
        one = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        copy = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        copy.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in inst.notes]
        one.instruments.append(copy)
        one.write(str(stems / f"{inst.name}.mid"))
    full.write(str(out / "arrangement.mid"))
    summary = {
        "format": "musicm8-composition-v3",
        "bpm": bpm,
        "style": plan.get("style"),
        "key_root": plan.get("key_root"),
        "mode": plan.get("mode"),
        "tracks": {inst.name: len(inst.notes) for inst in instruments},
        "note": "Phrase-aware genre composition with stable motifs, chord-tone anchoring, voice-led extended harmony, style-specific drums/bass and controlled humanization.",
    }
    (out / "composition_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 phrase-aware genre composer v3.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    bpm = float(plan.get("bpm", 120.0))
    rng = random.Random(args.seed)
    instruments = [
        make_drums(plan, bpm, rng),
        make_bass(plan, bpm, rng),
        make_chords(plan, bpm),
        make_melody(plan, bpm, rng),
    ]
    write_project(args.out, bpm, instruments, plan)
    print("✅ Composer v3 arrangement:", args.out / "arrangement.mid")
    print("Tracks:", {i.name: len(i.notes) for i in instruments})


if __name__ == "__main__":
    main()
