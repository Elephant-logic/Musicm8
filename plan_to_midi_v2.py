from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi

SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def add_note(inst, pitch, start, end, velocity):
    if end <= start:
        return
    inst.notes.append(
        pretty_midi.Note(
            velocity=int(np.clip(velocity, 1, 127)),
            pitch=int(np.clip(pitch, 0, 127)),
            start=max(0.0, float(start)),
            end=max(float(start) + 0.01, float(end)),
        )
    )


def scale_pitch(root: int, degree0: int, octave: int, mode: str) -> int:
    scale = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["minor"])
    degree0 = int(degree0)
    oct_add, i = divmod(degree0, 7)
    return 12 * (octave + 1 + oct_add) + int(root) + scale[i]


def triad(root: int, degree: int, octave: int, mode: str) -> list[int]:
    i = max(0, min(6, int(degree) - 1))
    return [scale_pitch(root, i + x, octave, mode) for x in (0, 2, 4)]


def section_bars(plan: dict[str, Any]) -> list[dict[str, Any]]:
    bars = int(plan["bars"])
    out: list[dict[str, Any]] = []
    for sec_i, sec in enumerate(plan.get("sections", [])):
        count = max(1, int(sec.get("bars", 1)))
        for local in range(count):
            out.append(
                {
                    "name": str(sec.get("name", "section")).lower(),
                    "energy": clamp(sec.get("energy", 0.6), 0.05, 1.0),
                    "section_index": sec_i,
                    "bar_in_section": local,
                    "section_bars": count,
                }
            )
    if not out:
        out = [
            {"name": "section", "energy": 0.65, "section_index": 0, "bar_in_section": i, "section_bars": bars}
            for i in range(bars)
        ]
    while len(out) < bars:
        extra = dict(out[-1])
        extra["bar_in_section"] = int(extra.get("bar_in_section", 0)) + 1
        out.append(extra)
    return out[:bars]


def timing(step: int, step_len: float, swing: float) -> float:
    return step * step_len + (step_len * swing if step % 2 else 0.0)


def style_patterns(style: str) -> dict[str, list[int]]:
    patterns = {
        "house": {"kick": [0, 4, 8, 12], "snare": [4, 12], "hat": [2, 6, 10, 14]},
        "techno": {"kick": [0, 4, 8, 12], "snare": [4, 12], "hat": [2, 3, 6, 7, 10, 11, 14, 15]},
        "uk_garage": {"kick": [0, 6, 10], "snare": [4, 12], "hat": [2, 5, 7, 10, 14]},
        "dnb": {"kick": [0, 10], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14]},
        "trap": {"kick": [0, 7, 11], "snare": [8], "hat": list(range(0, 16, 2))},
        "hiphop": {"kick": [0, 6, 10], "snare": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14]},
        "ambient": {"kick": [0], "snare": [12], "hat": [6, 14]},
        "electronic": {"kick": [0, 8], "snare": [4, 12], "hat": [2, 6, 10, 14]},
    }
    return patterns.get(style, patterns["electronic"])


def make_drums(plan: dict[str, Any], bpm: float, rng: random.Random) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    beat = 60.0 / bpm
    step_len = beat / 4
    bar_len = beat * 4
    pat = style_patterns(plan["style"])
    groove = plan.get("groove", {})
    swing = clamp(groove.get("swing", 0), 0, 0.35)
    density = clamp(groove.get("drum_density", 0.75), 0.1, 1)

    for bar, meta in enumerate(section_bars(plan)):
        energy = meta["energy"]
        base = bar * bar_len
        local = clamp(density * (0.56 + 0.58 * energy), 0.1, 1)
        section_name = meta["name"]
        phrase_end = (bar + 1) % 4 == 0
        section_end = meta["bar_in_section"] == meta["section_bars"] - 1

        # Keep the backbone stable. Variation is added around it rather than deleting it randomly.
        for s in pat["kick"]:
            keep = s == pat["kick"][0] or rng.random() < min(1.0, local + 0.22)
            if keep:
                t = base + timing(s, step_len, swing)
                add_note(inst, 36, t, t + 0.10, 80 + int(34 * energy))

        for s in pat["snare"]:
            t = base + timing(s, step_len, swing)
            add_note(inst, 38, t, t + 0.09, 78 + int(34 * energy))

        for s in pat["hat"]:
            if rng.random() < local:
                pitch = 46 if s in {6, 14} and energy > 0.74 else 42
                t = base + timing(s, step_len, swing)
                vel = 40 + int(36 * energy) + (6 if s % 4 == 2 else 0)
                add_note(inst, pitch, t, t + (0.10 if pitch == 46 else 0.05), vel)

        # Genre-specific ghost notes create motion but remain phrase-aware.
        if plan["style"] in {"uk_garage", "dnb", "hiphop"} and energy > 0.55:
            ghosts = [15] if bar % 2 == 0 else [7]
            for s in ghosts:
                if rng.random() < 0.35 + 0.25 * energy:
                    t = base + timing(s, step_len, swing)
                    add_note(inst, 38, t, t + 0.045, 36 + int(18 * energy))

        # Deliberate fills only at phrase/section boundaries.
        if energy > 0.72 and (phrase_end or section_end) and section_name not in {"intro", "breakdown"}:
            for j, s in enumerate((13, 14, 15)):
                t = base + timing(s, step_len, swing)
                add_note(inst, 42 if j < 2 else 38, t, t + 0.05, 56 + 8 * j)

    return inst


def bass_positions(style: str) -> list[int]:
    return {
        "uk_garage": [0, 3, 6, 10, 13],
        "house": [0, 4, 8, 12],
        "techno": [0, 3, 6, 8, 11, 14],
        "dnb": [0, 3, 7, 10, 13],
        "trap": [0, 5, 9, 12],
        "hiphop": [0, 4, 7, 11],
        "ambient": [0, 8],
        "electronic": [0, 4, 10, 13],
    }.get(style, [0, 4, 10, 13])


def make_bass(plan: dict[str, Any], bpm: float, rng: random.Random) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Synth Bass 1"), name="bass")
    beat = 60.0 / bpm
    step_len = beat / 4
    bar_len = beat * 4
    root = int(plan["key_root"])
    mode = plan["mode"]
    progression = [int(x) for x in plan["progression_degrees"]]
    density = clamp(plan.get("groove", {}).get("bass_density", 0.6), 0.1, 1)
    swing = clamp(plan.get("groove", {}).get("swing", 0), 0, 0.35)
    positions = bass_positions(plan["style"])

    # A stable two-bar articulation mask gives a bass line an identity.
    keep_mask = [True] + [rng.random() < 0.82 for _ in positions[1:]]
    for bar, meta in enumerate(section_bars(plan)):
        energy = meta["energy"]
        degree = progression[bar % len(progression)]
        root_pitch = scale_pitch(root, degree - 1, 1, mode)
        fifth_pitch = root_pitch + 7
        base = bar * bar_len
        for j, pos in enumerate(positions):
            chance = clamp(density * (0.48 + 0.72 * energy), 0.15, 1)
            if j and (not keep_mask[j] or rng.random() > chance):
                continue
            pitch = root_pitch
            # Fifths and octaves are supporting colors, not arbitrary scale jumps.
            if j in {2, 4} and energy > 0.60 and rng.random() < 0.32:
                pitch = fifth_pitch
            elif j == len(positions) - 1 and energy > 0.80 and rng.random() < 0.28:
                pitch = root_pitch + 12
            start = base + timing(pos, step_len, swing)
            dur_steps = 2 if plan["style"] in {"uk_garage", "dnb", "trap"} else 3
            add_note(inst, pitch, start, min(base + bar_len, start + dur_steps * step_len * 0.90), 72 + int(34 * energy))

    return inst


def _voiced_triad(root: int, degree: int, mode: str, previous: list[int] | None) -> list[int]:
    base = triad(root, degree, 3, mode)
    candidates: list[list[int]] = []
    for inversion in range(3):
        c = list(base)
        for i in range(inversion):
            c[i] += 12
        c = sorted(c)
        for shift in (-12, 0, 12):
            shifted = [p + shift for p in c]
            if min(shifted) >= 42 and max(shifted) <= 76:
                candidates.append(shifted)
    if not candidates:
        return base
    if previous is None:
        return min(candidates, key=lambda c: abs(np.mean(c) - 58))
    return min(candidates, key=lambda c: sum(abs(a - b) for a, b in zip(sorted(c), sorted(previous))))


def make_chords(plan: dict[str, Any], bpm: float) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Electric Piano 1"), name="chords")
    beat = 60.0 / bpm
    bar_len = beat * 4
    root = int(plan["key_root"])
    mode = plan["mode"]
    prog = [int(x) for x in plan["progression_degrees"]]
    density = clamp(plan.get("groove", {}).get("chord_density", 0.55), 0.1, 1)
    previous: list[int] | None = None

    for bar, meta in enumerate(section_bars(plan)):
        energy = meta["energy"]
        degree = prog[bar % len(prog)]
        chord = _voiced_triad(root, degree, mode, previous)
        previous = chord
        start = bar * bar_len
        section_name = meta["name"]
        if density > 0.64 and energy > 0.58 and section_name not in {"intro", "breakdown"}:
            slots = [(start, start + bar_len * 0.43), (start + bar_len * 0.50, start + bar_len * 0.94)]
        else:
            slots = [(start, start + bar_len * 0.92)]
        for a, b in slots:
            for pitch in chord:
                add_note(inst, pitch, a, b, 46 + int(33 * energy))
    return inst


def _nearest_scale_pitch(root: int, mode: str, midi_pitch: int) -> int:
    pcs = {(root + x) % 12 for x in SCALE_INTERVALS[mode]}
    candidates = [p for p in range(max(36, midi_pitch - 4), min(96, midi_pitch + 5)) if p % 12 in pcs]
    return min(candidates, key=lambda p: abs(p - midi_pitch)) if candidates else midi_pitch


def make_melody(plan: dict[str, Any], bpm: float, rng: random.Random) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Lead 2 (sawtooth)"), name="melody")
    beat = 60.0 / bpm
    step_len = beat / 4
    bar_len = beat * 4
    root = int(plan["key_root"])
    mode = plan["mode"]
    prog = [int(x) for x in plan["progression_degrees"]]
    density = clamp(plan.get("groove", {}).get("melody_density", 0.4), 0.05, 0.95)
    swing = clamp(plan.get("groove", {}).get("swing", 0), 0, 0.35)

    # Repeated rhythmic/contour motif. Strong beats use chord tones; weak beats use nearby scale tones.
    rhythm = [0, 2, 5, 7, 10, 12, 14]
    contour = [0, 1, 2, 1, 0, 2, 1]
    weak_shift = [0, 1, -1, 1, 0, -1, 1]
    last_pitch = 72 + root

    for bar, meta in enumerate(section_bars(plan)):
        energy = meta["energy"]
        name = meta["name"]
        degree = prog[bar % len(prog)]
        chord = [p + 12 for p in triad(root, degree, 3, mode)]
        base = bar * bar_len

        # Intros/breakdowns breathe; drops/finals get the full motif.
        section_factor = 0.48 if name == "intro" else 0.55 if "break" in name else 1.08 if name in {"drop", "final"} else 0.82
        chance = clamp(density * (0.38 + 0.72 * energy) * section_factor, 0.08, 0.98)
        if energy < 0.34 and bar % 2:
            continue

        for j, pos in enumerate(rhythm):
            strong = pos in {0, 5, 10, 14}
            if j and rng.random() > chance * (1.08 if strong else 0.86):
                continue

            chord_pitch = chord[contour[j] % len(chord)]
            if strong:
                pitch = chord_pitch
            else:
                pitch = _nearest_scale_pitch(root, mode, int(round((last_pitch + chord_pitch) / 2)) + weak_shift[j])

            # Keep melodic motion singable and resolve phrase endings.
            while pitch - last_pitch > 7:
                pitch -= 12
            while last_pitch - pitch > 7:
                pitch += 12
            if (bar + 1) % 4 == 0 and j == len(rhythm) - 1:
                pitch = chord[0]
            if name in {"drop", "final"} and energy > 0.85 and bar % 4 == 2 and j in {1, 4}:
                pitch += 12

            start = base + timing(pos, step_len, swing)
            dur = step_len * (1.55 if strong else 0.78)
            add_note(inst, pitch, start, min(base + bar_len, start + dur), 48 + int(38 * energy) + (5 if strong else 0))
            last_pitch = pitch

    return inst


def write_project(plan: dict[str, Any], out: Path, seed: int) -> None:
    bpm = float(plan["bpm"])
    rng = random.Random(seed)
    tracks = {
        "drums": make_drums(plan, bpm, rng),
        "bass": make_bass(plan, bpm, rng),
        "chords": make_chords(plan, bpm),
        "melody": make_melody(plan, bpm, rng),
    }
    out.mkdir(parents=True, exist_ok=True)
    stems = out / "midi_stems"
    stems.mkdir(parents=True, exist_ok=True)
    full = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for name, inst in tracks.items():
        if not inst.notes:
            continue
        full.instruments.append(inst)
        one = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        one.instruments.append(inst)
        one.write(str(stems / f"{name}.mid"))
    full.write(str(out / "arrangement.mid"))


def main() -> None:
    p = argparse.ArgumentParser(description="Chord-aware hierarchical Musicm8 MIDI composer.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    write_project(plan, args.out, args.seed)
    print(f"✅ Hierarchical arrangement MIDI: {args.out / 'arrangement.mid'}")
    print(f"✅ MIDI stems: {args.out / 'midi_stems'}")


if __name__ == "__main__":
    main()
