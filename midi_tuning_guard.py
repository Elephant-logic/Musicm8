from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pretty_midi

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}


def scale_pc(root: int, mode: str, degree0: int) -> int:
    scale = SCALES.get(mode, SCALES["minor"])
    oct_add, idx = divmod(int(degree0), 7)
    return (int(root) + scale[idx]) % 12


def nearest_pitch_with_pc(pitch: int, pcs: set[int], lo: int, hi: int) -> int:
    pitch = int(max(lo, min(hi, pitch)))
    candidates = [p for p in range(max(lo, pitch - 12), min(hi, pitch + 12) + 1) if p % 12 in pcs]
    if not candidates:
        return pitch
    return min(candidates, key=lambda p: (abs(p - pitch), p))


def bar_index(start: float, bpm: float, bars: int) -> int:
    bar_len = 4.0 * 60.0 / max(1.0, bpm)
    return max(0, min(max(0, bars - 1), int(math.floor(max(0.0, start) / bar_len + 1e-8))))


def chord_pcs(plan: dict[str, Any], bar: int) -> tuple[set[int], int, int]:
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    prog = [int(x) for x in plan.get("progression_degrees", [1, 6, 3, 7]) if 1 <= int(x) <= 7] or [1, 6, 3, 7]
    degree0 = prog[bar % len(prog)] - 1
    triad = {scale_pc(root, mode, degree0 + x) for x in (0, 2, 4)}
    seventh = scale_pc(root, mode, degree0 + 6)
    ninth = scale_pc(root, mode, degree0 + 8)
    chord = set(triad) | {seventh, ninth}
    bass_root = scale_pc(root, mode, degree0)
    bass_fifth = scale_pc(root, mode, degree0 + 4)
    return chord, bass_root, bass_fifth


def snap_track(inst: pretty_midi.Instrument, plan: dict[str, Any], bpm: float, bars: int) -> dict[str, int]:
    name = (inst.name or "").lower()
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    scale = {(root + x) % 12 for x in SCALES.get(mode, SCALES["minor"])}
    changed = 0
    checked = 0

    for note in inst.notes:
        if inst.is_drum or name == "drums":
            continue
        checked += 1
        before = int(note.pitch)
        bar = bar_index(note.start, bpm, bars)
        chord, bass_root, bass_fifth = chord_pcs(plan, bar)

        if name == "bass":
            # Bass is deliberately conservative: chord root/fifth only. This removes
            # accidental non-diatonic fifths and reference-derived pitch clashes.
            note.pitch = nearest_pitch_with_pc(before, {bass_root, bass_fifth}, 28, 55)
        elif name == "chords":
            # Keep all chord voices inside the current diatonic harmony. Seventh/ninth
            # colours remain allowed, but no chromatic rogue notes can slip through.
            note.pitch = nearest_pitch_with_pc(before, chord, 43, 84)
        elif name == "melody":
            beat = note.start / (60.0 / max(1.0, bpm))
            strong = abs(beat - round(beat)) < 0.085
            allowed = chord if strong else scale
            note.pitch = nearest_pitch_with_pc(before, allowed, 57, 84)
        else:
            note.pitch = nearest_pitch_with_pc(before, scale, 24, 96)

        if int(note.pitch) != before:
            changed += 1

    return {"checked": checked, "changed": changed}


def write_stems(pm: pretty_midi.PrettyMIDI, midi_path: Path) -> None:
    stems = midi_path.parent / "midi_stems"
    stems.mkdir(parents=True, exist_ok=True)
    for inst in pm.instruments:
        one = pretty_midi.PrettyMIDI(initial_tempo=float(pm.estimate_tempo()) if pm.get_end_time() > 0 else 120.0)
        copy = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        copy.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in inst.notes]
        one.instruments.append(copy)
        one.write(str(stems / f"{inst.name}.mid"))


def guard(plan_path: Path, midi_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    bpm = float(plan.get("bpm", 120.0))
    bars = int(plan.get("bars", 32))
    tracks: dict[str, Any] = {}
    for inst in pm.instruments:
        tracks[inst.name or "track"] = snap_track(inst, plan, bpm, bars)
    pm.write(str(midi_path))
    write_stems(pm, midi_path)
    report = {
        "format": "musicm8-tuning-guard-v1",
        "key_root": int(plan.get("key_root", 0)) % 12,
        "mode": str(plan.get("mode", "minor")),
        "bpm": bpm,
        "tracks": tracks,
        "note": "All pitched MIDI is forced into one key. Bass is restricted to current chord root/fifth; chords to diatonic chord tones/extensions; strong melody beats to current chord tones.",
    }
    (midi_path.parent / "tuning_guard_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Remove pitch clashes from a Musicm8 arrangement before any audio rendering.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--midi", type=Path, required=True)
    args = p.parse_args()
    r = guard(args.plan, args.midi)
    print("✅ STRICT TUNING GUARD")
    print("Key:", r["key_root"], r["mode"])
    for name, info in r["tracks"].items():
        if info["checked"]:
            print(f"   {name}: checked={info['checked']} corrected={info['changed']}")


if __name__ == "__main__":
    main()
