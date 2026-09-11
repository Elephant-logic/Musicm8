from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pretty_midi


def quantize(t: float, step: float) -> float:
    return max(0.0, round(float(t) / step) * step)


def write_stems(pm: pretty_midi.PrettyMIDI, midi_path: Path, bpm: float) -> None:
    stems = midi_path.parent / "midi_stems"
    stems.mkdir(parents=True, exist_ok=True)
    for inst in pm.instruments:
        one = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        copy = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        copy.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in inst.notes]
        one.instruments.append(copy)
        one.write(str(stems / f"{inst.name}.mid"))


def guard(plan_path: Path, midi_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    bpm = float(plan.get("bpm", 120.0))
    if bpm <= 0:
        bpm = 120.0
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # Foundation mode: all musical events share one exact 1/16-note clock.
    # We deliberately remove random millisecond jitter and per-track timing drift.
    step = 60.0 / bpm / 4.0
    half_step = step * 0.5
    report: dict[str, Any] = {
        "format": "musicm8-timing-grid-v1",
        "bpm": bpm,
        "grid": "1/16",
        "step_seconds": step,
        "tracks": {},
        "note": "All track starts are locked to one global 1/16 clock. Random humanize jitter is removed. Note lengths are quantized independently and duplicate same-pitch events are removed.",
    }

    for inst in pm.instruments:
        name = (inst.name or "track").lower()
        corrected = 0
        max_shift_ms = 0.0
        cleaned: list[pretty_midi.Note] = []
        seen: set[tuple[int, int]] = set()

        for note in sorted(inst.notes, key=lambda n: (n.start, n.pitch, n.end)):
            old_start = float(note.start)
            old_end = float(note.end)
            start = quantize(old_start, step)

            if inst.is_drum or name == "drums":
                # Drum renderers use only the onset, so keep an unambiguous short event.
                if int(note.pitch) == 46:
                    end = start + min(0.18, step * 1.5)
                else:
                    end = start + min(0.10, step * 0.8)
            else:
                length = max(half_step, old_end - old_start)
                q_len = max(half_step, round(length / half_step) * half_step)
                end = start + q_len

            shift_ms = abs(start - old_start) * 1000.0
            max_shift_ms = max(max_shift_ms, shift_ms)
            if shift_ms > 0.05 or abs(end - old_end) * 1000.0 > 0.05:
                corrected += 1

            key = (int(note.pitch), int(round(start / max(step, 1e-9))))
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=float(start),
                end=float(max(start + 0.015, end)),
            ))

        # Prevent overlapping repetitions of the same pitched note from smearing attacks.
        if not (inst.is_drum or name == "drums"):
            by_pitch: dict[int, list[pretty_midi.Note]] = {}
            for n in cleaned:
                by_pitch.setdefault(int(n.pitch), []).append(n)
            for notes in by_pitch.values():
                notes.sort(key=lambda n: n.start)
                for a, b in zip(notes, notes[1:]):
                    if a.end > b.start:
                        a.end = max(a.start + min(half_step, 0.05), b.start - 0.002)

        inst.notes = sorted(cleaned, key=lambda n: (n.start, n.pitch))
        report["tracks"][name] = {
            "events": len(inst.notes),
            "corrected": corrected,
            "max_start_correction_ms": round(max_shift_ms, 3),
        }

    pm.write(str(midi_path))
    write_stems(pm, midi_path, bpm)
    out = midi_path.parent / "timing_grid_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Lock every Musicm8 MIDI track to one exact global timing grid before audio rendering.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--midi", type=Path, required=True)
    args = p.parse_args()
    r = guard(args.plan, args.midi)
    print("✅ STRICT TIMING GRID")
    print(f"BPM: {r['bpm']:.3f} | grid: {r['grid']} | step: {r['step_seconds']:.6f}s")
    for name, info in r["tracks"].items():
        print(f"   {name}: events={info['events']} corrected={info['corrected']} max_shift={info['max_start_correction_ms']} ms")


if __name__ == "__main__":
    main()
