from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pretty_midi


def _notes(path: Path) -> list[pretty_midi.Note]:
    if not path.exists():
        return []
    pm = pretty_midi.PrettyMIDI(str(path))
    return sorted([n for inst in pm.instruments for n in inst.notes], key=lambda n: (n.start, n.pitch))


def _drum_kind(pitch: int) -> str:
    if pitch in {35, 36}:
        return "kick"
    if pitch in {37, 38, 39, 40}:
        return "snare"
    if pitch in {42, 44}:
        return "hat"
    if pitch in {46}:
        return "open_hat"
    if pitch in {41, 43, 45, 47, 48, 50}:
        return "tom"
    return "perc"


def _group_chords(notes: list[pretty_midi.Note], tolerance: float = 0.035) -> list[dict[str, Any]]:
    groups: list[list[pretty_midi.Note]] = []
    for note in notes:
        if not groups or abs(note.start - groups[-1][0].start) > tolerance:
            groups.append([note])
        else:
            groups[-1].append(note)
    out = []
    for g in groups:
        pitches = sorted({int(n.pitch) for n in g})
        if not pitches:
            continue
        out.append({
            "start": float(min(n.start for n in g)),
            "end": float(max(n.end for n in g)),
            "pitches": pitches,
            "root": int(min(pitches)),
            "velocity": int(round(sum(n.velocity for n in g) / len(g))),
        })
    return out


def build_bank(library_path: Path, out_path: Path) -> dict[str, Any]:
    library = json.loads(library_path.read_text(encoding="utf-8"))
    refs_out: dict[str, Any] = {}

    for ref in library.get("references", []):
        rid = str(ref.get("id"))
        stems = {k: str(v) for k, v in ref.get("stems", {}).items()}
        midi = {k: Path(v) for k, v in ref.get("midi", {}).items()}
        entry: dict[str, Any] = {
            "name": ref.get("name"),
            "bpm": ref.get("bpm"),
            "key": ref.get("key"),
            "stems": stems,
            "drums": defaultdict(list),
            "bass": [],
            "chords": [],
            "codec_tokens": list(ref.get("codec_tokens", [])),
        }

        drums_path = midi.get("drums")
        if drums_path and "drums" in stems:
            for n in _notes(drums_path):
                kind = _drum_kind(int(n.pitch))
                tail = {"kick": 0.55, "snare": 0.50, "hat": 0.24, "open_hat": 0.65, "tom": 0.55, "perc": 0.38}[kind]
                entry["drums"][kind].append({
                    "stem": stems["drums"],
                    "start": max(0.0, float(n.start) - 0.018),
                    "end": float(n.start) + tail,
                    "pitch": int(n.pitch),
                    "velocity": int(n.velocity),
                })

        bass_path = midi.get("bass")
        if bass_path and "bass" in stems:
            for n in _notes(bass_path):
                dur = max(0.12, min(1.60, float(n.end - n.start) + 0.18))
                entry["bass"].append({
                    "stem": stems["bass"],
                    "start": max(0.0, float(n.start) - 0.015),
                    "end": float(n.start) + dur,
                    "pitch": int(n.pitch),
                    "velocity": int(n.velocity),
                })

        chords_path = midi.get("chords")
        if chords_path and "other" in stems:
            for chord in _group_chords(_notes(chords_path)):
                chord["stem"] = stems["other"]
                chord["start"] = max(0.0, float(chord["start"]) - 0.025)
                chord["end"] = max(float(chord["start"]) + 0.25, min(float(chord["end"]) + 0.12, float(chord["start"]) + 2.5))
                entry["chords"].append(chord)

        entry["drums"] = {k: v[:48] for k, v in entry["drums"].items()}
        entry["bass"] = entry["bass"][:96]
        entry["chords"] = entry["chords"][:64]
        refs_out[rid] = entry

    payload = {
        "format": "musicm8-reference-instrument-bank-v1",
        "library": str(library_path),
        "references": refs_out,
        "notes": (
            "This bank does not copy whole reference songs into new renders. It stores short, role-aligned source windows "
            "from separated stems so Musicm8 can layer reference transients/timbre under its own MIDI and synth voices. "
            "Existing EnCodec token paths remain attached for future neural residual synthesis."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Build a reusable Musicm8 instrument bank from the reference stems + aligned MIDI.")
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    payload = build_bank(args.references, args.out)
    print("✅ Reference instrument bank:", args.out)
    print("References:", len(payload["references"]))
    for rid, ref in payload["references"].items():
        drum_count = sum(len(v) for v in ref.get("drums", {}).values())
        print(f" - {rid}: drums={drum_count} bass={len(ref.get('bass', []))} chords={len(ref.get('chords', []))}")


if __name__ == "__main__":
    main()
