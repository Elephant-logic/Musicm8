from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", str(text))


def syllable_weight(word: str) -> float:
    w = re.sub(r"[^a-z]", "", word.lower())
    groups = re.findall(r"[aeiouy]+", w)
    count = max(1, len(groups))
    if w.endswith("e") and count > 1 and not w.endswith(("le", "ye")):
        count -= 1
    return 1.0 + 0.28 * max(0, count - 1)


def section_windows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    bpm = float(plan.get("bpm", 120.0))
    bar_s = 4.0 * 60.0 / max(1.0, bpm)
    cursor = 0.0
    out = []
    for i, sec in enumerate(plan.get("sections", [])):
        bars = max(1, int(sec.get("bars", 1)))
        end = cursor + bars * bar_s
        out.append({
            "index": i,
            "name": str(sec.get("name", "section")),
            "start": cursor,
            "end": end,
            "bars": bars,
            "energy": float(sec.get("energy", 0.6)),
        })
        cursor = end
    return out


def scale_pcs(plan: dict[str, Any]) -> set[int]:
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    return {(root + x) % 12 for x in SCALES.get(mode, SCALES["minor"])}


def nearest_with_pcs(target: int, pcs: set[int], lo: int = 57, hi: int = 81) -> int:
    pool = [p for p in range(lo, hi + 1) if p % 12 in pcs]
    if not pool:
        return int(np.clip(target, lo, hi))
    return min(pool, key=lambda p: (abs(p - target), p))


def pitched_notes(pm: pretty_midi.PrettyMIDI | None) -> list[pretty_midi.Note]:
    if pm is None:
        return []
    return [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]


def active_pcs(notes: list[pretty_midi.Note], t: float, search_s: float = 0.55) -> set[int]:
    active = [n for n in notes if n.start - 0.03 <= t <= n.end + 0.03]
    if not active and notes:
        near = sorted(notes, key=lambda n: min(abs(t - n.start), abs(t - n.end)))[:8]
        active = [n for n in near if min(abs(t - n.start), abs(t - n.end)) <= search_s]
    return {int(n.pitch) % 12 for n in active}


def melody_target(notes: list[pretty_midi.Note], t: float, default: int) -> int:
    active = [n.pitch for n in notes if n.start - 0.04 <= t <= n.end + 0.04]
    if active:
        return int(round(float(np.median(active))))
    if notes:
        near = min(notes, key=lambda n: min(abs(t - n.start), abs(t - n.end)))
        if min(abs(t - near.start), abs(t - near.end)) <= 0.8:
            return int(near.pitch)
    return default


def choose_pitch(
    plan: dict[str, Any],
    harmony_notes: list[pretty_midi.Note],
    melody_notes: list[pretty_midi.Note],
    t: float,
    contour: int,
    previous: int | None,
    strong: bool,
) -> int:
    scale = scale_pcs(plan)
    harmony = active_pcs(harmony_notes, t)
    fallback = 66 + contour
    target = melody_target(melody_notes, t, fallback) + (0 if melody_notes else contour)
    allowed = harmony if strong and harmony else scale
    pitch = nearest_with_pcs(target, allowed)
    if previous is not None:
        while pitch - previous > 7:
            pitch -= 12
        while previous - pitch > 7:
            pitch += 12
        pitch = int(np.clip(pitch, 57, 81))
        if pitch % 12 not in allowed:
            pitch = nearest_with_pcs(pitch, allowed)
    return pitch


def build(
    plan: dict[str, Any],
    lyrics: dict[str, Any],
    harmony_pm: pretty_midi.PrettyMIDI | None,
    melody_pm: pretty_midi.PrettyMIDI | None,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any]]:
    bpm = float(plan.get("bpm", 120.0))
    beat = 60.0 / max(1.0, bpm)
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Voice Oohs"), name="vocal_melody")
    harmony_notes = pitched_notes(harmony_pm)
    melody_notes = pitched_notes(melody_pm)
    lyric_sections = lyrics.get("sections", []) if isinstance(lyrics.get("sections", []), list) else []
    previous: int | None = None
    scored_sections = []
    contour_bank = [0, 2, 1, 3, 2, 0, -1, 1, 0, -2, 0, 2, 1, -1, 0, 1]

    for si, sec in enumerate(section_windows(plan)):
        lines = []
        if si < len(lyric_sections) and isinstance(lyric_sections[si], dict):
            lines = [str(x).strip() for x in lyric_sections[si].get("lines", []) if str(x).strip()]
        notes_out: list[dict[str, Any]] = []
        if lines:
            margin = min(0.28 * beat, max(0.04, (sec["end"] - sec["start"]) * 0.02))
            a = sec["start"] + margin
            b = sec["end"] - margin
            line_weights = [max(2.0, sum(syllable_weight(w) for w in words(line))) for line in lines]
            total_lw = sum(line_weights) or float(len(lines))
            cursor = a
            for li, (line, lw) in enumerate(zip(lines, line_weights)):
                line_end = b if li == len(lines) - 1 else cursor + (b - a) * lw / total_lw
                line_end = min(b, line_end)
                gap = min(0.28 * beat, max(0.05, (line_end - cursor) * 0.06))
                phrase_end = max(cursor + 0.12, line_end - (gap if li < len(lines) - 1 else 0.0))
                ws = words(line)
                weights = [syllable_weight(w) for w in ws]
                gap_w = 0.18
                total = sum(weights) + gap_w * max(0, len(ws) - 1)
                unit = max(0.01, (phrase_end - cursor) / max(total, 1e-6))
                pos = cursor
                for wi, (word, ww) in enumerate(zip(ws, weights)):
                    remaining = len(ws) - wi - 1
                    dur = max(0.12, unit * ww)
                    latest = phrase_end - remaining * 0.11
                    end = max(pos + 0.09, min(latest, pos + dur))
                    strong = wi == 0 or wi == len(ws) - 1 or wi % 4 == 0
                    contour = contour_bank[(wi + li * 3 + si * 2) % len(contour_bank)]
                    pitch = choose_pitch(plan, harmony_notes, melody_notes, pos, contour, previous, strong)
                    row = {
                        "start": round(pos, 5), "end": round(end, 5), "duration": round(end - pos, 5),
                        "pitch": int(pitch), "velocity": 84 if strong else 76,
                        "word": word, "syllable": word, "line_index": li, "word_index": wi, "note_type": 2,
                        "harmony_pitch_classes": sorted(active_pcs(harmony_notes, pos)),
                    }
                    notes_out.append(row)
                    inst.notes.append(pretty_midi.Note(row["velocity"], row["pitch"], row["start"], row["end"]))
                    previous = pitch
                    pos = end
                    if wi < len(ws) - 1:
                        pos = min(phrase_end - remaining * 0.09, pos + max(0.025, unit * gap_w))
                cursor = line_end
        scored_sections.append({**sec, "lines": lines, "notes": notes_out})

    pm.instruments.append(inst)
    payload = {
        "format": "musicm8-vocal-score-ai-arrangement-v1",
        "title": lyrics.get("title", "Musicm8 Song"),
        "language": lyrics.get("language", "en"),
        "bpm": bpm,
        "key_root": int(plan.get("key_root", 0)),
        "mode": str(plan.get("mode", "minor")),
        "sections": scored_sections,
        "harmony_source": "actual learned AI chord MIDI" if harmony_notes else "song key fallback",
        "melodic_contour_source": "actual learned AI lead MIDI" if melody_notes else "generated singable contour",
        "note": "Words are timed inside the planned song sections, while pitches are anchored to the harmony and lead actually composed by MIDI-LLM. Strong words use active chord pitch classes; intervals are leap-limited for singability.",
    }
    return pm, payload


def main() -> None:
    p = argparse.ArgumentParser(description="Build a word-by-word lead vocal score that follows Musicm8's actual learned AI arrangement.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--lyrics", type=Path, required=True)
    p.add_argument("--harmony-midi", type=Path, default=None)
    p.add_argument("--melody-midi", type=Path, default=None)
    p.add_argument("--midi-out", type=Path, required=True)
    p.add_argument("--score-out", type=Path, required=True)
    args = p.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    lyrics = json.loads(args.lyrics.read_text(encoding="utf-8"))
    harmony = pretty_midi.PrettyMIDI(str(args.harmony_midi)) if args.harmony_midi and args.harmony_midi.exists() else None
    melody = pretty_midi.PrettyMIDI(str(args.melody_midi)) if args.melody_midi and args.melody_midi.exists() else None
    pm, score = build(plan, lyrics, harmony, melody)
    args.midi_out.parent.mkdir(parents=True, exist_ok=True)
    args.score_out.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(args.midi_out))
    score["melody_midi"] = str(args.midi_out)
    args.score_out.write_text(json.dumps(score, indent=2, ensure_ascii=False), encoding="utf-8")
    count = sum(len(s["notes"]) for s in score["sections"])
    print("✅ AI-ARRANGEMENT VOCAL SCORE:", args.score_out)
    print("Word-note events:", count)
    print("Harmony source:", score["harmony_source"])
    print("Melodic contour source:", score["melodic_contour_source"])


if __name__ == "__main__":
    main()
