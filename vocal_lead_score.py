from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pretty_midi

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", str(text))


def syllable_weight(word: str) -> float:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1.0
    groups = re.findall(r"[aeiouy]+", w)
    count = max(1, len(groups))
    if w.endswith("e") and count > 1 and not w.endswith(("le", "ye")):
        count -= 1
    return 1.0 + 0.28 * max(0, count - 1)


def scale_pitch(root: int, degree0: int, octave: int, mode: str) -> int:
    scale = SCALES.get(mode, SCALES["minor"])
    oct_add, idx = divmod(int(degree0), 7)
    return 12 * (octave + 1 + oct_add) + int(root) + scale[idx]


def chord_tones(root: int, degree: int, mode: str, octave: int = 4) -> list[int]:
    d = int(degree) - 1
    return [scale_pitch(root, d + x, octave, mode) for x in (0, 2, 4)]


def nearest(candidates: list[int], target: int, lo: int = 57, hi: int = 81) -> int:
    pool: list[int] = []
    for p in candidates:
        for shift in (-24, -12, 0, 12, 24):
            q = p + shift
            if lo <= q <= hi:
                pool.append(q)
    if not pool:
        return max(lo, min(hi, target))
    return min(pool, key=lambda q: (abs(q - target), q))


def section_windows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    bpm = float(plan.get("bpm", 120.0))
    bar_s = 4.0 * 60.0 / max(1.0, bpm)
    cursor = 0.0
    out: list[dict[str, Any]] = []
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


def current_degree(plan: dict[str, Any], t: float) -> int:
    bpm = float(plan.get("bpm", 120.0))
    bar_len = 4.0 * 60.0 / max(1.0, bpm)
    bar = max(0, int(math.floor(t / max(1e-6, bar_len))))
    prog = [int(x) for x in plan.get("progression_degrees", [1, 6, 3, 7]) if 1 <= int(x) <= 7] or [1, 6, 3, 7]
    return prog[bar % len(prog)]


def choose_pitch(plan: dict[str, Any], t: float, contour: int, previous: int | None, strong: bool) -> int:
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    degree = current_degree(plan, t)
    chord = chord_tones(root, degree, mode, octave=4)
    scale = [scale_pitch(root, i, 4, mode) for i in range(7)]
    target = 66 + contour
    pitch = nearest(chord if strong else scale, target)
    if previous is not None:
        while pitch - previous > 6:
            pitch -= 12
        while previous - pitch > 6:
            pitch += 12
        pitch = max(57, min(81, pitch))
    return int(pitch)


def line_note_specs(plan: dict[str, Any], line: str, start: float, end: float, line_index: int, previous: int | None) -> tuple[list[dict[str, Any]], int | None]:
    ws = words(line)
    if not ws or end <= start + 0.12:
        return [], previous

    # The lead singer gets one explicit word event at a time. Longer words get more
    # duration, but no word is crammed into an arbitrarily tiny slot.
    weights = [syllable_weight(w) for w in ws]
    gap_weight = 0.22
    total_weight = sum(weights) + gap_weight * max(0, len(ws) - 1)
    usable = max(0.12, end - start)
    unit = usable / max(total_weight, 1e-6)

    contour_bank = [0, 2, 1, 3, 2, 0, -1, 1, 0, -2, 0, 2, 1, -1, 0, 1]
    cursor = start
    out: list[dict[str, Any]] = []
    for i, (word, weight) in enumerate(zip(ws, weights)):
        dur = max(0.13, unit * weight)
        remaining = len(ws) - i - 1
        latest_end = end - remaining * 0.13
        note_end = min(latest_end, cursor + dur)
        note_end = max(cursor + 0.10, note_end)
        strong = (i == 0 or i == len(ws) - 1 or i % 4 == 0)
        contour = contour_bank[(i + line_index * 3) % len(contour_bank)]
        pitch = choose_pitch(plan, cursor, contour, previous, strong)
        out.append({
            "start": round(cursor, 5),
            "end": round(note_end, 5),
            "duration": round(note_end - cursor, 5),
            "pitch": pitch,
            "velocity": 82 if strong else 74,
            "word": word,
            "syllable": word,
            "line_index": line_index,
            "word_index": i,
            "note_type": 2,
        })
        previous = pitch
        cursor = note_end
        if i < len(ws) - 1:
            cursor = min(end - remaining * 0.10, cursor + max(0.035, unit * gap_weight))

    return out, previous


def build(plan: dict[str, Any], lyrics: dict[str, Any]) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any]]:
    bpm = float(plan.get("bpm", 120.0))
    beat = 60.0 / max(1.0, bpm)
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Voice Oohs"), name="vocal_melody")

    windows = section_windows(plan)
    lyric_sections = lyrics.get("sections", []) if isinstance(lyrics.get("sections", []), list) else []
    scored_sections: list[dict[str, Any]] = []
    previous_pitch: int | None = None

    for i, sec in enumerate(windows):
        lines = []
        if i < len(lyric_sections) and isinstance(lyric_sections[i], dict):
            lines = [str(x).strip() for x in lyric_sections[i].get("lines", []) if str(x).strip()]

        notes: list[dict[str, Any]] = []
        if lines:
            margin = min(0.35 * beat, max(0.04, (sec["end"] - sec["start"]) * 0.03))
            a = sec["start"] + margin
            b = sec["end"] - margin
            line_weights = [max(2.0, sum(syllable_weight(w) for w in words(line))) for line in lines]
            total = sum(line_weights) or float(len(lines))
            cursor = a
            for line_i, (line, lw) in enumerate(zip(lines, line_weights)):
                share = (b - a) * lw / total
                line_end = b if line_i == len(lines) - 1 else min(b, cursor + share)
                gap = min(0.32 * beat, max(0.06, share * 0.07))
                phrase_end = max(cursor + 0.12, line_end - (gap if line_i < len(lines) - 1 else 0.0))
                phrase_notes, previous_pitch = line_note_specs(plan, line, cursor, phrase_end, line_i, previous_pitch)
                notes.extend(phrase_notes)
                cursor = line_end

        for n in notes:
            inst.notes.append(pretty_midi.Note(
                velocity=int(n["velocity"]),
                pitch=int(n["pitch"]),
                start=float(n["start"]),
                end=float(n["end"]),
            ))

        scored_sections.append({**sec, "lines": lines, "notes": notes})

    pm.instruments.append(inst)
    payload = {
        "format": "musicm8-vocal-score-v3",
        "title": lyrics.get("title", "Musicm8 Song"),
        "language": lyrics.get("language", "en"),
        "bpm": bpm,
        "key_root": int(plan.get("key_root", 0)),
        "mode": str(plan.get("mode", "minor")),
        "sections": scored_sections,
        "note": "Diction-first lead score: one explicit word event per note, exact absolute timing, diatonic pitch, chord-tone anchors on strong words. This score is generated for a score-conditioned neural singer; no post-hoc time stretching is required.",
    }
    return pm, payload


def main() -> None:
    p = argparse.ArgumentParser(description="Build a diction-first, score-conditioned lead vocal melody from Musicm8 lyrics and harmony.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--lyrics", type=Path, required=True)
    p.add_argument("--midi-out", type=Path, required=True)
    p.add_argument("--score-out", type=Path, required=True)
    args = p.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    lyrics = json.loads(args.lyrics.read_text(encoding="utf-8"))
    pm, score = build(plan, lyrics)
    args.midi_out.parent.mkdir(parents=True, exist_ok=True)
    args.score_out.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(args.midi_out))
    score["melody_midi"] = str(args.midi_out)
    args.score_out.write_text(json.dumps(score, indent=2, ensure_ascii=False), encoding="utf-8")
    count = sum(len(s["notes"]) for s in score["sections"])
    print("✅ DICTION-FIRST VOCAL SCORE:", args.score_out)
    print("✅ Vocal melody MIDI:", args.midi_out)
    print("Word-note events:", count)


if __name__ == "__main__":
    main()
