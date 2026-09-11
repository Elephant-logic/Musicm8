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
    return 1.0 + 0.34 * max(0, count - 1)


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


def scale_pcs(plan: dict[str, Any]) -> set[int]:
    root = int(plan.get("key_root", 0)) % 12
    mode = str(plan.get("mode", "minor"))
    return {(root + x) % 12 for x in SCALES.get(mode, SCALES["minor"])}


def pitched_notes(pm: pretty_midi.PrettyMIDI | None) -> list[pretty_midi.Note]:
    if pm is None:
        return []
    return sorted(
        [n for inst in pm.instruments if not inst.is_drum for n in inst.notes],
        key=lambda n: (n.start, n.pitch, n.end),
    )


def harmony_pcs_at(notes: list[pretty_midi.Note], t: float, beat: float) -> set[int]:
    if not notes:
        return set()
    active = [n for n in notes if n.start - 0.06 <= t <= n.end + 0.06]
    if len({n.pitch % 12 for n in active}) < 2:
        local = [n for n in notes if abs(n.start - t) <= max(0.25, 0.85 * beat)]
        active.extend(local)
    if not active:
        near = sorted(notes, key=lambda n: min(abs(t - n.start), abs(t - n.end)))[:8]
        active = [n for n in near if min(abs(t - n.start), abs(t - n.end)) <= 1.5 * beat]
    return {int(n.pitch) % 12 for n in active}


def collapse_melody_events(notes: list[pretty_midi.Note], a: float, b: float) -> list[pretty_midi.Note]:
    local = [n for n in notes if a - 0.04 <= n.start < b + 0.04]
    if not local:
        return []
    groups: list[list[pretty_midi.Note]] = []
    for n in local:
        if not groups or abs(n.start - groups[-1][0].start) > 0.035:
            groups.append([n])
        else:
            groups[-1].append(n)
    out: list[pretty_midi.Note] = []
    for g in groups:
        # Prefer a clear top-line note: higher register, then velocity and duration.
        out.append(max(g, key=lambda n: (n.pitch, n.velocity, n.end - n.start)))
    return out


def nearest_allowed_pitch(target: int, pcs: set[int], previous: int | None, lo: int = 59, hi: int = 79) -> int:
    pool = [p for p in range(lo, hi + 1) if p % 12 in pcs]
    if not pool:
        return int(np.clip(target, lo, hi))

    def cost(p: int) -> float:
        c = abs(p - target)
        if previous is not None:
            leap = abs(p - previous)
            c += 0.32 * leap
            if leap > 7:
                c += 3.0 + 0.8 * (leap - 7)
        return c

    return int(min(pool, key=cost))


def source_pitch_at(events: list[pretty_midi.Note], t: float, previous: int | None) -> int:
    if events:
        near = min(events, key=lambda n: abs(n.start - t))
        if abs(near.start - t) <= 1.0:
            return int(near.pitch)
    return int(previous if previous is not None else 67)


def lyrics_for_section(lyrics: dict[str, Any], sec: dict[str, Any], index: int) -> list[str]:
    lyric_sections = lyrics.get("sections", []) if isinstance(lyrics.get("sections", []), list) else []
    if index < len(lyric_sections) and isinstance(lyric_sections[index], dict):
        lines = [str(x).strip() for x in lyric_sections[index].get("lines", []) if str(x).strip()]
        if lines:
            return lines
    wanted = re.sub(r"[^a-z0-9]", "", str(sec.get("name", "")).lower())
    for item in lyric_sections:
        if not isinstance(item, dict):
            continue
        got = re.sub(r"[^a-z0-9]", "", str(item.get("name", item.get("section", ""))).lower())
        if got and got == wanted:
            return [str(x).strip() for x in item.get("lines", []) if str(x).strip()]
    return []


def line_windows(sec: dict[str, Any], lines: list[str], beat: float) -> list[tuple[float, float]]:
    if not lines:
        return []
    margin = max(0.18, 0.50 * beat)
    gap = max(0.14, 0.42 * beat)
    a = sec["start"] + margin
    b = sec["end"] - margin
    available = max(0.4, b - a - gap * max(0, len(lines) - 1))
    desired = []
    for line in lines:
        ws = words(line)
        syll = sum(syllable_weight(w) for w in ws)
        # Aim for recognisable consonants/vowels rather than cramming words into tiny notes.
        wanted = max(2.2 * beat, 0.30 * len(ws) + 0.08 * syll + 0.15)
        desired.append(wanted)
    total = sum(desired) or available
    if total > available:
        scale = available / total
        desired = [max(1.55 * beat, x * scale) for x in desired]
        if sum(desired) > available:
            scale = available / max(sum(desired), 1e-6)
            desired = [x * scale for x in desired]

    eighth = beat * 0.5
    cursor = a
    out: list[tuple[float, float]] = []
    for dur in desired:
        start = round(cursor / eighth) * eighth
        start = max(a, start)
        end = min(b, start + dur)
        if end - start < max(0.38, 1.15 * beat):
            end = min(b, start + max(0.38, 1.15 * beat))
        out.append((start, end))
        cursor = end + gap
    return out


def choose_word_starts(
    events: list[pretty_midi.Note],
    line_start: float,
    line_end: float,
    count: int,
    beat: float,
) -> list[float]:
    if count <= 0:
        return []
    local = collapse_melody_events(events, line_start, line_end)
    if len(local) >= count:
        idxs = np.linspace(0, len(local) - 1, count)
        starts = [float(local[int(round(i))].start) for i in idxs]
    else:
        span = max(0.2, line_end - line_start)
        starts = [line_start + span * i / count for i in range(count)]
        eighth = beat * 0.5
        starts = [round(x / eighth) * eighth for x in starts]
        # Pull planned starts toward nearby AI-composed lead onsets when possible.
        used: set[int] = set()
        for i, t in enumerate(starts):
            if local:
                candidates = [(j, n) for j, n in enumerate(local) if j not in used]
                if candidates:
                    j, n = min(candidates, key=lambda item: abs(item[1].start - t))
                    if abs(n.start - t) <= 0.45 * beat:
                        starts[i] = float(n.start)
                        used.add(j)
    starts = sorted(max(line_start, min(line_end - 0.12, float(x))) for x in starts)
    # Guarantee forward motion after quantisation/snapping.
    min_gap = max(0.10, 0.28 * beat)
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i - 1] + min_gap)
    if starts and starts[-1] > line_end - 0.12:
        shift = starts[-1] - (line_end - 0.12)
        starts = [max(line_start, x - shift) for x in starts]
    return starts


def build(
    plan: dict[str, Any],
    lyrics: dict[str, Any],
    harmony_pm: pretty_midi.PrettyMIDI | None,
    melody_pm: pretty_midi.PrettyMIDI | None,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any]]:
    bpm = float(plan.get("bpm", 120.0))
    beat = 60.0 / max(1.0, bpm)
    scale = scale_pcs(plan)
    harmony_notes = pitched_notes(harmony_pm)
    melody_notes = pitched_notes(melody_pm)

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Voice Oohs"), name="vocal_melody")
    previous: int | None = None
    scored_sections: list[dict[str, Any]] = []
    harmonic_checked = 0
    harmonic_good = 0

    for si, sec in enumerate(section_windows(plan)):
        lines = lyrics_for_section(lyrics, sec, si)
        notes_out: list[dict[str, Any]] = []
        windows = line_windows(sec, lines, beat)

        for li, (line, (line_start, line_end)) in enumerate(zip(lines, windows)):
            ws = words(line)
            if not ws:
                continue
            starts = choose_word_starts(melody_notes, line_start, line_end, len(ws), beat)
            local_melody = collapse_melody_events(melody_notes, line_start, line_end)

            for wi, (word, start) in enumerate(zip(ws, starts)):
                next_start = starts[wi + 1] if wi + 1 < len(starts) else line_end
                gap = min(0.06, 0.10 * beat)
                word_end = max(start + 0.18, next_start - gap)
                word_end = min(line_end, word_end)
                if word_end <= start + 0.11:
                    word_end = min(line_end, start + 0.16)

                target = source_pitch_at(local_melody, start, previous)
                pcs = harmony_pcs_at(harmony_notes, start, beat) if harmony_notes else scale
                allowed = pcs or scale
                pitch = nearest_allowed_pitch(target, allowed, previous)

                # Long/multi-syllable phrase-ending words may naturally span a second
                # note when the AI lead itself moves inside that word window.
                split_event: pretty_midi.Note | None = None
                if (syllable_weight(word) > 1.12 or wi == len(ws) - 1) and word_end - start >= 0.48:
                    later = [n for n in local_melody if start + 0.16 <= n.start <= word_end - 0.14]
                    if later:
                        split_event = later[0]

                segments: list[tuple[float, float, int, int]] = []
                if split_event is not None:
                    split = float(split_event.start)
                    first_end = max(start + 0.14, split - 0.02)
                    if first_end < word_end - 0.12:
                        segments.append((start, first_end, pitch, 2))
                        pcs2 = harmony_pcs_at(harmony_notes, split, beat) if harmony_notes else scale
                        pitch2 = nearest_allowed_pitch(int(split_event.pitch), pcs2 or scale, pitch)
                        segments.append((split, word_end, pitch2, 3))
                    else:
                        segments.append((start, word_end, pitch, 2))
                else:
                    segments.append((start, word_end, pitch, 2))

                for part_index, (a, b, p, note_type) in enumerate(segments):
                    context = harmony_pcs_at(harmony_notes, a, beat) if harmony_notes else set()
                    if context:
                        harmonic_checked += 1
                        harmonic_good += int(p % 12 in context)
                    row = {
                        "start": round(a, 5),
                        "end": round(b, 5),
                        "duration": round(b - a, 5),
                        "pitch": int(p),
                        "velocity": 84 if wi in {0, len(ws) - 1} else 77,
                        "word": word,
                        "syllable": word,
                        "line_index": li,
                        "word_index": wi,
                        "note_type": int(note_type),
                        "word_part": part_index,
                        "harmony_pitch_classes": sorted(context),
                    }
                    notes_out.append(row)
                    inst.notes.append(pretty_midi.Note(row["velocity"], row["pitch"], row["start"], row["end"]))
                    previous = int(p)

        scored_sections.append({**sec, "lines": lines, "notes": notes_out})

    if not inst.notes:
        raise RuntimeError("No lyric note events were created for the vocal score")
    pm.instruments.append(inst)
    harmony_fit = harmonic_good / harmonic_checked if harmonic_checked else None
    if harmony_notes and harmonic_checked and harmony_fit is not None and harmony_fit < 0.96:
        raise RuntimeError(f"Vocal harmony guard failed before synthesis: fit={harmony_fit:.3f}")

    payload = {
        "format": "musicm8-vocal-score-ai-arrangement-v2",
        "title": lyrics.get("title", "Musicm8 Song"),
        "language": lyrics.get("language", "en"),
        "bpm": bpm,
        "key_root": int(plan.get("key_root", 0)),
        "mode": str(plan.get("mode", "minor")),
        "sections": scored_sections,
        "harmony_source": "actual tightened AI chord MIDI" if harmony_notes else "song-key fallback",
        "rhythm_source": "actual tightened AI lead onsets with lyric-friendly phrase spacing" if melody_notes else "shared beat grid",
        "harmony_fit": None if harmony_fit is None else round(float(harmony_fit), 4),
        "harmonic_events_checked": harmonic_checked,
        "note": (
            "Vocal timing is no longer free-running across a section. Each lyric line gets breathing space, word onsets follow the actual AI lead rhythm when available, "
            "and every scored pitch is projected into the active harmony before SoulX sees it. Long words only use melisma when the AI lead itself moves inside that word."
        ),
    }
    return pm, payload


def main() -> None:
    p = argparse.ArgumentParser(description="Build a natural, harmony-locked lead vocal score from Musicm8's actual AI arrangement.")
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
    print("✅ NATURAL HARMONY-LOCKED VOCAL SCORE:", args.score_out)
    print("Word/note events:", count)
    print("Harmony source:", score["harmony_source"])
    print("Rhythm source:", score["rhythm_source"])
    print("Harmony fit:", score["harmony_fit"])


if __name__ == "__main__":
    main()
