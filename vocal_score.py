from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pretty_midi

VOWELS = "aeiouy"


def simple_syllables(word: str) -> list[str]:
    clean = re.sub(r"[^A-Za-z']", "", word).lower()
    if not clean:
        return []
    if len(clean) <= 3:
        return [clean]
    chunks: list[str] = []
    start = 0
    in_vowel = clean[0] in VOWELS
    for i in range(1, len(clean)):
        now = clean[i] in VOWELS
        if in_vowel and not now and i + 1 < len(clean) and clean[i + 1] in VOWELS:
            chunks.append(clean[start:i + 1])
            start = i + 1
        in_vowel = now
    chunks.append(clean[start:])
    chunks = [x for x in chunks if x]
    if len(chunks) > 1 and chunks[-1] == "e":
        chunks[-2] += chunks[-1]
        chunks.pop()
    return chunks or [clean]


def phonemes_for_text(text: str) -> str | None:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        return None
    try:
        result = subprocess.run([exe, "-q", "--ipa=3", text], check=True, text=True, capture_output=True)
        value = result.stdout.strip()
        return value or None
    except Exception:
        return None


def section_windows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    bpm = float(plan.get("bpm", 120.0))
    bar_s = 4.0 * 60.0 / max(1.0, bpm)
    cursor = 0.0
    out = []
    for i, sec in enumerate(plan.get("sections", [])):
        bars = max(1, int(sec.get("bars", 1)))
        end = cursor + bars * bar_s
        out.append({"index": i, "name": str(sec.get("name", "section")), "start": cursor, "end": end, "bars": bars})
        cursor = end
    return out


def lyric_units(lines: list[str]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for line_i, line in enumerate(lines):
        words = re.findall(r"[A-Za-z0-9']+", str(line))
        for word_i, word in enumerate(words):
            syllables = simple_syllables(word)
            for syl_i, syl in enumerate(syllables):
                units.append({
                    "text": syl,
                    "word": word,
                    "line_index": line_i,
                    "word_index": word_i,
                    "syllable_index": syl_i,
                    "syllables_in_word": len(syllables),
                })
    return units


def align_section(notes: list[pretty_midi.Note], lines: list[str]) -> list[dict[str, Any]]:
    if not notes:
        return []
    units = lyric_units(lines)
    if not units:
        return [{
            "start": round(n.start, 5),
            "end": round(n.end, 5),
            "duration": round(n.end - n.start, 5),
            "pitch": int(n.pitch),
            "velocity": int(n.velocity),
            "syllable": "_",
            "word": None,
            "line_index": None,
            "word_index": None,
            "phonemes": None,
        } for n in notes]

    aligned: list[dict[str, Any]] = []
    n_notes = len(notes)
    n_units = len(units)
    assignments: list[list[dict[str, Any]]] = [[] for _ in notes]
    for ui, unit in enumerate(units):
        ni = min(n_notes - 1, int(ui * n_notes / max(1, n_units)))
        assignments[ni].append(unit)

    last_word: str | None = None
    last_line: int | None = None
    last_word_index: int | None = None
    for i, note in enumerate(notes):
        here = assignments[i]
        if here:
            syllable = "-".join(x["text"] for x in here)
            word = here[-1]["word"]
            line_index = int(here[-1]["line_index"])
            word_index = int(here[-1]["word_index"])
            last_word = word
            last_line = line_index
            last_word_index = word_index
            phonemes = phonemes_for_text(" ".join(dict.fromkeys(x["word"] for x in here)))
        else:
            syllable = "_"
            word = last_word
            line_index = last_line
            word_index = last_word_index
            phonemes = None
        aligned.append({
            "start": round(note.start, 5),
            "end": round(note.end, 5),
            "duration": round(note.end - note.start, 5),
            "pitch": int(note.pitch),
            "velocity": int(note.velocity),
            "syllable": syllable,
            "word": word,
            "line_index": line_index,
            "word_index": word_index,
            "phonemes": phonemes,
        })
    return aligned


def main() -> None:
    p = argparse.ArgumentParser(description="Align Musicm8 lyrics to its melody MIDI and save an editable vocal score.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--lyrics", type=Path, required=True)
    p.add_argument("--melody-midi", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="vocal_score.json")
    args = p.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    lyrics = json.loads(args.lyrics.read_text(encoding="utf-8"))
    pm = pretty_midi.PrettyMIDI(str(args.melody_midi))
    all_notes = sorted([n for inst in pm.instruments for n in inst.notes], key=lambda n: (n.start, n.pitch))
    windows = section_windows(plan)
    lyric_sections = lyrics.get("sections", [])
    scored_sections = []

    for i, window in enumerate(windows):
        lines = lyric_sections[i].get("lines", []) if i < len(lyric_sections) else []
        notes = [n for n in all_notes if window["start"] <= n.start < window["end"]]
        scored_sections.append({**window, "lines": lines, "notes": align_section(notes, lines)})

    payload = {
        "format": "musicm8-vocal-score-v2",
        "title": lyrics.get("title", "Musicm8 Song"),
        "language": lyrics.get("language", "en"),
        "bpm": float(plan.get("bpm", 120.0)),
        "key_root": int(plan.get("key_root", 0)),
        "mode": plan.get("mode", "minor"),
        "melody_midi": str(args.melody_midi),
        "sections": scored_sections,
        "note": "Each melody note is aligned to a lyric syllable, lyric line and word index. These timings are the authority for vocal synchronization.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    copy_midi = args.out.parent / "vocal_melody.mid"
    shutil.copy2(args.melody_midi, copy_midi)
    print("✅ Vocal score:", args.out)
    print("✅ Vocal melody MIDI:", copy_midi)
    print("Aligned notes:", sum(len(s["notes"]) for s in scored_sections))


if __name__ == "__main__":
    main()
