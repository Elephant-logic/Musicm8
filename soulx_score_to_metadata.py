from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from g2p_en import G2p

_G2P = G2p()
_PHONE_RE = re.compile(r"^[A-Z]+[0-2]?$")


def phoneme_token(word: str) -> str:
    clean = re.sub(r"[^A-Za-z']", "", str(word)).strip().lower()
    if not clean:
        return "<SP>"
    raw = _G2P(clean)
    phones = [str(x) for x in raw if _PHONE_RE.match(str(x))]
    if not phones:
        return "<SP>"
    return "en_" + "-".join(phones)


def add_event(events: list[dict[str, Any]], text: str, phone: str, pitch: int, typ: int, duration: float) -> None:
    duration = float(duration)
    if duration < 0.025:
        return
    events.append({"text": text, "phoneme": phone, "pitch": int(pitch), "type": int(typ), "duration": duration})


def line_metadata(section: dict[str, Any], line_index: int, notes: list[dict[str, Any]]) -> dict[str, Any] | None:
    notes = sorted(notes, key=lambda n: (float(n.get("start", 0.0)), int(n.get("word_index", 0))))
    if not notes:
        return None
    sec_start = float(section.get("start", 0.0))
    sec_end = float(section.get("end", max(float(n["end"]) for n in notes)))
    phrase_start = max(sec_start, min(float(n["start"]) for n in notes) - 0.06)
    phrase_end = min(sec_end, max(float(n["end"]) for n in notes) + 0.10)
    phrase_end = max(phrase_start + 0.10, phrase_end)

    events: list[dict[str, Any]] = []
    cursor = phrase_start
    previous_word: str | None = None
    for note in notes:
        a = max(phrase_start, float(note["start"]))
        b = min(phrase_end, float(note["end"]))
        if a - cursor >= 0.025:
            add_event(events, "<SP>", "<SP>", 0, 1, a - cursor)
        if b <= a + 0.025:
            continue
        word = str(note.get("word") or note.get("syllable") or "").strip()
        phone = phoneme_token(word)
        if phone == "<SP>":
            add_event(events, "<SP>", "<SP>", 0, 1, b - a)
        else:
            typ = 3 if previous_word and word.lower() == previous_word.lower() else 2
            add_event(events, word, phone, int(note.get("pitch", 60)), typ, b - a)
            previous_word = word
        cursor = max(cursor, b)
    if phrase_end - cursor >= 0.025:
        add_event(events, "<SP>", "<SP>", 0, 1, phrase_end - cursor)

    if not events:
        return None
    total = sum(e["duration"] for e in events)
    target = phrase_end - phrase_start
    events[-1]["duration"] = max(0.025, events[-1]["duration"] + (target - total))

    return {
        "index": f"musicm8_{int(round(phrase_start * 1000))}_{int(round(phrase_end * 1000))}",
        "language": "English",
        "time": [int(round(phrase_start * 1000)), int(round(phrase_end * 1000))],
        "duration": " ".join(f"{e['duration']:.5f}" for e in events),
        "text": " ".join(e["text"] for e in events),
        "phoneme": " ".join(e["phoneme"] for e in events),
        "note_pitch": " ".join(str(e["pitch"]) for e in events),
        "note_type": " ".join(str(e["type"]) for e in events),
    }


def build(score: dict[str, Any]) -> list[dict[str, Any]]:
    metas: list[dict[str, Any]] = []
    for section in score.get("sections", []):
        grouped: dict[int, list[dict[str, Any]]] = {}
        for n in section.get("notes", []):
            li = n.get("line_index")
            if li is None:
                continue
            grouped.setdefault(int(li), []).append(n)
        for li in sorted(grouped):
            row = line_metadata(section, li, grouped[li])
            if row:
                metas.append(row)
    return sorted(metas, key=lambda m: m["time"][0])


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Musicm8's exact vocal score into SoulX-Singer score metadata.")
    p.add_argument("--score", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    score = json.loads(args.score.read_text(encoding="utf-8"))
    metas = build(score)
    if not metas:
        raise RuntimeError("No lyric/word note events were available for SoulX-Singer")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metas, indent=2, ensure_ascii=False), encoding="utf-8")
    word_count = sum(sum(1 for x in m["phoneme"].split() if x.startswith("en_")) for m in metas)
    print("✅ SoulX score metadata:", args.out)
    print("Phrase segments:", len(metas), "word events:", word_count)
    for m in metas[:5]:
        print(f"  {m['time'][0]/1000:.2f}-{m['time'][1]/1000:.2f}s | {m['text']}")


if __name__ == "__main__":
    main()
