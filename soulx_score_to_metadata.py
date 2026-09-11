from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from g2p_en import G2p

_G2P = G2p()
_PHONE_RE = re.compile(r"^[A-Z]+[0-2]?$", re.ASCII)


def phoneme_token(word: str) -> str:
    clean = re.sub(r"[^A-Za-z']", "", str(word)).strip().lower()
    if not clean:
        return "<SP>"
    raw = _G2P(clean)
    phones = [str(x) for x in raw if _PHONE_RE.match(str(x))]
    return "en_" + "-".join(phones) if phones else "<SP>"


def add_event(events: list[dict[str, Any]], text: str, phone: str, pitch: int, typ: int, duration: float) -> None:
    duration = float(duration)
    if duration < 0.025:
        return
    events.append({"text": text, "phoneme": phone, "pitch": int(pitch), "type": int(typ), "duration": duration})


def coalesce_spaces(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent silence tokens so SoulX sees phrases, not stuttering gaps."""
    out: list[dict[str, Any]] = []
    for event in events:
        e = dict(event)
        is_space = e.get("phoneme") == "<SP>" or e.get("text") == "<SP>"
        if is_space and out and (out[-1].get("phoneme") == "<SP>" or out[-1].get("text") == "<SP>"):
            out[-1]["duration"] = float(out[-1]["duration"]) + float(e["duration"])
        else:
            out.append(e)
    return out


def line_metadata(section: dict[str, Any], line_index: int, notes: list[dict[str, Any]]) -> dict[str, Any] | None:
    notes = sorted(notes, key=lambda n: (float(n.get("start", 0.0)), int(n.get("word_index", 0)), int(n.get("word_part", 0))))
    if not notes:
        return None
    sec_start = float(section.get("start", 0.0))
    sec_end = float(section.get("end", max(float(n["end"]) for n in notes)))
    phrase_start = max(sec_start, min(float(n["start"]) for n in notes) - 0.08)
    phrase_end = min(sec_end, max(float(n["end"]) for n in notes) + 0.14)
    phrase_end = max(phrase_start + 0.15, phrase_end)

    events: list[dict[str, Any]] = []
    cursor = phrase_start
    previous_word: str | None = None
    for note in notes:
        a = max(phrase_start, float(note["start"]))
        b = min(phrase_end, float(note["end"]))
        gap = a - cursor
        if gap >= 0.14:
            add_event(events, "<SP>", "<SP>", 0, 1, gap)
        elif gap >= 0.025:
            # Tiny 25-100 ms inter-word gaps were previously emitted as a silence
            # token between almost every word, producing robotic stop/start diction.
            # Fold them into the preceding sung event so the phrase stays legato.
            if events:
                events[-1]["duration"] = float(events[-1]["duration"]) + gap
            else:
                add_event(events, "<SP>", "<SP>", 0, 1, gap)
        if b <= a + 0.025:
            continue
        word = str(note.get("word") or note.get("syllable") or "").strip()
        phone = phoneme_token(word)
        if phone == "<SP>":
            add_event(events, "<SP>", "<SP>", 0, 1, b - a)
        else:
            # SoulX uses type 2 for a word/note onset and type 3 for continuation
            # notes of the same sung word. Respect the score rather than guessing.
            typ = int(note.get("note_type", 3 if previous_word and word.lower() == previous_word.lower() else 2))
            typ = 3 if typ == 3 else 2
            add_event(events, word, phone, int(note.get("pitch", 60)), typ, b - a)
            previous_word = word
        cursor = max(cursor, b)
    if phrase_end - cursor >= 0.025:
        add_event(events, "<SP>", "<SP>", 0, 1, phrase_end - cursor)
    events = coalesce_spaces(events)
    if not events:
        return None

    total = sum(float(e["duration"]) for e in events)
    target = phrase_end - phrase_start
    events[-1]["duration"] = max(0.025, float(events[-1]["duration"]) + (target - total))
    return {
        "index": f"musicm8_{int(round(phrase_start * 1000))}_{int(round(phrase_end * 1000))}",
        "language": "English",
        "time": [int(round(phrase_start * 1000)), int(round(phrase_end * 1000))],
        "duration": " ".join(f"{float(e['duration']):.5f}" for e in events),
        "text": " ".join(str(e["text"]) for e in events),
        "phoneme": " ".join(str(e["phoneme"]) for e in events),
        "note_pitch": " ".join(str(int(e["pitch"])) for e in events),
        "note_type": " ".join(str(int(e["type"])) for e in events),
        "line_count": 1,
    }


def _split(row: dict[str, Any]) -> list[dict[str, Any]]:
    ds = [float(x) for x in row["duration"].split()]
    tx = row["text"].split()
    ph = row["phoneme"].split()
    pi = [int(x) for x in row["note_pitch"].split()]
    ty = [int(x) for x in row["note_type"].split()]
    n = min(len(ds), len(tx), len(ph), len(pi), len(ty))
    return [dict(duration=ds[i], text=tx[i], phoneme=ph[i], pitch=pi[i], type=ty[i]) for i in range(n)]


def merge_rows(rows: list[dict[str, Any]], max_duration_s: float = 9.0, max_gap_s: float = 1.15) -> list[dict[str, Any]]:
    """Join neighbouring lyric lines into short continuous musical phrases."""
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_events: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current, current_events
        if current is None or not current_events:
            current = None
            current_events = []
            return
        current_events = coalesce_spaces(current_events)
        start_ms, end_ms = current["time"]
        total = sum(float(e["duration"]) for e in current_events)
        target = max(0.025, (end_ms - start_ms) / 1000.0)
        current_events[-1]["duration"] = max(0.025, float(current_events[-1]["duration"]) + (target - total))
        out.append({
            "index": f"musicm8_{start_ms}_{end_ms}",
            "language": "English",
            "time": [start_ms, end_ms],
            "duration": " ".join(f"{float(e['duration']):.5f}" for e in current_events),
            "text": " ".join(str(e["text"]) for e in current_events),
            "phoneme": " ".join(str(e["phoneme"]) for e in current_events),
            "note_pitch": " ".join(str(int(e["pitch"])) for e in current_events),
            "note_type": " ".join(str(int(e["type"])) for e in current_events),
            "line_count": int(current.get("line_count", 1)),
        })
        current = None
        current_events = []

    for row in rows:
        if current is None:
            current = dict(row)
            current_events = _split(row)
            continue
        gap_s = (int(row["time"][0]) - int(current["time"][1])) / 1000.0
        merged_duration = (int(row["time"][1]) - int(current["time"][0])) / 1000.0
        if gap_s <= max_gap_s and merged_duration <= max_duration_s and int(current.get("line_count", 1)) < 2:
            if gap_s >= 0.14:
                current_events.append(dict(duration=gap_s, text="<SP>", phoneme="<SP>", pitch=0, type=1))
            elif gap_s >= 0.025 and current_events:
                current_events[-1]["duration"] = float(current_events[-1]["duration"]) + gap_s
            current_events.extend(_split(row))
            current_events = coalesce_spaces(current_events)
            current["time"][1] = int(row["time"][1])
            current["line_count"] = int(current.get("line_count", 1)) + 1
        else:
            flush()
            current = dict(row)
            current_events = _split(row)
    flush()
    return out


def build(score: dict[str, Any]) -> list[dict[str, Any]]:
    line_rows: list[dict[str, Any]] = []
    for section in score.get("sections", []):
        grouped: dict[int, list[dict[str, Any]]] = {}
        for n in section.get("notes", []):
            li = n.get("line_index")
            if li is not None:
                grouped.setdefault(int(li), []).append(n)
        for li in sorted(grouped):
            row = line_metadata(section, li, grouped[li])
            if row:
                line_rows.append(row)
    line_rows.sort(key=lambda m: m["time"][0])
    return merge_rows(line_rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Musicm8's harmony-locked vocal score into natural short SoulX phrase metadata.")
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
    print("✅ SoulX natural phrase metadata:", args.out)
    print("Phrase chunks:", len(metas), "word/note events:", word_count)
    for m in metas[:6]:
        print(f"  {m['time'][0]/1000:.2f}-{m['time'][1]/1000:.2f}s | lines={m.get('line_count',1)} | {m['text']}")


if __name__ == "__main__":
    main()
