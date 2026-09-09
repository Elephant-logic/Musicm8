from __future__ import annotations

from typing import Any


def phonemize_text(text: str, language: str = "en-us") -> list[str]:
    text = text.strip()
    if not text:
        return []
    try:
        from phonemizer import phonemize
        out = phonemize(
            [text], language=language, backend="espeak", strip=True,
            preserve_punctuation=False, with_stress=True, njobs=1,
        )[0]
        return [p for p in out.replace("|", " ").split() if p]
    except Exception:
        return [c.lower() for c in text if c.isalpha() or c == "'"]


def expand_lyric_events_to_phonemes(
    events: list[dict[str, Any]],
    language: str = "en-us",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        start = float(event.get("start", 0.0))
        end = float(event.get("end", start))
        text = str(event.get("text", event.get("word", "")))
        phones = phonemize_text(text, language=language)
        if not phones or end <= start:
            continue
        dt = (end - start) / len(phones)
        for i, phone in enumerate(phones):
            out.append({"start": start + i * dt, "end": start + (i + 1) * dt, "phoneme": phone})
    return out
