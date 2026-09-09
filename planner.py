from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from conditioning import build_condition_tensors
from phonemes import expand_lyric_events_to_phonemes


@dataclass
class SectionPlan:
    label: str
    seconds: float
    energy: float = 0.5
    chords: list[str] = field(default_factory=list)
    lyrics: str = ""


@dataclass
class SongPlan:
    prompt: str
    duration: float
    bpm: float
    key: str
    meter: str = "4/4"
    sections: list[SectionPlan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SongPlan":
        return cls(
            prompt=d.get("prompt", ""), duration=float(d["duration"]),
            bpm=float(d.get("bpm", 100)), key=d.get("key", "C major"),
            meter=d.get("meter", "4/4"),
            sections=[SectionPlan(**s) for s in d.get("sections", [])],
        )


def _contains(prompt: str, *words: str) -> bool:
    p = prompt.lower()
    return any(w in p for w in words)


def heuristic_plan(prompt: str, duration: float = 90.0, bpm: float | None = None,
                   key: str | None = None, lyrics: str = "") -> SongPlan:
    if bpm is None:
        if _contains(prompt, "ballad", "ambient", "slow"):
            bpm = 78
        elif _contains(prompt, "drum and bass", "dnb"):
            bpm = 174
        elif _contains(prompt, "house", "techno", "dance"):
            bpm = 124
        elif _contains(prompt, "trap", "hip hop", "r&b", "rnb"):
            bpm = 92
        else:
            bpm = 108
    if key is None:
        key = "F# minor" if _contains(prompt, "dark", "sad", "melanch", "moody") else "C major"

    if duration < 35:
        blueprint = [("intro", .12, .25), ("verse", .38, .48), ("chorus", .38, .82), ("outro", .12, .3)]
    else:
        blueprint = [
            ("intro", .07, .25), ("verse", .18, .45), ("prechorus", .08, .62),
            ("chorus", .17, .88), ("verse", .17, .5), ("chorus", .16, .9),
            ("bridge", .08, .58), ("chorus", .07, .95), ("outro", .02, .3),
        ]
    if "minor" in key.lower():
        chords = ["i", "VI", "III", "VII"]
        actual = ["F#m", "D", "A", "E"] if key.lower().startswith("f#") else ["Am", "F", "C", "G"]
    else:
        chords = ["I", "V", "vi", "IV"]
        actual = ["C", "G", "Am", "F"]
    _ = chords
    sections = [SectionPlan(label=n, seconds=duration * r, energy=e, chords=actual) for n, r, e in blueprint]
    if lyrics:
        lyric_sections = [s for s in sections if s.label in {"verse", "prechorus", "chorus", "bridge"}]
        chunks = [x.strip() for x in re.split(r"\n\s*\n", lyrics) if x.strip()] or [lyrics.strip()]
        for i, sec in enumerate(lyric_sections):
            sec.lyrics = chunks[min(i, len(chunks) - 1)]
    return SongPlan(prompt=prompt, duration=duration, bpm=float(bpm), key=key, sections=sections)


def save_plan(plan: SongPlan, path: str | Path) -> None:
    Path(path).write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_plan(path: str | Path) -> SongPlan:
    return SongPlan.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def plan_to_row(plan: SongPlan) -> dict[str, Any]:
    row: dict[str, Any] = {
        "caption": plan.prompt, "bpm": plan.bpm, "key": plan.key, "meter": plan.meter,
        "sections": [], "chords": [], "lyric_events": [],
    }
    cursor = 0.0
    beat_seconds = 60.0 / plan.bpm
    bar_seconds = beat_seconds * 4
    for section in plan.sections:
        start, end = cursor, min(plan.duration, cursor + section.seconds)
        row["sections"].append({"start": start, "end": end, "label": section.label})
        if section.chords:
            chord_len = bar_seconds
            i = 0
            t = start
            while t < end:
                row["chords"].append({"start": t, "end": min(end, t + chord_len), "chord": section.chords[i % len(section.chords)]})
                i += 1
                t += chord_len
        if section.lyrics.strip():
            words = section.lyrics.split()
            if words:
                dt = (end - start) / len(words)
                for i, word in enumerate(words):
                    row["lyric_events"].append({"start": start + i * dt, "end": start + (i + 1) * dt, "text": word})
        cursor = end
    row["phonemes"] = expand_lyric_events_to_phonemes(row["lyric_events"])
    return row


def plan_to_conditions(plan: SongPlan, frames: int, frame_rate: float, stem: str = "mix") -> dict[str, torch.Tensor]:
    row = plan_to_row(plan)
    row["stem"] = stem
    cond = build_condition_tensors(row, frames, frame_rate, 0.0)
    energy = torch.zeros(frames)
    cursor = 0.0
    for sec in plan.sections:
        a = max(0, int(math.floor(cursor * frame_rate)))
        b = min(frames, int(math.ceil((cursor + sec.seconds) * frame_rate)))
        energy[a:b] = float(sec.energy)
        cursor += sec.seconds
    cond["energy"] = energy.clamp(0, 1)
    return {k: v.unsqueeze(0) if v.ndim > 0 else v.reshape(1) for k, v in cond.items()}
