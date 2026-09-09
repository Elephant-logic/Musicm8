from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn

SECTION_NAMES = [
    "unknown", "intro", "verse", "prechorus", "chorus", "postchorus",
    "bridge", "breakdown", "drop", "solo", "interlude", "outro",
]
SECTION_TO_ID = {name: i for i, name in enumerate(SECTION_NAMES)}
STEM_NAMES = ["mix", "vocals", "drums", "bass", "guitar", "keys", "other"]
STEM_TO_ID = {name: i for i, name in enumerate(STEM_NAMES)}

ROOTS = {name: i for i, name in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
ROOT_ALIASES = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
CHORD_QUALITIES = ["maj", "min", "dim", "aug", "7", "maj7", "min7", "sus"]
QUALITY_TO_ID = {q: i for i, q in enumerate(CHORD_QUALITIES)}

KEY_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)(?:\s+|:)?(major|minor|maj|min|m)?\s*$")
CHORD_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)(.*)$")


def stable_bucket(text: str, buckets: int, offset: int = 1) -> int:
    if not text:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return offset + (int.from_bytes(digest, "little") % max(1, buckets - offset))


def key_to_id(value: str | None) -> int:
    """0=unknown, 1..12 major, 13..24 minor."""
    if not value:
        return 0
    m = KEY_RE.match(value)
    if not m:
        return 0
    root = m.group(1).upper() + m.group(2)
    root = ROOT_ALIASES.get(root, root)
    if root not in ROOTS:
        return 0
    mode = (m.group(3) or "major").lower()
    is_minor = mode in {"minor", "min", "m"}
    return 1 + ROOTS[root] + (12 if is_minor else 0)


def chord_to_id(value: str | None) -> int:
    """0=unknown/N, then 12 roots x 8 common qualities."""
    if not value or value.strip().upper() in {"N", "NC", "N.C."}:
        return 0
    m = CHORD_RE.match(value)
    if not m:
        return 0
    root = m.group(1).upper() + m.group(2)
    root = ROOT_ALIASES.get(root, root)
    if root not in ROOTS:
        return 0
    tail = m.group(3).lower().replace("-", "min").replace("+", "aug")
    if "maj7" in tail:
        quality = "maj7"
    elif "m7" in tail or "min7" in tail:
        quality = "min7"
    elif "dim" in tail or "°" in tail:
        quality = "dim"
    elif "aug" in tail:
        quality = "aug"
    elif "sus" in tail:
        quality = "sus"
    elif "7" in tail:
        quality = "7"
    elif tail.startswith("m") or "min" in tail:
        quality = "min"
    else:
        quality = "maj"
    return 1 + ROOTS[root] * len(CHORD_QUALITIES) + QUALITY_TO_ID[quality]


def section_to_id(value: str | None) -> int:
    if not value:
        return 0
    normalized = value.lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {"prechorus": "prechorus", "postchorus": "postchorus"}
    normalized = aliases.get(normalized, normalized)
    for name, idx in SECTION_TO_ID.items():
        if name.replace("_", "") == normalized:
            return idx
    return stable_bucket(value.lower(), 32)


def stem_to_id(value: str | None) -> int:
    if not value:
        return 0
    value = value.lower()
    return STEM_TO_ID.get(value, stable_bucket(value, 16))


def serialize_caption(row: dict[str, Any]) -> str:
    """Turn structured metadata into a dense, consistent text condition."""
    parts: list[str] = []
    caption = str(row.get("caption", "")).strip()
    if caption:
        parts.append(caption)
    if row.get("bpm") is not None:
        parts.append(f"tempo {round(float(row['bpm']))} bpm")
    if row.get("key"):
        parts.append(f"key {row['key']}")
    if row.get("meter"):
        parts.append(f"meter {row['meter']}")
    if row.get("stem") and row.get("stem") != "mix":
        parts.append(f"render {row['stem']} stem")
    if row.get("lyrics"):
        parts.append("lyrics: " + str(row["lyrics"]).strip())
    return ", ".join(p for p in parts if p)


def make_frame_array(frames: int, fill: int | float, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((frames,), fill, dtype=dtype)


def apply_events(
    target: torch.Tensor,
    events: Iterable[dict[str, Any]],
    clip_start: float,
    frame_rate: float,
    value_fn,
    end_fallback: float | None = None,
) -> None:
    """Apply absolute-time events onto one clip-local frame array."""
    frames = target.numel()
    clip_end = clip_start + frames / frame_rate
    for event in events:
        start = float(event.get("start", clip_start))
        end = event.get("end", end_fallback)
        if end is None:
            end = start + 1.0 / frame_rate
        end = float(end)
        if end <= clip_start or start >= clip_end:
            continue
        local_a = max(0, int(math.floor((start - clip_start) * frame_rate)))
        local_b = min(frames, int(math.ceil((end - clip_start) * frame_rate)))
        if local_b <= local_a:
            local_b = min(frames, local_a + 1)
        target[local_a:local_b] = value_fn(event)


def build_condition_tensors(
    row: dict[str, Any],
    frames: int,
    frame_rate: float,
    clip_start: float,
    energy: torch.Tensor | None = None,
    phoneme_buckets: int = 512,
    semantic_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    section = make_frame_array(frames, 0, torch.long)
    chord = make_frame_array(frames, 0, torch.long)
    melody = make_frame_array(frames, 0, torch.long)
    phoneme = make_frame_array(frames, 0, torch.long)

    apply_events(section, row.get("sections", []), clip_start, frame_rate,
                 lambda e: section_to_id(str(e.get("label", e.get("section", "unknown")))))
    apply_events(chord, row.get("chords", []), clip_start, frame_rate,
                 lambda e: chord_to_id(str(e.get("chord", e.get("label", "N")))))
    apply_events(melody, row.get("melody", []), clip_start, frame_rate,
                 lambda e: max(0, min(128, int(round(float(e.get("midi", 0)))) + 1)))

    def phoneme_value(e: dict[str, Any]) -> int:
        token = e.get("phoneme") or e.get("text") or e.get("word") or ""
        return stable_bucket(str(token), phoneme_buckets)

    apply_events(phoneme, row.get("phonemes", row.get("lyric_events", [])), clip_start,
                 frame_rate, phoneme_value)

    if energy is None:
        energy = torch.zeros(frames, dtype=torch.float32)
    else:
        energy = energy.float()
        if energy.numel() != frames:
            energy = torch.nn.functional.interpolate(
                energy.view(1, 1, -1), size=frames, mode="linear", align_corners=False
            ).view(-1)
        energy = energy.clamp(0, 1)

    if semantic_ids is None:
        semantic_ids = torch.zeros(frames, dtype=torch.long)
    else:
        semantic_ids = semantic_ids.long()
        if semantic_ids.numel() != frames:
            semantic_ids = torch.nn.functional.interpolate(
                semantic_ids.float().view(1, 1, -1), size=frames, mode="nearest"
            ).long().view(-1)

    return {
        "bpm": torch.tensor(int(round(float(row.get("bpm", 0) or 0))), dtype=torch.long),
        "key_id": torch.tensor(key_to_id(row.get("key")), dtype=torch.long),
        "stem_id": torch.tensor(stem_to_id(row.get("stem", "mix")), dtype=torch.long),
        "section_ids": section,
        "chord_ids": chord,
        "melody_ids": melody,
        "energy": energy,
        "phoneme_ids": phoneme,
        "semantic_ids": semantic_ids,
    }


@dataclass
class ConditionConfig:
    bpm_vocab: int = 256
    key_vocab: int = 25
    stem_vocab: int = 32
    section_vocab: int = 32
    chord_vocab: int = 128
    melody_vocab: int = 129
    phoneme_vocab: int = 512
    semantic_vocab: int = 1024


class FrameConditioner(nn.Module):
    def __init__(self, d_model: int, cfg: ConditionConfig):
        super().__init__()
        self.cfg = cfg
        self.bpm = nn.Embedding(cfg.bpm_vocab, d_model)
        self.key = nn.Embedding(cfg.key_vocab, d_model)
        self.stem = nn.Embedding(cfg.stem_vocab, d_model)
        self.section = nn.Embedding(cfg.section_vocab, d_model)
        self.chord = nn.Embedding(cfg.chord_vocab, d_model)
        self.melody = nn.Embedding(cfg.melody_vocab, d_model)
        self.phoneme = nn.Embedding(cfg.phoneme_vocab, d_model)
        self.semantic = nn.Embedding(cfg.semantic_vocab, d_model)
        self.energy = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)

    def _gather_frames(self, x: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        indices = frame_indices.clamp(0, x.shape[1] - 1)
        return x.index_select(1, indices)

    def forward(self, cond: dict[str, torch.Tensor], frame_indices: torch.Tensor) -> torch.Tensor:
        bpm = cond["bpm"].clamp(0, self.cfg.bpm_vocab - 1)
        key = cond["key_id"].clamp(0, self.cfg.key_vocab - 1)
        stem = cond["stem_id"].clamp(0, self.cfg.stem_vocab - 1)
        section = self._gather_frames(cond["section_ids"], frame_indices).clamp(0, self.cfg.section_vocab - 1)
        chord = self._gather_frames(cond["chord_ids"], frame_indices).clamp(0, self.cfg.chord_vocab - 1)
        melody = self._gather_frames(cond["melody_ids"], frame_indices).clamp(0, self.cfg.melody_vocab - 1)
        phoneme = self._gather_frames(cond["phoneme_ids"], frame_indices).clamp(0, self.cfg.phoneme_vocab - 1)
        semantic = self._gather_frames(cond["semantic_ids"], frame_indices).clamp(0, self.cfg.semantic_vocab - 1)
        energy = self._gather_frames(cond["energy"], frame_indices).unsqueeze(-1)

        global_emb = self.bpm(bpm) + self.key(key) + self.stem(stem)
        x = global_emb[:, None, :]
        x = x + self.section(section) + self.chord(chord) + self.melody(melody)
        x = x + self.phoneme(phoneme) + self.semantic(semantic) + self.energy(energy)
        return self.norm(x)
