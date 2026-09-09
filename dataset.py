from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


FRAME_KEYS = ["section_ids", "chord_ids", "melody_ids", "energy", "phoneme_ids", "semantic_ids"]
GLOBAL_KEYS = ["bpm", "key_id", "stem_id"]


class TokenDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        max_frames: int | None = None,
        min_frames: int = 1,
        random_crop: bool = True,
    ):
        self.index_path = Path(index_path)
        self.root = self.index_path.parent
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.random_crop = random_crop
        self.rows = [json.loads(x) for x in self.index_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not self.rows:
            raise ValueError(f"Dataset index is empty: {self.index_path}")
        meta_path = self.root / "meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        payload = torch.load(self.root / row["tokens"], map_location="cpu", weights_only=True)
        if isinstance(payload, torch.Tensor):
            codes = payload.long()
            t = codes.shape[-1]
            cond = {
                "bpm": torch.tensor(0), "key_id": torch.tensor(0), "stem_id": torch.tensor(0),
                "section_ids": torch.zeros(t, dtype=torch.long),
                "chord_ids": torch.zeros(t, dtype=torch.long),
                "melody_ids": torch.zeros(t, dtype=torch.long),
                "energy": torch.zeros(t), "phoneme_ids": torch.zeros(t, dtype=torch.long),
                "semantic_ids": torch.zeros(t, dtype=torch.long),
            }
        else:
            codes = payload["codes"].long()
            cond = payload["conditioning"]
        t = codes.shape[-1]
        if t < self.min_frames:
            raise ValueError(f"clip {row['tokens']} has only {t} frames (< min_frames={self.min_frames})")
        length = min(t, self.max_frames or t)
        start = random.randint(0, t - length) if self.random_crop and t > length else 0
        end = start + length
        cond = {k: (v[start:end] if k in FRAME_KEYS else v) for k, v in cond.items()}
        return {"codes": codes[:, start:end], "conditioning": cond, "caption": row.get("caption", "")}


def collate_batch(items: list[dict], pad_id: int) -> dict:
    b = len(items)
    q = items[0]["codes"].shape[0]
    max_t = max(item["codes"].shape[-1] for item in items)
    codes = torch.full((b, q, max_t), pad_id, dtype=torch.long)
    frame_mask = torch.zeros((b, max_t), dtype=torch.bool)
    cond: dict[str, torch.Tensor] = {}
    for key in GLOBAL_KEYS:
        cond[key] = torch.stack([item["conditioning"][key].long().reshape(()) for item in items])
    for key in FRAME_KEYS:
        dtype = torch.float32 if key == "energy" else torch.long
        cond[key] = torch.zeros((b, max_t), dtype=dtype)

    for i, item in enumerate(items):
        t = item["codes"].shape[-1]
        codes[i, :, :t] = item["codes"]
        frame_mask[i, :t] = True
        for key in FRAME_KEYS:
            cond[key][i, :t] = item["conditioning"][key].to(cond[key].dtype)
    return {
        "codes": codes,
        "frame_mask": frame_mask,
        "conditioning": cond,
        "captions": [item["caption"] for item in items],
    }
