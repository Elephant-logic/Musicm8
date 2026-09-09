from __future__ import annotations

from pathlib import Path
import torch

from codec import codec_from_info
from model import AudioTokenTransformer, ModelConfig


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = AudioTokenTransformer(cfg).to(device)
    model.load_trainable_state_dict(ckpt["model"])
    model.eval()
    codec_info = ckpt.get("data_meta", {}).get("codec")
    if not codec_info:
        raise ValueError("V2 generation requires checkpoint data_meta.codec")
    codec = codec_from_info(codec_info, device)
    return ckpt, model, codec
