from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SymbolicConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 1024

    def to_dict(self) -> dict:
        return asdict(self)


class SymbolicTransformer(nn.Module):
    def __init__(self, cfg: SymbolicConfig):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_mult * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2:
            raise ValueError(f"ids must be [B,T], got {tuple(ids.shape)}")
        if ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.pos(pos)[None]
        mask = torch.full((ids.shape[1], ids.shape[1]), float("-inf"), device=ids.device)
        mask = torch.triu(mask, diagonal=1)
        x = self.blocks(x, mask=mask)
        return self.head(self.norm(x))

    def loss(self, ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        logits = self(ids[:, :-1])
        target = ids[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=pad_id)

    @torch.no_grad()
    def next_logits(self, ids: torch.Tensor) -> torch.Tensor:
        self.eval()
        if ids.shape[1] > self.cfg.max_seq_len:
            ids = ids[:, -self.cfg.max_seq_len :]
        return self(ids)[:, -1]
