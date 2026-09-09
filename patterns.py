from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DelayPattern:
    """MusicGen-style delayed codebook schedule.

    Codebook q is shifted by delays[q] global autoregressive steps. This lets a
    later codebook for frame t see earlier codebooks for the same frame in its
    causal history while keeping one Transformer step per audio frame (plus a
    short delay tail).
    """

    delays: tuple[int, ...]
    pad_id: int

    def __post_init__(self) -> None:
        if not self.delays:
            raise ValueError("delays must be non-empty")
        if min(self.delays) < 0:
            raise ValueError("delays must be >= 0")

    @property
    def num_codebooks(self) -> int:
        return len(self.delays)

    @property
    def max_delay(self) -> int:
        return max(self.delays)

    def sequence_length(self, frames: int) -> int:
        return int(frames + self.max_delay)

    def build(
        self,
        codes: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if codes.ndim != 3:
            raise ValueError(f"codes must be [B,Q,T], got {tuple(codes.shape)}")
        b, q, t = codes.shape
        if q != self.num_codebooks:
            raise ValueError(f"pattern has {self.num_codebooks} codebooks but got Q={q}")
        if frame_mask is None:
            frame_mask = torch.ones((b, t), device=codes.device, dtype=torch.bool)
        if frame_mask.shape != (b, t):
            raise ValueError(f"frame_mask must be {(b,t)}, got {tuple(frame_mask.shape)}")

        s = self.sequence_length(t)
        delayed = torch.full((b, q, s), self.pad_id, device=codes.device, dtype=codes.dtype)
        valid = torch.zeros((b, q, s), device=codes.device, dtype=torch.bool)
        for qi, delay in enumerate(self.delays):
            delayed[:, qi, delay : delay + t] = codes[:, qi]
            valid[:, qi, delay : delay + t] = frame_mask
            target_slice = delayed[:, qi, delay : delay + t]
            target_slice.masked_fill_(~frame_mask, self.pad_id)
        return delayed, valid

    def revert(self, delayed: torch.Tensor, frames: int) -> torch.Tensor:
        if delayed.ndim != 3:
            raise ValueError("delayed must be [B,Q,S]")
        b, q, _ = delayed.shape
        if q != self.num_codebooks:
            raise ValueError(f"expected Q={self.num_codebooks}, got {q}")
        out = torch.empty((b, q, frames), device=delayed.device, dtype=delayed.dtype)
        for qi, delay in enumerate(self.delays):
            out[:, qi] = delayed[:, qi, delay : delay + frames]
        return out

    def valid_codebooks_at_step(self, step: int, frames: int) -> list[tuple[int, int]]:
        slots: list[tuple[int, int]] = []
        for q, delay in enumerate(self.delays):
            t = step - delay
            if 0 <= t < frames:
                slots.append((q, t))
        return slots


def default_delays(num_codebooks: int, stereo: bool = False) -> tuple[int, ...]:
    if num_codebooks < 1:
        raise ValueError("num_codebooks must be >= 1")
    if stereo:
        if num_codebooks % 2:
            raise ValueError("stereo-interleaved codebooks must be even")
        return tuple(i // 2 for i in range(num_codebooks))
    return tuple(range(num_codebooks))
