from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from model import AudioTokenTransformer as _BaseAudioTokenTransformer
from model import ModelConfig


class AudioTokenTransformer(_BaseAudioTokenTransformer):
    """Training-capable Musicm8 transformer.

    The base model owns the architecture and autoregressive generation path.
    This subclass adds the teacher-forced forward pass, masked training loss,
    and compact checkpoint state used by train.py.
    """

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only trainable parameters, excluding the frozen text encoder."""
        trainable = {name for name, param in self.named_parameters() if param.requires_grad}
        return {name: value for name, value in self.state_dict().items() if name in trainable}

    def _drop_text_conditions(self, captions: list[str]) -> list[str]:
        if not self.training or self.cfg.text_condition_dropout <= 0:
            return captions
        p = float(self.cfg.text_condition_dropout)
        return ["" if random.random() < p else text for text in captions]

    def _drop_structured_conditions(
        self,
        cond: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if not self.training or self.cfg.structured_condition_dropout <= 0:
            return cond

        p = float(self.cfg.structured_condition_dropout)
        drop = torch.tensor(
            [random.random() < p for _ in range(batch_size)],
            device=device,
            dtype=torch.bool,
        )
        if not bool(drop.any()):
            return cond

        out: dict[str, torch.Tensor] = {}
        for key, value in cond.items():
            value = value.clone()
            if value.ndim > 0 and value.shape[0] == batch_size:
                value[drop] = 0
            out[key] = value
        return out

    def forward(
        self,
        codes: torch.Tensor,
        captions: list[str],
        cond: dict[str, torch.Tensor],
        *,
        frame_mask: torch.Tensor | None = None,
        audio_context_codes: torch.Tensor | None = None,
        audio_context_known: torch.Tensor | None = None,
        loss_frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced delayed-codebook forward pass used for training."""
        if codes.ndim != 3:
            raise ValueError(f"codes must be [B,Q,T], got {tuple(codes.shape)}")
        b, q, t = codes.shape
        if q != self.cfg.num_codebooks:
            raise ValueError(f"Expected Q={self.cfg.num_codebooks}, got Q={q}")
        if t < 1:
            raise ValueError("codes must contain at least one frame")
        if len(captions) != b:
            raise ValueError(f"Expected {b} captions, got {len(captions)}")

        device = codes.device
        if frame_mask is None:
            frame_mask = torch.ones((b, t), device=device, dtype=torch.bool)
        else:
            frame_mask = frame_mask.to(device=device, dtype=torch.bool)
        if frame_mask.shape != (b, t):
            raise ValueError(f"frame_mask must be {(b, t)}, got {tuple(frame_mask.shape)}")

        delayed, valid = self.pattern.build(codes, frame_mask)
        s = delayed.shape[-1]
        if s > self.cfg.max_seq_len:
            raise ValueError(
                f"Delayed sequence length {s} exceeds max_seq_len={self.cfg.max_seq_len}"
            )

        # Autoregressive teacher forcing: BOS at step 0, previous delayed token
        # column at every later step. This mirrors generate() exactly.
        if s == 1:
            x = self.bos.expand(b, 1, -1)
        else:
            prev = self.embed_code_columns(delayed[:, :, :-1])
            x = torch.cat([self.bos.expand(b, 1, -1), prev], dim=1)

        positions = torch.arange(s, device=device)
        frame_indices = positions.clamp(max=t - 1)
        cond_used = self._drop_structured_conditions(cond, b, device)
        x = x + self.pos(positions)[None]
        x = x + self.frame_conditioner(cond_used, frame_indices)
        x = self.input_dropout(x)

        captions_used = self._drop_text_conditions(captions)
        text_ctx, text_mask = self.encode_text(captions_used, device)
        audio_ctx, audio_mask = self.encode_audio_context(
            audio_context_codes,
            audio_context_known,
        )

        for block in self.blocks:
            x = block(x, text_ctx, text_mask, audio_ctx, audio_mask)

        h = self.final_ln(x)
        logits = torch.stack([head(h) for head in self.heads], dim=1)

        loss_mask = valid
        if loss_frame_mask is not None:
            loss_frame_mask = loss_frame_mask.to(device=device, dtype=torch.bool)
            if loss_frame_mask.shape != (b, t):
                raise ValueError(
                    f"loss_frame_mask must be {(b, t)}, got {tuple(loss_frame_mask.shape)}"
                )
            # Shift frame-level infill mask into each delayed codebook lane.
            _, requested = self.pattern.build(codes, loss_frame_mask & frame_mask)
            loss_mask = valid & requested
            if not bool(loss_mask.any()):
                # Very short clips can fail to produce an infill span; keep the
                # batch trainable rather than returning a zero/NaN objective.
                loss_mask = valid

        return {
            "logits": logits,
            "targets": delayed,
            "loss_mask": loss_mask,
        }

    def masked_cross_entropy(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Cross-entropy over only valid/requested delayed-token positions."""
        logits = out["logits"]
        targets = out["targets"]
        mask = out["loss_mask"].bool()

        if logits.ndim != 4:
            raise ValueError(f"logits must be [B,Q,S,V], got {tuple(logits.shape)}")
        if targets.shape != logits.shape[:3]:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} does not match logits {tuple(logits.shape[:3])}"
            )
        if mask.shape != targets.shape:
            raise ValueError(
                f"loss_mask shape {tuple(mask.shape)} does not match targets {tuple(targets.shape)}"
            )

        vocab = logits.shape[-1]
        per_token = F.cross_entropy(
            logits.reshape(-1, vocab),
            targets.reshape(-1),
            reduction="none",
            ignore_index=self.cfg.codebook_size,
        ).view_as(targets)

        weight = mask.to(per_token.dtype)
        denom = weight.sum().clamp_min(1.0)
        return (per_token * weight).sum() / denom


__all__ = ["AudioTokenTransformer", "ModelConfig"]
