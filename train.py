from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TokenDataset, collate_batch
from model import AudioTokenTransformer, ModelConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_infill_masks(frame_mask: torch.Tensor, min_frac: float, max_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return known-context mask and gap-only loss mask."""
    known = frame_mask.clone()
    loss_mask = torch.zeros_like(frame_mask)
    for i in range(frame_mask.shape[0]):
        length = int(frame_mask[i].sum())
        if length < 4:
            continue
        gap = max(1, int(length * random.uniform(min_frac, max_frac)))
        gap = min(gap, length - 2)
        start = random.randint(1, max(1, length - gap - 1))
        end = start + gap
        known[i, start:end] = False
        loss_mask[i, start:end] = True
    return known, loss_mask


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, cfg: ModelConfig, args, data_meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_config": cfg.to_dict(),
        "model": model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_args": vars(args),
        "data_meta": data_meta,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description="Train Musicm8 V2 acoustic token LM.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/v2-small.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-every", type=int, default=1_000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--infill-prob", type=float, default=0.25,
                   help="Fraction of batches trained as suffix-aware masked-span infilling.")
    p.add_argument("--infill-min-frac", type=float, default=0.08)
    p.add_argument("--infill-max-frac", type=float, default=0.30)
    args = p.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")

    cfg_dict = json.loads(args.config.read_text(encoding="utf-8"))
    ds_probe = TokenDataset(args.data, random_crop=False)
    data_meta = ds_probe.meta
    codec_meta = data_meta.get("codec", {})
    if codec_meta:
        cfg_dict["codebook_size"] = int(codec_meta["codebook_size"])
        cfg_dict["num_codebooks"] = int(codec_meta["num_codebooks"])
        cfg_dict["delays"] = data_meta.get("delays")
    cfg = ModelConfig(**cfg_dict)

    max_delay = max(cfg.resolved_delays())
    max_frames = cfg.max_seq_len - max_delay
    if max_frames < 2:
        raise ValueError("max_seq_len is too small for the selected delay pattern")
    ds = TokenDataset(args.data, max_frames=max_frames, random_crop=True)
    first = ds[0]["codes"]
    if first.shape[0] != cfg.num_codebooks:
        raise ValueError(f"dataset Q={first.shape[0]} but model Q={cfg.num_codebooks}")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        collate_fn=partial(collate_batch, pad_id=cfg.codebook_size),
        persistent_workers=args.num_workers > 0,
    )
    if len(loader) == 0:
        raise ValueError("Dataset is smaller than --batch-size")

    model = AudioTokenTransformer(cfg).to(device)
    trainable = [x for x in model.parameters() if x.requires_grad]
    print(f"Device: {device}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    print(f"Codec: {codec_meta or 'legacy'}")
    print(f"Delay pattern: {cfg.resolved_delays()}")
    print(f"Max raw frames/window: {max_frames}")

    opt = AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max(step, 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = LambdaLR(opt, lr_lambda)
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt["step"])
        print(f"Resumed from step {start_step}")

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext
    args.out.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    running = 0.0
    infill_running = 0
    opt.zero_grad(set_to_none=True)

    bar = tqdm(range(start_step + 1, args.steps + 1), initial=start_step, total=args.steps, desc="train")
    for step in bar:
        accum = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            codes = batch["codes"].to(device, non_blocking=True)
            frame_mask = batch["frame_mask"].to(device, non_blocking=True)
            cond = {k: v.to(device, non_blocking=True) for k, v in batch["conditioning"].items()}

            do_infill = random.random() < args.infill_prob
            if do_infill:
                known, loss_mask = make_infill_masks(frame_mask, args.infill_min_frac, args.infill_max_frac)
                audio_ctx_codes = codes
                audio_ctx_known = known
                infill_running += 1
            else:
                loss_mask = None
                audio_ctx_codes = None
                audio_ctx_known = None

            with amp_ctx():
                out = model(
                    codes, batch["captions"], cond, frame_mask=frame_mask,
                    audio_context_codes=audio_ctx_codes,
                    audio_context_known=audio_ctx_known,
                    loss_frame_mask=loss_mask,
                )
                loss = model.masked_cross_entropy(out)
                scaled = loss / args.grad_accum
            scaled.backward()
            accum += float(loss.detach())

        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        running += accum / args.grad_accum

        if step % args.log_every == 0:
            avg = running / args.log_every
            bar.set_postfix(loss=f"{avg:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}", infill=infill_running)
            running = 0.0; infill_running = 0
        if step % args.save_every == 0:
            save_checkpoint(args.out / f"step_{step:07d}.pt", model, opt, sched, step, cfg, args, data_meta)
            save_checkpoint(args.out / "latest.pt", model, opt, sched, step, cfg, args, data_meta)

    save_checkpoint(args.out / "final.pt", model, opt, sched, args.steps, cfg, args, data_meta)
    save_checkpoint(args.out / "latest.pt", model, opt, sched, args.steps, cfg, args, data_meta)
    print(f"Finished: {args.out/'latest.pt'}")


if __name__ == "__main__":
    main()
