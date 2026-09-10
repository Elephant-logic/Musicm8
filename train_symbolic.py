from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from midi_tokens import (
    BAR, BOS, EOS, PAD, PITCH_BASE, N_PITCH, TRACKS, TRACK_BASE, VOCAB_SIZE,
    encode_midi,
)
from symbolic_model import SymbolicConfig, SymbolicTransformer


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_bars(tokens: list[int], max_seq_len: int) -> list[list[int]]:
    body = [t for t in tokens if t not in (BOS, EOS, PAD)]
    bars: list[list[int]] = []
    cur: list[int] = []
    for tok in body:
        if tok == BAR:
            if cur:
                bars.append(cur)
            cur = [BAR]
        elif cur:
            cur.append(tok)
    if cur:
        bars.append(cur)
    out: list[list[int]] = []
    pack: list[int] = [BOS]
    for bar in bars:
        if len(pack) + len(bar) + 1 > max_seq_len and len(pack) > 1:
            out.append(pack + [EOS])
            # overlap one bar when possible so transitions are seen more than once
            pack = [BOS]
        if len(bar) + 2 <= max_seq_len:
            pack.extend(bar)
    if len(pack) > 1:
        out.append(pack + [EOS])
    return out


def transpose_tokens(tokens: list[int], semitones: int) -> list[int]:
    if semitones == 0:
        return list(tokens)
    out = list(tokens)
    track = None
    for i, tok in enumerate(out):
        if TRACK_BASE <= tok < TRACK_BASE + len(TRACKS):
            track = tok - TRACK_BASE
        elif PITCH_BASE <= tok < PITCH_BASE + N_PITCH:
            if track != TRACKS["drums"]:
                pitch = tok - PITCH_BASE
                out[i] = PITCH_BASE + max(0, min(127, pitch + semitones))
    return out


class Windows(Dataset):
    def __init__(self, sequences: list[list[int]], max_len: int):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        return torch.tensor(seq[: self.max_len], dtype=torch.long)


def collate(items: list[torch.Tensor]) -> torch.Tensor:
    m = max(x.numel() for x in items)
    out = torch.full((len(items), m), PAD, dtype=torch.long)
    for i, x in enumerate(items):
        out[i, : x.numel()] = x
    return out


def build_sequences(index_path: Path, max_seq_len: int) -> tuple[list[list[int]], float]:
    rows = [json.loads(x) for x in index_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    sequences: list[list[int]] = []
    bpms: list[float] = []
    for row in rows:
        midi = Path(row["midi"]["arrangement"])
        if not midi.exists():
            continue
        tokens, bpm = encode_midi(midi)
        bpms.append(float(bpm))
        base = split_bars(tokens, max_seq_len)
        for shift in (-5, -2, 0, 2, 5):
            sequences.extend(transpose_tokens(seq, shift) for seq in base)
    if not sequences:
        raise RuntimeError(f"No MIDI sequences found from {index_path}")
    median_bpm = sorted(bpms)[len(bpms) // 2] if bpms else 120.0
    return sequences, median_bpm


def save(path: Path, model, opt, sched, step: int, cfg: SymbolicConfig, median_bpm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "median_bpm": median_bpm,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description="Train Musicm8's small DAW/MIDI arrangement model.")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--max-seq-len", type=int, default=768)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = SymbolicConfig(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        max_seq_len=args.max_seq_len,
    )
    sequences, median_bpm = build_sequences(args.index, cfg.max_seq_len)
    ds = Windows(sequences, cfg.max_seq_len)
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True, collate_fn=collate, drop_last=False)
    print(f"Device: {device}")
    print(f"MIDI windows: {len(ds)} (includes pitch transposition augmentation)")
    print(f"Median BPM: {median_bpm:.1f}")

    model = SymbolicTransformer(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in params)/1e6:.2f}M")
    opt = AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    def schedule(step: int) -> float:
        if step < args.warmup:
            return max(step, 1) / max(args.warmup, 1)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = LambdaLR(opt, schedule)
    start = 0
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        old_cfg = SymbolicConfig(**ck["config"])
        if old_cfg.to_dict() != cfg.to_dict():
            raise ValueError("Resume checkpoint config does not match requested symbolic model config")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start = int(ck.get("step", 0))
        median_bpm = float(ck.get("median_bpm", median_bpm))
        print(f"Resumed from step {start}")

    iterator = iter(loader)
    running = 0.0
    bar = tqdm(range(start + 1, args.steps + 1), initial=start, total=args.steps, desc="symbolic")
    for step in bar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device)
        loss = model.loss(batch, pad_id=PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        running += float(loss.detach())
        if step % args.log_every == 0:
            avg = running / args.log_every
            running = 0.0
            bar.set_postfix(loss=f"{avg:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")
        if step % args.save_every == 0:
            save(args.out / f"step_{step:07d}.pt", model, opt, sched, step, cfg, median_bpm)
            save(args.out / "latest.pt", model, opt, sched, step, cfg, median_bpm)

    save(args.out / "latest.pt", model, opt, sched, args.steps, cfg, median_bpm)
    print(f"✅ Symbolic checkpoint: {args.out/'latest.pt'}")


if __name__ == "__main__":
    main()
