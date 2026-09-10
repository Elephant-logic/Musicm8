from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from midi_tokens import (
    BAR, BOS, EOS, VOCAB_SIZE, advance_phase, save_tokens_as_midi,
    valid_next_tokens,
)
from symbolic_model import SymbolicConfig, SymbolicTransformer


def sample_allowed(
    logits: torch.Tensor,
    allowed: list[int],
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> int:
    if not allowed:
        raise RuntimeError("No legal MIDI tokens are available for the current grammar state")
    idx = torch.tensor(allowed, device=logits.device, dtype=torch.long)
    vals = logits.index_select(0, idx) / max(temperature, 1e-4)
    if 0 < top_k < vals.numel():
        top_vals, top_idx = torch.topk(vals, top_k)
        probs = torch.softmax(top_vals, dim=-1)
        chosen = torch.multinomial(probs, 1, generator=generator)
        return int(idx[top_idx[chosen]].item())
    probs = torch.softmax(vals, dim=-1)
    chosen = torch.multinomial(probs, 1, generator=generator)
    return int(idx[chosen].item())


def split_tracks(combined: Path, out_dir: Path) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(combined))
    _, tempi = pm.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) else 120.0
    out_dir.mkdir(parents=True, exist_ok=True)
    for inst in pm.instruments:
        one = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        copied = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        copied.notes = list(inst.notes)
        copied.control_changes = list(inst.control_changes)
        copied.pitch_bends = list(inst.pitch_bends)
        one.instruments.append(copied)
        name = (inst.name or ("drums" if inst.is_drum else "track")).lower().replace(" ", "_")
        one.write(str(out_dir / f"{name}.mid"))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate editable Musicm8 MIDI arrangement.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bars", type=int, default=8)
    p.add_argument("--bpm", type=float, default=None)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.bars < 1:
        raise ValueError("--bars must be >= 1")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = SymbolicConfig(**ck["config"])
    if cfg.vocab_size != VOCAB_SIZE:
        raise ValueError(f"Checkpoint vocab {cfg.vocab_size} != current vocab {VOCAB_SIZE}")
    model = SymbolicTransformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    bpm = float(args.bpm or ck.get("median_bpm", 120.0))

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    tokens = [BOS, BAR]
    phase = "event_or_bar"
    bars_seen = 1
    events_this_bar = 0
    min_events_per_bar = 6
    max_events_per_bar = 36
    max_tokens = min(cfg.max_seq_len, 2 + args.bars * 180)

    while len(tokens) < max_tokens and phase != "done":
        ids = torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)
        logits = model.next_logits(ids)[0]
        allowed = valid_next_tokens(phase, allow_eos=True)
        if phase == "event_or_bar":
            if bars_seen < args.bars:
                # Every requested bar gets real content. This also prevents an
                # undertrained model from emitting a run of empty BAR tokens.
                allowed = [x for x in allowed if x != EOS]
                if events_this_bar < min_events_per_bar:
                    allowed = [x for x in allowed if x != BAR]
                elif events_this_bar >= max_events_per_bar:
                    allowed = [BAR]
            else:
                # Final requested bar: no extra BAR tokens, and do not end it empty.
                allowed = [x for x in allowed if x != BAR]
                if events_this_bar < min_events_per_bar:
                    allowed = [x for x in allowed if x != EOS]
                elif events_this_bar >= max_events_per_bar:
                    allowed = [EOS]
        token = sample_allowed(logits, allowed, args.temperature, args.top_k, generator)
        if token == BAR:
            bars_seen += 1
            events_this_bar = 0
        elif phase == "vel":
            events_this_bar += 1
        tokens.append(token)
        phase = advance_phase(token, phase)

    if tokens[-1] != EOS:
        # Never leave a half event in the MIDI parser. Trim back to a complete boundary.
        while tokens and phase not in ("event_or_bar", "done"):
            tokens.pop()
            phase = "start"
            for tok in tokens[1:]:
                phase = advance_phase(tok, phase)
        tokens.append(EOS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_tokens_as_midi(tokens, args.out, bpm=bpm, max_bars=args.bars)
    split_tracks(args.out, args.out.parent / "midi_stems")
    print(f"✅ MIDI arrangement: {args.out}")
    print(f"✅ MIDI stems: {args.out.parent/'midi_stems'}")
    print(f"Bars: {min(bars_seen, args.bars)} | BPM: {bpm:.1f} | Tokens: {len(tokens)}")


if __name__ == "__main__":
    main()
