from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from codec import create_codec
from conditioning import apply_events, build_condition_tensors, serialize_caption
from patterns import default_delays
from phonemes import expand_lyric_events_to_phonemes


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_manifest(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "audio" not in row:
                raise ValueError(f"{path}:{line_no}: each row needs 'audio'")
            yield row


def normalize_audio(wav: torch.Tensor, sr: int, target_sr: int, channels: int) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if channels == 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif channels == 2:
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        else:
            wav = wav[:2]
    return wav


def frame_energy(wav: torch.Tensor, frames: int) -> torch.Tensor:
    mono = wav.mean(dim=0)
    if frames <= 0:
        return torch.empty(0)
    bounds = torch.linspace(0, mono.numel(), frames + 1).long()
    vals = []
    for i in range(frames):
        x = mono[bounds[i] : bounds[i + 1]]
        vals.append(x.square().mean().sqrt() if x.numel() else torch.tensor(0.0))
    e = torch.stack(vals).float()
    scale = torch.quantile(e, 0.95).clamp_min(1e-5)
    return (e / scale).clamp(0, 1)


def semantic_for_clip(row: dict, frames: int, frame_rate: float, clip_start: float, manifest_dir: Path | None = None) -> torch.Tensor:
    target = torch.zeros(frames, dtype=torch.long)
    sidecar = row.get("semantic_tokens")
    if sidecar:
        path = Path(sidecar)
        if not path.is_absolute() and manifest_dir is not None:
            path = (manifest_dir / path).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        ids = payload["ids"].long().view(-1)
        source_rate = float(payload["frame_rate"])
        a = max(0, int(round(clip_start * source_rate)))
        b = min(ids.numel(), int(round((clip_start + frames / frame_rate) * source_rate)))
        piece = ids[a:b]
        if piece.numel():
            target = torch.nn.functional.interpolate(piece.float().view(1,1,-1), size=frames, mode="nearest").long().view(-1)
        return target
    apply_events(
        target, row.get("semantic", []), clip_start, frame_rate,
        lambda e: int(e.get("id", e.get("token", 0))),
    )
    return target


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-tokenize audio and aligned musical controls.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--codec", choices=["encodec24", "encodec32", "dac", "musiccodec"], default="encodec24")
    p.add_argument("--channels", type=int, choices=[1, 2], default=1)
    p.add_argument("--bandwidth", type=float, default=3.0, choices=[1.5, 3.0, 6.0, 12.0, 24.0])
    p.add_argument("--dac-model", default="44khz", choices=["16khz", "24khz", "44khz"])
    p.add_argument("--dac-quantizers", type=int, default=None)
    p.add_argument("--custom-codec", type=Path, default=None, help="Checkpoint from train_music_codec.py")
    p.add_argument("--clip-seconds", type=float, default=12.0)
    p.add_argument("--stride-seconds", type=float, default=None)
    p.add_argument("--keep-tail", action="store_true")
    p.add_argument("--phonemize", action="store_true", help="Expand timed lyric events into phoneme events.")
    p.add_argument("--language", default="en-us")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = p.parse_args()

    if args.clip_seconds <= 0:
        raise ValueError("--clip-seconds must be > 0")
    stride_seconds = args.stride_seconds or args.clip_seconds
    device = choose_device(args.device)
    codec = create_codec(
        args.codec, device,
        channels=args.channels,
        bandwidth=args.bandwidth,
        model_type=args.dac_model,
        n_quantizers=args.dac_quantizers,
        custom_checkpoint=str(args.custom_codec) if args.custom_codec else None,
    )
    sr = codec.info.sample_rate
    clip_samples = round(args.clip_seconds * sr)
    stride_samples = round(stride_seconds * sr)

    args.out.mkdir(parents=True, exist_ok=True)
    clip_dir = args.out / "clips"
    clip_dir.mkdir(exist_ok=True)
    index_path = args.out / "index.jsonl"

    delays = default_delays(codec.info.num_codebooks, stereo=codec.info.stereo_interleaved)
    meta = {
        "version": 2,
        "codec": codec.info.to_dict(),
        "delays": list(delays),
        "clip_seconds": args.clip_seconds,
        "stride_seconds": stride_seconds,
        "format": "dict-v2",
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rows = list(iter_manifest(args.manifest))
    written = 0
    with index_path.open("w", encoding="utf-8") as index_f:
        for original_row in tqdm(rows, desc="songs"):
            row = dict(original_row)
            audio_path = Path(row["audio"])
            if not audio_path.is_absolute():
                audio_path = (args.manifest.parent / audio_path).resolve()
            wav, source_sr = torchaudio.load(audio_path)
            wav = normalize_audio(wav, source_sr, sr, args.channels)
            total = wav.shape[-1]

            if args.phonemize and row.get("lyric_events") and not row.get("phonemes"):
                row["phonemes"] = expand_lyric_events_to_phonemes(row["lyric_events"], args.language)

            starts = list(range(0, max(total - clip_samples + 1, 0), stride_samples))
            if total >= clip_samples:
                last = total - clip_samples
                if not starts or last > starts[-1]:
                    starts.append(last)
            elif args.keep_tail:
                starts = [0]

            for start in starts:
                chunk = wav[:, start : start + clip_samples]
                if chunk.shape[-1] < clip_samples:
                    if not args.keep_tail:
                        continue
                    chunk = torch.nn.functional.pad(chunk, (0, clip_samples - chunk.shape[-1]))

                codes = codec.encode(chunk, sr).long()
                frames = int(codes.shape[-1])
                start_seconds = start / sr
                energy = frame_energy(chunk, frames)
                semantic = semantic_for_clip(row, frames, codec.info.frame_rate, start_seconds, args.manifest.parent)
                cond = build_condition_tensors(
                    row, frames, codec.info.frame_rate, start_seconds,
                    energy=energy, semantic_ids=semantic,
                )
                payload = {"codes": codes.to(torch.int16), "conditioning": cond}

                uid_src = f"{audio_path}|{start}|{args.clip_seconds}|{codec.info.name}|{codec.info.num_codebooks}|{args.channels}"
                uid = hashlib.sha1(uid_src.encode("utf-8")).hexdigest()[:20]
                rel = Path("clips") / f"{uid}.pt"
                torch.save(payload, args.out / rel)
                out_row = {
                    "tokens": rel.as_posix(),
                    "caption": serialize_caption(row),
                    "source_audio": str(audio_path),
                    "start_seconds": start_seconds,
                    "clip_seconds": args.clip_seconds,
                    "frames": frames,
                    "num_codebooks": int(codes.shape[0]),
                    "stem": row.get("stem", "mix"),
                }
                index_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1

    if written == 0:
        raise RuntimeError("No clips were written. Use longer source audio or --keep-tail.")
    print(f"Wrote {written} clips to {args.out}")
    print(f"Codec: {codec.info}")
    print(f"Delay pattern: {delays}")


if __name__ == "__main__":
    main()
