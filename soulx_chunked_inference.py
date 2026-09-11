from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def edge_window(n: int, sr: int, fade_in_s: float = 0.006, fade_out_s: float = 0.030) -> np.ndarray:
    w = np.ones(max(1, n), dtype=np.float32)
    fi = min(len(w) // 3, max(1, int(sr * fade_in_s)))
    fo = min(len(w) // 3, max(1, int(sr * fade_out_s)))
    if fi > 1:
        w[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 1:
        w[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return w


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 SoulX phrase-by-phrase inference with clip-safe overlap-add stitching.")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--prompt-wav", type=Path, required=True)
    p.add_argument("--prompt-metadata", type=Path, required=True)
    p.add_argument("--target-metadata", type=Path, required=True)
    p.add_argument("--phoneset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    # This script is stored in /content/Musicm8 but is launched with cwd set to
    # /content/SoulX-Singer. Python otherwise puts the script directory first and
    # may not discover the sibling SoulX source checkout. The parent runner also
    # sets PYTHONPATH; this local guard makes the import robust on its own.
    cwd = str(Path.cwd().resolve())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    from soulxsinger.models.soulxsinger import SoulXSinger
    from soulxsinger.utils.data_processor import DataProcessor
    from soulxsinger.utils.file_utils import load_config

    config = load_config(str(args.config))
    sr = int(config.audio.sample_rate)
    hop = int(config.audio.hop_size)

    model = SoulXSinger(config).to(args.device)
    checkpoint = torch.load(args.model_path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if args.fp16 and str(args.device).startswith("cuda"):
        model.half()
        model.mel.float()
    model.eval().to(args.device)

    processor = DataProcessor(
        hop_size=hop,
        sample_rate=sr,
        phoneset_path=str(args.phoneset),
        device=args.device,
    )

    prompt_meta_list = json.loads(args.prompt_metadata.read_text(encoding="utf-8"))
    target_meta_list = json.loads(args.target_metadata.read_text(encoding="utf-8"))
    if not prompt_meta_list:
        raise RuntimeError("SoulX prompt metadata is empty")
    if not target_meta_list:
        raise RuntimeError("SoulX target metadata is empty")

    prompt = processor.process(prompt_meta_list[0], str(args.prompt_wav))
    song_n = max(int(round(float(m["time"][1]) * sr / 1000.0)) for m in target_meta_list)
    mix = np.zeros(song_n, dtype=np.float32)
    weight = np.zeros(song_n, dtype=np.float32)
    report = []

    print(f"🎤 Clip-safe SoulX chunking: {len(target_meta_list)} lyric phrases")
    for i, meta in enumerate(target_meta_list):
        a = max(0, int(round(float(meta["time"][0]) * sr / 1000.0)))
        b = min(song_n, int(round(float(meta["time"][1]) * sr / 1000.0)))
        target_n = max(1, b - a)
        target = processor.process(dict(meta), None)
        infer_data = {"prompt": prompt, "target": target}

        with torch.no_grad():
            audio = model.infer(
                infer_data,
                auto_shift=False,
                pitch_shift=0,
                n_steps=int(config.infer.n_steps),
                cfg=float(config.infer.cfg),
                control="score",
                use_fp16=bool(args.fp16),
            )
        y = audio.squeeze().float().cpu().numpy().astype(np.float32)
        generated_n = int(len(y))

        if len(y) > target_n:
            y = y[:target_n]
        elif len(y) < target_n:
            y = np.pad(y, (0, target_n - len(y)))

        w = edge_window(target_n, sr)
        mix[a:b] += y[: b - a] * w[: b - a]
        weight[a:b] += w[: b - a]
        report.append({
            "index": i,
            "start_s": round(a / sr, 5),
            "end_s": round(b / sr, 5),
            "target_samples": target_n,
            "generated_samples": generated_n,
            "trimmed_samples": max(0, generated_n - target_n),
            "padded_samples": max(0, target_n - generated_n),
            "text": str(meta.get("text", "")),
        })
        print(f"  chunk {i+1}/{len(target_meta_list)} {a/sr:.2f}-{b/sr:.2f}s | {meta.get('text','')}")

    mask = weight > 1e-6
    mix[mask] /= weight[mask]
    mix = np.nan_to_num(mix)
    peak = float(np.max(np.abs(mix))) + 1e-9
    if peak > 0.98:
        mix *= 0.98 / peak

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, mix, sr, subtype="PCM_24")
    args.out.with_suffix(".chunks.json").write_text(json.dumps({
        "format": "musicm8-soulx-chunk-stitch-v2",
        "sample_rate": sr,
        "chunks": report,
        "method": "short musical phrase chunks + exact timestamp placement + overlap-add edge fades + no post-hoc word stretching",
        "note": "Each short lyric phrase is generated independently. Excess vocoder tail cannot overwrite the next phrase, and phrase edges are softened without stretching the words.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("✅ Clip-safe chunked SoulX vocal:", args.out)


if __name__ == "__main__":
    main()
