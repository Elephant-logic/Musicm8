from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


def load_mono(path: Path, sr: int = 24000) -> np.ndarray:
    y, in_sr = sf.read(path, always_2d=True, dtype="float32")
    mono = np.mean(y, axis=1).astype(np.float32)
    if int(in_sr) != sr:
        mono = librosa.resample(mono, orig_sr=int(in_sr), target_sr=sr, res_type="kaiser_fast").astype(np.float32)
    return np.nan_to_num(mono)


def f0_for(path: Path, sr: int = 24000, hop: int = 480) -> np.ndarray:
    y = load_mono(path, sr)
    f0, voiced, prob = librosa.pyin(y, fmin=65.0, fmax=900.0, sr=sr, frame_length=2048, hop_length=hop)
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    voiced = np.asarray(voiced, dtype=bool)
    prob = np.nan_to_num(prob, nan=0.0)
    f0[(~voiced) | (prob < 0.30)] = 0.0
    wanted = max(1, int(np.ceil(len(y) / hop)))
    if len(f0) < wanted:
        f0 = np.pad(f0, (0, wanted - len(f0)))
    return f0[:wanted]


def main() -> None:
    p = argparse.ArgumentParser(description="SoulX-Singer-SVC timbre conversion for an already score-correct Musicm8 guide vocal.")
    p.add_argument("--soulx-root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True, help="Clean authorized voice reference, ideally 5-30 seconds of dry singing.")
    p.add_argument("--target", type=Path, required=True, help="Score-controlled guide singing to convert.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=24)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(args.soulx_root))
    from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC
    from soulxsinger.utils.audio_utils import load_wav
    from soulxsinger.utils.file_utils import load_config

    config = load_config(str(args.soulx_root / "soulxsinger/config/soulxsinger.yaml"))
    model = SoulXSingerSVC(config).to("cuda")
    checkpoint = torch.load(args.model, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.half()
    model.mel.float()
    model.eval().to("cuda")

    sr = int(config.audio.sample_rate)
    pt_wav = load_wav(str(args.prompt), sr).to("cuda")
    gt_wav = load_wav(str(args.target), sr).to("cuda")
    pt_f0 = torch.from_numpy(f0_for(args.prompt, sr=sr)).unsqueeze(0).to("cuda")
    gt_f0 = torch.from_numpy(f0_for(args.target, sr=sr)).unsqueeze(0).to("cuda")

    with torch.no_grad():
        generated_audio, shift = model.infer(
            pt_wav=pt_wav,
            gt_wav=gt_wav,
            pt_f0=pt_f0,
            gt_f0=gt_f0,
            auto_shift=True,
            pitch_shift=0,
            n_steps=max(8, min(40, int(args.steps))),
            cfg=3.0,
            use_fp16=True,
        )
    audio = generated_audio.squeeze().float().cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, sr, subtype="PCM_24")
    args.out.with_suffix(".json").write_text(json.dumps({
        "format": "musicm8-soulx-svc-v1",
        "prompt": str(args.prompt),
        "target": str(args.target),
        "output": str(args.out),
        "pitch_shift": int(shift),
        "note": "Zero-shot timbre conversion. Use only a voice reference you own or have permission to use.",
    }, indent=2), encoding="utf-8")
    print("✅ SoulX voice clone/conversion:", args.out)
    print("Auto register shift:", shift)


if __name__ == "__main__":
    main()
