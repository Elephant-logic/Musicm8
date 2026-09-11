from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from musicm8_synth_v2 import SR, block_compressor, filt, reverb, saturate, three_band_eq


def ensure_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.vstack([x, x])
    if x.shape[0] == 2 and x.shape[1] > 2:
        return x
    if x.ndim == 2 and x.shape[1] >= 2:
        return x[:, :2].T.astype(np.float32)
    mono = x.reshape(-1)
    return np.vstack([mono, mono]).astype(np.float32)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    return audio, int(sr)


def resample_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    mono = np.mean(audio, axis=1).astype(np.float32)
    if sr != SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR, res_type="kaiser_fast").astype(np.float32)
    return mono


def rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(math.sqrt(float(np.mean(np.square(x, dtype=np.float64))) + 1e-12) + 1e-12)


def set_rms(x: np.ndarray, target_db: float, max_up_db: float = 9.0) -> np.ndarray:
    current = rms_db(x)
    gain_db = max(-24.0, min(max_up_db, target_db - current))
    return (x * (10 ** (gain_db / 20.0))).astype(np.float32)


def simple_deesser(stereo: np.ndarray, amount: float = 0.45) -> np.ndarray:
    amount = max(0.0, min(1.0, float(amount)))
    if amount <= 0:
        return stereo
    high = filt(stereo, 5200.0, "high")
    body = stereo - high
    controlled = block_compressor(high, threshold_db=-27.0, ratio=3.5 + 3.0 * amount, makeup_db=0.0)
    return (body + high * (1.0 - 0.22 * amount) + controlled * (0.22 * amount)).astype(np.float32)


def delay_mono(x: np.ndarray, seconds: float) -> np.ndarray:
    samples = max(1, int(seconds * SR))
    out = np.zeros_like(x)
    if samples < x.size:
        out[samples:] = x[:-samples]
    return out


def vocal_activity(mono: np.ndarray) -> np.ndarray:
    block = 512
    n_blocks = max(1, math.ceil(len(mono) / block))
    levels = np.zeros(n_blocks, dtype=np.float32)
    for i in range(n_blocks):
        part = mono[i * block:(i + 1) * block]
        if part.size:
            levels[i] = math.sqrt(float(np.mean(part * part)) + 1e-12)
    levels /= float(levels.max()) + 1e-8
    kernel = np.ones(11, dtype=np.float32) / 11.0
    levels = np.convolve(levels, kernel, mode="same")
    return np.interp(np.arange(len(mono)), np.linspace(0, len(mono) - 1, len(levels)), levels).astype(np.float32)


def process_vocals(raw: np.ndarray, sr: int, bpm: float) -> dict[str, np.ndarray]:
    mono = resample_mono(raw, sr)
    mono = np.nan_to_num(mono)
    mono = filt(mono, 82.0, "high")
    mono = filt(mono, 14500.0, "low")
    lead = np.vstack([mono, mono])
    lead = three_band_eq(lead, low_db=-2.5, mid_db=1.6, high_db=0.8)
    lead = simple_deesser(lead, 0.55)
    lead = block_compressor(lead, threshold_db=-20.0, ratio=3.2, makeup_db=1.0)
    lead = saturate(lead, 0.035)
    lead = set_rms(lead, -20.0, 8.0)

    left_mono = delay_mono(mono, 0.018)
    right_mono = delay_mono(mono, 0.027)
    left_mono = filt(left_mono, 10500.0, "low")
    right_mono = filt(right_mono, 9300.0, "low")
    left = set_rms(np.vstack([left_mono, np.zeros_like(left_mono)]), -31.0, 5.0)
    right = set_rms(np.vstack([np.zeros_like(right_mono), right_mono]), -31.0, 5.0)

    vocal_mix = lead + left + right
    room = reverb(vocal_mix, 0.16)
    beat_delay = max(1, int((60.0 / max(1.0, bpm)) * 0.75 * SR))
    echo = np.zeros_like(vocal_mix)
    if beat_delay < vocal_mix.shape[-1]:
        echo[:, beat_delay:] = vocal_mix[:, :-beat_delay] * 0.10
    vocal_mix = (0.88 * vocal_mix + 0.12 * room + echo).astype(np.float32)
    vocal_mix = block_compressor(vocal_mix, threshold_db=-15.0, ratio=1.8, makeup_db=0.0)
    return {"lead": lead, "double_L": left, "double_R": right, "vocal_mix": vocal_mix}


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[-1] >= n:
        return x[:, :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[-1]))).astype(np.float32)


def score_lock_if_available(project: Path, vocal: Path) -> Path:
    score = project / "vocal_score.json"
    if not score.exists() or vocal.name == "neural_lead_synced.wav":
        return vocal
    synced = project / "vocals" / "neural_lead_synced.wav"
    print("🎯 Locking neural vocal to Musicm8 vocal score before mixing...")
    subprocess.run([
        sys.executable, "-u", str(Path(__file__).resolve().parent / "vocal_sync.py"),
        "--vocal", str(vocal),
        "--score", str(score),
        "--out", str(synced),
    ], check=True)
    return synced


def main() -> None:
    p = argparse.ArgumentParser(description="Process a neural vocal stem and mix it with the polished Musicm8 instrumental.")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--vocal", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    project = args.project
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    backing_path = project / "master.wav"
    if not backing_path.exists():
        raise FileNotFoundError(backing_path)
    if not args.vocal.exists():
        raise FileNotFoundError(args.vocal)

    vocal_path = score_lock_if_available(project, args.vocal)

    instrumental = project / "master_instrumental.wav"
    shutil.copy2(backing_path, instrumental)
    backing_raw, backing_sr = read_audio(backing_path)
    if backing_sr != SR:
        left = librosa.resample(backing_raw[:, 0], orig_sr=backing_sr, target_sr=SR, res_type="kaiser_fast")
        right = librosa.resample(backing_raw[:, 1], orig_sr=backing_sr, target_sr=SR, res_type="kaiser_fast")
        backing = np.vstack([left, right]).astype(np.float32)
    else:
        backing = ensure_stereo(backing_raw)

    raw_vocal, vocal_sr = read_audio(vocal_path)
    parts = process_vocals(raw_vocal, vocal_sr, float(plan.get("bpm", 120.0)))
    n = backing.shape[-1]
    parts = {k: fit_length(v, n) for k, v in parts.items()}
    vocals_dir = project / "vocals"
    vocals_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in parts.items():
        sf.write(vocals_dir / f"{name}.wav", audio.T, SR, subtype="PCM_24")

    activity = vocal_activity(parts["lead"].mean(axis=0))
    duck = 1.0 - 0.16 * np.clip(activity, 0.0, 1.0)
    combined = backing * duck[None] + parts["vocal_mix"]
    combined = block_compressor(combined, threshold_db=-8.5, ratio=1.45, makeup_db=0.0)
    combined = saturate(combined, 0.018)
    peak = float(np.max(np.abs(combined))) + 1e-9
    target = 10 ** (-1.0 / 20.0)
    if peak > target:
        combined *= target / peak

    out = args.out or (project / "master.wav")
    sf.write(out, combined.T, SR, subtype="PCM_24")
    report = {
        "format": "musicm8-vocal-mix-v2",
        "raw_neural_vocal": str(args.vocal),
        "score_locked_vocal": str(vocal_path),
        "instrumental_master": str(instrumental),
        "lead": str(vocals_dir / "lead.wav"),
        "double_L": str(vocals_dir / "double_L.wav"),
        "double_R": str(vocals_dir / "double_R.wav"),
        "vocal_mix": str(vocals_dir / "vocal_mix.wav"),
        "master": str(out),
        "lead_rms_db": round(rms_db(parts["lead"]), 3),
        "notes": "Vocal is score-locked first, then centered/de-essed/compressed with quiet doubles, reverb/delay and light backing ducking.",
    }
    (project / "vocal_mix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("✅ Score-locked vocal:", report["score_locked_vocal"])
    print("✅ Vocal lead:", report["lead"])
    print("✅ Vocal mix:", report["vocal_mix"])
    print("✅ Instrumental preserved:", instrumental)
    print("✅ Final song master:", out)


if __name__ == "__main__":
    main()
