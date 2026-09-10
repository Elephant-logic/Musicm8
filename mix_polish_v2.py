from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from musicm8_synth_v2 import SR, block_compressor, filt, saturate, set_rms, three_band_eq


def ensure_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.vstack([x, x])
    if x.shape[1] == 1:
        return np.vstack([x[:, 0], x[:, 0]])
    return x[:, :2].T.astype(np.float32)


def delay(x: np.ndarray, samples: int) -> np.ndarray:
    y = np.zeros_like(x)
    if samples <= 0:
        return x.copy()
    if samples < len(x):
        y[samples:] = x[:-samples]
    return y


def true_stereo_from_mono(stereo: np.ndarray, width: float, low_mono_hz: float = 180.0) -> np.ndarray:
    """Create real side information while keeping low frequencies mono and mono-sum stable."""
    width = max(0.0, min(1.0, float(width)))
    mono = stereo.mean(axis=0)
    if width <= 1e-5:
        return np.vstack([mono, mono]).astype(np.float32)

    low = filt(mono, low_mono_hz, "low")
    high = mono - low
    d1 = max(1, int((0.006 + 0.008 * width) * SR))
    d2 = max(1, int((0.011 + 0.012 * width) * SR))
    side = (delay(high, d1) - delay(high, d2)) * (0.24 * width)
    left = mono + side
    right = mono - side
    return np.vstack([left, right]).astype(np.float32)


def highpass(stereo: np.ndarray, cutoff: float) -> np.ndarray:
    return filt(stereo, cutoff, "high")


def polish_role(role: str, audio: np.ndarray) -> np.ndarray:
    if role == "drums":
        y = highpass(audio, 25.0)
        y = three_band_eq(y, low_db=-0.8, mid_db=0.3, high_db=1.0)
        y = set_rms(y, -18.5, max_gain_db=5.0)
        return true_stereo_from_mono(y, 0.18, low_mono_hz=500.0)
    if role == "bass":
        y = highpass(audio, 27.0)
        y = three_band_eq(y, low_db=-1.5, mid_db=0.8, high_db=0.0)
        y = set_rms(y, -21.5, max_gain_db=4.0)
        return true_stereo_from_mono(y, 0.03, low_mono_hz=220.0)
    if role == "chords":
        y = highpass(audio, 115.0)
        y = three_band_eq(y, low_db=-2.5, mid_db=1.2, high_db=1.1)
        y = set_rms(y, -20.5, max_gain_db=7.0)
        return true_stereo_from_mono(y, 0.68, low_mono_hz=180.0)
    if role == "melody":
        y = highpass(audio, 145.0)
        y = three_band_eq(y, low_db=-3.0, mid_db=1.0, high_db=1.4)
        y = set_rms(y, -20.0, max_gain_db=7.0)
        return true_stereo_from_mono(y, 0.48, low_mono_hz=220.0)
    return audio


def band_ratios(x: np.ndarray) -> dict[str, float]:
    mono = x.mean(axis=0)
    n = min(len(mono), SR * 45)
    if n < 4096:
        return {}
    mono = mono[:n]
    spec = np.fft.rfft(mono * np.hanning(n))
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    total = float(power.sum()) + 1e-12
    def band(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(power[m].sum() / total)
    return {
        "sub_20_80": round(band(20, 80), 4),
        "bass_80_250": round(band(80, 250), 4),
        "low_mid_250_1000": round(band(250, 1000), 4),
        "mid_1k_4k": round(band(1000, 4000), 4),
        "high_4k_12k": round(band(4000, 12000), 4),
    }


def stereo_stats(x: np.ndarray) -> dict[str, float]:
    left, right = x[0], x[1]
    corr = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 1e-8 and np.std(right) > 1e-8 else 1.0
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    ratio = math.sqrt(float(np.mean(side * side)) + 1e-12) / (math.sqrt(float(np.mean(mid * mid)) + 1e-12) + 1e-12)
    return {"lr_correlation": round(corr, 5), "side_to_mid_rms": round(ratio, 5)}


def main() -> None:
    p = argparse.ArgumentParser(description="Polish Musicm8 Synth v2 stems into a wider, better-balanced master.")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    project = args.project
    stems_dir = project / "audio_stems"
    if not stems_dir.exists():
        raise FileNotFoundError(stems_dir)

    before = project / "master.wav"
    if before.exists():
        shutil.copy2(before, project / "master_prepolish.wav")

    polished_dir = project / "audio_stems_polished"
    polished_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, np.ndarray] = {}
    max_n = 0

    for role in ("drums", "bass", "chords", "melody"):
        path = stems_dir / f"{role}.wav"
        if not path.exists():
            continue
        audio, sr = sf.read(path, always_2d=True)
        if sr != SR:
            raise ValueError(f"Expected {SR} Hz for {path}, got {sr}")
        stereo = ensure_stereo(audio)
        polished = polish_role(role, stereo)
        rendered[role] = polished
        max_n = max(max_n, polished.shape[-1])
        sf.write(polished_dir / f"{role}.wav", polished.T, SR, subtype="PCM_24")

    if not rendered:
        raise RuntimeError("No stems found to polish")

    mix = np.zeros((2, max_n), dtype=np.float32)
    for role, audio in rendered.items():
        mix[:, : audio.shape[-1]] += audio

    # Gentle bus processing. Preserve punch instead of normalizing every sample into a wall.
    mix = highpass(mix, 22.0)
    mix = three_band_eq(mix, low_db=-0.8, mid_db=0.5, high_db=0.7)
    mix = block_compressor(mix, threshold_db=-9.5, ratio=1.55, makeup_db=0.0)
    mix = saturate(mix, 0.025)

    peak = float(np.max(np.abs(mix))) + 1e-9
    target = 10 ** (-1.0 / 20.0)
    if peak > target:
        mix *= target / peak

    out = args.out or (project / "master.wav")
    sf.write(out, mix.T, SR, subtype="PCM_24")

    report = {
        "format": "musicm8-mix-polish-v1",
        "master": str(out),
        "prepolish": str(project / "master_prepolish.wav") if (project / "master_prepolish.wav").exists() else None,
        "polished_stems": str(polished_dir),
        "band_ratios": band_ratios(mix),
        "stereo": stereo_stats(mix),
        "notes": "Bass/sub are kept near-mono; chords and melody receive true mid/side-safe stereo decorrelation. Low-end balance is reduced and musical mids/highs are brought forward.",
    }
    (project / "mix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("✅ Polished master:", out)
    print("✅ Pre-polish master:", project / "master_prepolish.wav")
    print("✅ Polished stems:", polished_dir)
    print("Band balance:", report["band_ratios"])
    print("Stereo:", report["stereo"])


if __name__ == "__main__":
    main()
