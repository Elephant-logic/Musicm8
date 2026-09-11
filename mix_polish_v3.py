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
    if x.shape[0] == 2 and x.shape[1] > 2:
        return x
    if x.ndim == 2 and x.shape[1] >= 2:
        return x[:, :2].T.astype(np.float32)
    mono = x.reshape(-1)
    return np.vstack([mono, mono]).astype(np.float32)


def fit(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[-1] >= n:
        return x[:, :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[-1]))).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(math.sqrt(float(np.mean(np.square(x, dtype=np.float64))) + 1e-12) + 1e-12)


def mix_profile(plan: dict) -> dict:
    p = dict(plan.get("production", {}).get("mix", {}))
    defaults = {
        "drums_db": -17.0,
        "bass_db": -19.0,
        "chords_db": -24.0,
        "melody_db": -24.5,
        "bus_ratio": 1.35,
        "bus_drive": 0.012,
    }
    for k, v in defaults.items():
        p.setdefault(k, v)
    # Foundation limits: loud/bright processing must not hide timing or tuning problems.
    p["drums_db"] = float(np.clip(float(p["drums_db"]), -20.0, -14.5))
    p["bass_db"] = float(np.clip(float(p["bass_db"]), -22.0, -16.5))
    p["chords_db"] = float(np.clip(float(p["chords_db"]), -27.0, -20.0))
    p["melody_db"] = float(np.clip(float(p["melody_db"]), -28.0, -21.0))
    p["bus_ratio"] = float(np.clip(float(p["bus_ratio"]), 1.1, 1.6))
    p["bus_drive"] = float(np.clip(float(p["bus_drive"]), 0.0, 0.025))
    return p


def center(x: np.ndarray) -> np.ndarray:
    # No Haas/delay widening in foundation mode. A delayed copy can make perfectly
    # quantized sources sound late or phasey on phones/headphones.
    mono = x.mean(axis=0)
    return np.vstack([mono, mono]).astype(np.float32)


def process(role: str, x: np.ndarray, profile: dict, style: str) -> np.ndarray:
    y = center(x)
    if role == "drums":
        y = filt(y, 28.0, "high")
        y = three_band_eq(y, low_db=0.0, mid_db=0.2, high_db=0.3)
        y = saturate(y, 0.010)
        y = block_compressor(y, threshold_db=-16.0, ratio=2.2, makeup_db=0.0)
        return set_rms(y, float(profile["drums_db"]), max_gain_db=5.0)
    if role == "bass":
        y = filt(y, 28.0, "high")
        y = filt(y, 6500.0, "low")
        y = three_band_eq(y, low_db=0.3, mid_db=0.2, high_db=-1.0)
        y = saturate(y, 0.014)
        y = block_compressor(y, threshold_db=-19.0, ratio=2.0, makeup_db=0.0)
        return set_rms(y, float(profile["bass_db"]), max_gain_db=5.0)
    if role == "chords":
        y = filt(y, 120.0, "high")
        y = three_band_eq(y, low_db=-1.5, mid_db=0.3, high_db=0.0)
        y = block_compressor(y, threshold_db=-22.0, ratio=1.5, makeup_db=0.0)
        return set_rms(y, float(profile["chords_db"]), max_gain_db=5.0)
    if role == "melody":
        y = filt(y, 150.0, "high")
        y = three_band_eq(y, low_db=-2.0, mid_db=0.5, high_db=0.2)
        y = block_compressor(y, threshold_db=-22.0, ratio=1.5, makeup_db=0.0)
        return set_rms(y, float(profile["melody_db"]), max_gain_db=5.0)
    return y


def spectral_report(x: np.ndarray) -> dict:
    mono = x.mean(axis=0)
    n = min(len(mono), SR * 30)
    if n < 4096:
        return {}
    y = mono[:n] * np.hanning(n)
    p = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    total = float(p.sum()) + 1e-12
    def band(a: float, b: float) -> float:
        m = (f >= a) & (f < b)
        return float(p[m].sum() / total)
    return {
        "sub": round(band(20, 80), 4),
        "bass": round(band(80, 250), 4),
        "low_mid": round(band(250, 1000), 4),
        "mid": round(band(1000, 4000), 4),
        "high": round(band(4000, 12000), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 timing-safe foundation mix v4.")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    project = args.project
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    style = str(plan.get("style", "electronic"))
    profile = mix_profile(plan)
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
        audio, sr = sf.read(path, always_2d=True, dtype="float32")
        if int(sr) != SR:
            raise ValueError(f"Expected {SR} Hz for {path}, got {sr}")
        y = process(role, ensure_stereo(audio), profile, style)
        rendered[role] = y
        max_n = max(max_n, y.shape[-1])
        sf.write(polished_dir / f"{role}.wav", y.T, SR, subtype="PCM_24")
    if not rendered:
        raise RuntimeError("No stems found")

    mix = np.zeros((2, max_n), dtype=np.float32)
    for y in rendered.values():
        mix += fit(y, max_n)

    mix = filt(mix, 22.0, "high")
    mix = block_compressor(mix, threshold_db=-8.5, ratio=float(profile["bus_ratio"]), makeup_db=0.0)
    mix = saturate(mix, float(profile["bus_drive"]))
    target = 10 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(mix))) + 1e-9
    if peak > target:
        mix *= target / peak

    out = args.out or project / "master.wav"
    sf.write(out, mix.T, SR, subtype="PCM_24")
    report = {
        "format": "musicm8-foundation-mix-v4",
        "style": style,
        "profile": profile,
        "master": str(out),
        "master_rms_db": round(rms_db(mix), 3),
        "spectrum": spectral_report(mix),
        "timing_safe": True,
        "delay_based_widening": False,
        "reference_master": False,
        "note": "Foundation mix keeps every stem centred and avoids Haas delays/reference remastering so timing and tuning are audible exactly as written.",
    }
    (project / "mix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("✅ TIMING-SAFE FOUNDATION MIX V4:", out)
    print("Style:", style)
    print("✅ No delay-based stereo widening")
    print("✅ No automatic reference remaster")
    print("Spectrum:", report["spectrum"])


if __name__ == "__main__":
    main()
