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


VOCAL_PROFILES = {
    "uk_garage": {"presence": 2.6, "air": 1.0, "comp": 3.0, "reverb": 0.07, "delay": 0.045, "double_db": -36.0, "duck": 0.23, "target_db": -18.5},
    "house": {"presence": 2.4, "air": 1.2, "comp": 3.2, "reverb": 0.09, "delay": 0.055, "double_db": -35.0, "duck": 0.22, "target_db": -18.5},
    "techno": {"presence": 2.1, "air": 1.1, "comp": 3.4, "reverb": 0.11, "delay": 0.06, "double_db": -36.0, "duck": 0.21, "target_db": -19.0},
    "dnb": {"presence": 2.8, "air": 1.3, "comp": 3.5, "reverb": 0.06, "delay": 0.04, "double_db": -37.0, "duck": 0.26, "target_db": -18.2},
    "trap": {"presence": 2.5, "air": 1.0, "comp": 3.5, "reverb": 0.08, "delay": 0.05, "double_db": -35.0, "duck": 0.24, "target_db": -18.5},
    "hiphop": {"presence": 2.8, "air": 0.8, "comp": 3.2, "reverb": 0.05, "delay": 0.025, "double_db": -38.0, "duck": 0.25, "target_db": -18.0},
    "ambient": {"presence": 1.8, "air": 1.4, "comp": 2.3, "reverb": 0.18, "delay": 0.10, "double_db": -34.0, "duck": 0.16, "target_db": -20.0},
    "electronic": {"presence": 2.5, "air": 1.0, "comp": 3.1, "reverb": 0.08, "delay": 0.045, "double_db": -36.0, "duck": 0.23, "target_db": -18.7},
}


def ensure_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1: return np.vstack([x, x])
    if x.shape[0] == 2 and x.shape[1] > 2: return x
    if x.ndim == 2 and x.shape[1] >= 2: return x[:, :2].T.astype(np.float32)
    mono = x.reshape(-1); return np.vstack([mono, mono]).astype(np.float32)


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
    current = rms_db(x); gain_db = max(-24.0, min(max_up_db, target_db - current))
    return (x * (10 ** (gain_db / 20.0))).astype(np.float32)


def simple_deesser(stereo: np.ndarray, amount: float = 0.45) -> np.ndarray:
    amount = max(0.0, min(1.0, float(amount)))
    if amount <= 0: return stereo
    high = filt(stereo, 5600.0, "high"); body = stereo - high
    controlled = block_compressor(high, threshold_db=-25.0, ratio=3.0 + 2.0 * amount, makeup_db=0.0)
    return (body + high * (1.0 - 0.16 * amount) + controlled * (0.16 * amount)).astype(np.float32)


def delay_mono(x: np.ndarray, seconds: float) -> np.ndarray:
    samples = max(1, int(seconds * SR)); out = np.zeros_like(x)
    if samples < x.size: out[samples:] = x[:-samples]
    return out


def vocal_activity(mono: np.ndarray) -> np.ndarray:
    block = 512; n_blocks = max(1, math.ceil(len(mono) / block)); levels = np.zeros(n_blocks, dtype=np.float32)
    for i in range(n_blocks):
        part = mono[i * block:(i + 1) * block]
        if part.size: levels[i] = math.sqrt(float(np.mean(part * part)) + 1e-12)
    levels /= float(levels.max()) + 1e-8
    levels = np.convolve(levels, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    return np.interp(np.arange(len(mono)), np.linspace(0, len(mono) - 1, len(levels)), levels).astype(np.float32)


def process_vocals(raw: np.ndarray, sr: int, bpm: float, style: str, clarity: bool = True) -> dict[str, np.ndarray]:
    profile = dict(VOCAL_PROFILES.get(style, VOCAL_PROFILES["electronic"]))
    if clarity:
        # Keep the lead dry, centered and forward. FX are deliberately quieter than old Musicm8 mixes.
        profile["reverb"] *= 0.72; profile["delay"] *= 0.65; profile["double_db"] -= 2.0; profile["presence"] += 0.5; profile["duck"] += 0.03

    mono = np.nan_to_num(resample_mono(raw, sr))
    mono = filt(mono, 88.0, "high"); mono = filt(mono, 15000.0, "low")
    lead = np.vstack([mono, mono])
    lead = three_band_eq(lead, low_db=-2.8, mid_db=float(profile["presence"]), high_db=float(profile["air"]))
    lead = simple_deesser(lead, 0.42)
    lead = block_compressor(lead, threshold_db=-21.0, ratio=float(profile["comp"]), makeup_db=1.2)
    lead = saturate(lead, 0.022)
    lead = set_rms(lead, float(profile["target_db"]), 9.0)

    left_mono = filt(delay_mono(mono, 0.019), 10800.0, "low")
    right_mono = filt(delay_mono(mono, 0.028), 9800.0, "low")
    left = set_rms(np.vstack([left_mono, np.zeros_like(left_mono)]), float(profile["double_db"]), 5.0)
    right = set_rms(np.vstack([np.zeros_like(right_mono), right_mono]), float(profile["double_db"]), 5.0)

    dry = lead + left + right
    room = reverb(dry, float(profile["reverb"]))
    beat_delay = max(1, int((60.0 / max(1.0, bpm)) * 0.75 * SR))
    echo = np.zeros_like(dry)
    if beat_delay < dry.shape[-1]: echo[:, beat_delay:] = dry[:, :-beat_delay] * float(profile["delay"])
    vocal_mix = (0.94 * dry + 0.06 * room + echo).astype(np.float32)
    vocal_mix = block_compressor(vocal_mix, threshold_db=-14.5, ratio=1.55, makeup_db=0.0)
    return {"lead": lead, "double_L": left, "double_R": right, "vocal_mix": vocal_mix, "_profile": profile}


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[-1] >= n: return x[:, :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[-1]))).astype(np.float32)


def score_lock_if_available(project: Path, vocal: Path) -> Path:
    score = project / "vocal_score.json"
    if not score.exists() or vocal.name == "neural_lead_synced.wav": return vocal
    synced = project / "vocals" / "neural_lead_synced.wav"
    print("🎯 Locking neural vocal to Musicm8 vocal score before mixing...")
    subprocess.run([sys.executable, "-u", str(Path(__file__).resolve().parent / "vocal_sync.py"), "--vocal", str(vocal), "--score", str(score), "--out", str(synced)], check=True)
    return synced


def main() -> None:
    p = argparse.ArgumentParser(description="Genre-aware, clarity-first vocal processing and mix.")
    p.add_argument("--project", type=Path, required=True); p.add_argument("--vocal", type=Path, required=True); p.add_argument("--plan", type=Path, required=True); p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-clarity", action="store_true", help="Use the style profile without the extra lyric-clarity bias.")
    args = p.parse_args()

    project = args.project; plan = json.loads(args.plan.read_text(encoding="utf-8")); style = str(plan.get("style", "electronic"))
    backing_path = project / "master.wav"
    if not backing_path.exists(): raise FileNotFoundError(backing_path)
    if not args.vocal.exists(): raise FileNotFoundError(args.vocal)
    vocal_path = score_lock_if_available(project, args.vocal)

    instrumental = project / "master_instrumental.wav"; shutil.copy2(backing_path, instrumental)
    backing_raw, backing_sr = read_audio(backing_path)
    if backing_sr != SR:
        backing = np.vstack([librosa.resample(backing_raw[:, ch], orig_sr=backing_sr, target_sr=SR, res_type="kaiser_fast") for ch in (0, 1)]).astype(np.float32)
    else: backing = ensure_stereo(backing_raw)

    raw_vocal, vocal_sr = read_audio(vocal_path)
    result = process_vocals(raw_vocal, vocal_sr, float(plan.get("bpm", 120.0)), style, clarity=not args.no_clarity)
    profile = result.pop("_profile")
    n = backing.shape[-1]; parts = {k: fit_length(v, n) for k, v in result.items()}
    vocals_dir = project / "vocals"; vocals_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in parts.items(): sf.write(vocals_dir / f"{name}.wav", audio.T, SR, subtype="PCM_24")

    activity = vocal_activity(parts["lead"].mean(axis=0)); duck = 1.0 - float(profile["duck"]) * np.clip(activity, 0.0, 1.0)
    combined = backing * duck[None] + parts["vocal_mix"]
    combined = block_compressor(combined, threshold_db=-8.5, ratio=1.35, makeup_db=0.0); combined = saturate(combined, 0.010)
    peak = float(np.max(np.abs(combined))) + 1e-9; target = 10 ** (-1.0 / 20.0)
    if peak > target: combined *= target / peak

    out = args.out or (project / "master.wav"); sf.write(out, combined.T, SR, subtype="PCM_24")
    report = {"format": "musicm8-vocal-mix-v3", "style": style, "clarity_mode": not args.no_clarity, "profile": profile, "raw_neural_vocal": str(args.vocal), "score_locked_vocal": str(vocal_path), "instrumental_master": str(instrumental), "lead": str(vocals_dir / "lead.wav"), "vocal_mix": str(vocals_dir / "vocal_mix.wav"), "master": str(out), "lead_rms_db": round(rms_db(parts["lead"]), 3), "notes": "Genre-aware vocal chain with clarity-first presence, quieter doubles/FX and stronger activity-based backing ducking."}
    (project / "vocal_mix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("✅ Vocal style profile:", style)
    print("✅ Lyric clarity mode:", not args.no_clarity)
    print("✅ Vocal lead:", report["lead"]); print("✅ Vocal mix:", report["vocal_mix"]); print("✅ Final song master:", out)


if __name__ == "__main__":
    main()
