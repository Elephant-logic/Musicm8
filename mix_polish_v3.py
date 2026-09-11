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


def delay(x: np.ndarray, samples: int) -> np.ndarray:
    y = np.zeros_like(x)
    if samples < x.shape[-1]:
        y[:, samples:] = x[:, :-samples]
    return y


def stereoize(stereo: np.ndarray, width: float, mono_below: float) -> np.ndarray:
    width = max(0.0, min(1.0, float(width)))
    mid = stereo.mean(axis=0)
    low = filt(mid, mono_below, "low")
    high = mid - low
    if width <= 1e-6:
        return np.vstack([mid, mid]).astype(np.float32)
    d1 = max(1, int((0.005 + width * 0.008) * SR))
    d2 = max(1, int((0.011 + width * 0.012) * SR))
    side = (delay(np.vstack([high, high]), d1)[0] - delay(np.vstack([high, high]), d2)[0]) * (0.22 * width)
    return np.vstack([low + high + side, low + high - side]).astype(np.float32)


def parallel_compress(x: np.ndarray, threshold: float, ratio: float, blend: float) -> np.ndarray:
    wet = block_compressor(x, threshold_db=threshold, ratio=ratio, makeup_db=0.0)
    blend = max(0.0, min(1.0, blend))
    return (x * (1.0 - blend) + wet * blend).astype(np.float32)


def fit(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[-1] >= n:
        return x[:, :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[-1]))).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(math.sqrt(float(np.mean(np.square(x, dtype=np.float64))) + 1e-12) + 1e-12)


def mix_profile(plan: dict) -> dict:
    p = dict(plan.get("production", {}).get("mix", {}))
    defaults = {
        "drums_db": -17.0, "bass_db": -19.5, "chords_db": -23.5, "melody_db": -24.0,
        "bus_ratio": 1.45, "bus_drive": 0.02, "low_tilt_db": -0.7, "mid_tilt_db": 0.5, "high_tilt_db": 0.7,
    }
    for k, v in defaults.items(): p.setdefault(k, v)
    return p


def process(role: str, x: np.ndarray, profile: dict, style: str) -> np.ndarray:
    if role == "drums":
        y = filt(x, 25.0, "high")
        y = three_band_eq(y, low_db=-0.5, mid_db=0.5, high_db=1.2)
        y = saturate(y, 0.018 if style in {"house", "uk_garage", "hiphop"} else 0.030)
        y = parallel_compress(y, -18.0, 4.2, 0.30 if style != "ambient" else 0.10)
        y = set_rms(y, float(profile["drums_db"]), max_gain_db=6.0)
        return stereoize(y, 0.20 if style != "ambient" else 0.45, 520.0)
    if role == "bass":
        y = filt(x, 28.0, "high")
        y = filt(y, 7800.0 if style not in {"dnb", "techno"} else 11000.0, "low")
        y = three_band_eq(y, low_db=-0.8, mid_db=1.1 if style in {"uk_garage", "dnb", "techno"} else 0.5, high_db=-0.4)
        y = saturate(y, 0.020 if style == "ambient" else 0.040 if style in {"dnb", "techno"} else 0.028)
        y = block_compressor(y, threshold_db=-20.0, ratio=2.4, makeup_db=0.0)
        y = set_rms(y, float(profile["bass_db"]), max_gain_db=5.0)
        return stereoize(y, 0.025, 260.0)
    if role == "chords":
        hp = 95.0 if style == "ambient" else 125.0
        y = filt(x, hp, "high")
        y = three_band_eq(y, low_db=-2.3, mid_db=1.0, high_db=0.9)
        y = block_compressor(y, threshold_db=-23.0, ratio=1.7, makeup_db=0.0)
        y = set_rms(y, float(profile["chords_db"]), max_gain_db=7.0)
        return stereoize(y, 0.76 if style in {"ambient", "uk_garage"} else 0.62, 180.0)
    if role == "melody":
        y = filt(x, 145.0, "high")
        y = three_band_eq(y, low_db=-3.0, mid_db=1.5, high_db=1.1)
        y = block_compressor(y, threshold_db=-22.0, ratio=1.8, makeup_db=0.0)
        y = set_rms(y, float(profile["melody_db"]), max_gain_db=7.0)
        return stereoize(y, 0.50 if style != "ambient" else 0.78, 220.0)
    return x


def spectral_report(x: np.ndarray) -> dict:
    mono = x.mean(axis=0)
    n = min(len(mono), SR * 45)
    if n < 4096:
        return {}
    y = mono[:n] * np.hanning(n)
    p = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    total = float(p.sum()) + 1e-12
    def band(a: float, b: float) -> float:
        m = (f >= a) & (f < b)
        return float(p[m].sum() / total)
    return {"sub": round(band(20,80),4), "bass": round(band(80,250),4), "low_mid": round(band(250,1000),4), "mid": round(band(1000,4000),4), "high": round(band(4000,12000),4)}


def stereo_report(x: np.ndarray) -> dict:
    l, r = x[0], x[1]
    corr = float(np.corrcoef(l, r)[0,1]) if np.std(l)>1e-8 and np.std(r)>1e-8 else 1.0
    mid=(l+r)*0.5; side=(l-r)*0.5
    sm = math.sqrt(float(np.mean(side*side))+1e-12)/(math.sqrt(float(np.mean(mid*mid))+1e-12)+1e-12)
    return {"lr_correlation": round(corr,4), "side_to_mid_rms": round(sm,4)}


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 genre-aware mix/master v3.")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    project = args.project
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    style = str(plan.get("style", "electronic"))
    profile = mix_profile(plan)
    stems_dir = project / "audio_stems"
    if not stems_dir.exists(): raise FileNotFoundError(stems_dir)

    before = project / "master.wav"
    if before.exists(): shutil.copy2(before, project / "master_prepolish.wav")
    polished_dir = project / "audio_stems_polished"; polished_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[str,np.ndarray] = {}
    max_n=0
    for role in ("drums","bass","chords","melody"):
        path=stems_dir/f"{role}.wav"
        if not path.exists(): continue
        audio,sr=sf.read(path,always_2d=True,dtype="float32")
        if int(sr)!=SR: raise ValueError(f"Expected {SR} Hz for {path}, got {sr}")
        y=process(role,ensure_stereo(audio),profile,style)
        rendered[role]=y; max_n=max(max_n,y.shape[-1])
        sf.write(polished_dir/f"{role}.wav",y.T,SR,subtype="PCM_24")
    if not rendered: raise RuntimeError("No stems found")

    mix=np.zeros((2,max_n),dtype=np.float32)
    for role,y in rendered.items(): mix += fit(y,max_n)

    # Genre-aware bus: leave transient room, tilt overall tone, then light glue.
    mix=filt(mix,22.0,"high")
    mix=three_band_eq(mix,float(profile["low_tilt_db"]),float(profile["mid_tilt_db"]),float(profile["high_tilt_db"]))
    mix=block_compressor(mix,threshold_db=-9.5,ratio=float(profile["bus_ratio"]),makeup_db=0.0)
    mix=saturate(mix,float(profile["bus_drive"]))

    # Do not loudness-maximize a bad balance. Only catch peaks and retain dynamics.
    target=10**(-1.0/20.0)
    peak=float(np.max(np.abs(mix)))+1e-9
    if peak>target: mix*=target/peak

    out=args.out or project/"master.wav"
    sf.write(out,mix.T,SR,subtype="PCM_24")
    report={
        "format":"musicm8-mix-polish-v3","style":style,"profile":profile,"master":str(out),
        "prepolish":str(project/"master_prepolish.wav") if (project/"master_prepolish.wav").exists() else None,
        "polished_stems":str(polished_dir),"master_rms_db":round(rms_db(mix),3),"spectrum":spectral_report(mix),"stereo":stereo_report(mix),
        "notes":"Style-specific stem targets, EQ, saturation, parallel drum compression, bass control, stereo hierarchy and gentle bus glue."
    }
    (project/"mix_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("✅ Genre-aware mix v3:",out)
    print("Style:",style)
    print("Spectrum:",report["spectrum"])
    print("Stereo:",report["stereo"])


if __name__ == "__main__":
    main()
