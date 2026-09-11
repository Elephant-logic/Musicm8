from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from musicm8_synth_v2 import SR, block_compressor, saturate, three_band_eq


def load_audio(path: Path, seconds: float = 70.0) -> np.ndarray:
    y, sr = sf.read(path, always_2d=True, dtype="float32")
    x = y.T.astype(np.float32)
    if int(sr) != SR:
        x = np.vstack([librosa.resample(ch, orig_sr=int(sr), target_sr=SR, res_type="kaiser_fast") for ch in x]).astype(np.float32)
    if x.shape[0] == 1:
        x = np.vstack([x[0], x[0]])
    elif x.shape[0] > 2:
        x = x[:2]
    return np.nan_to_num(x[:, : int(seconds * SR)])


def band_profile(x: np.ndarray) -> dict[str, float]:
    mono = x.mean(axis=0)
    n = min(len(mono), SR * 60)
    if n < 4096:
        return {"low": 1/3, "mid": 1/3, "high": 1/3}
    # Average several FFT windows instead of trusting one long global FFT.
    n_fft = 8192
    hop = 4096
    power_sum = None
    count = 0
    for a in range(0, max(1, n - n_fft + 1), hop):
        part = mono[a:a+n_fft]
        if len(part) < n_fft:
            break
        spec = np.fft.rfft(part * np.hanning(n_fft))
        p = np.abs(spec) ** 2
        power_sum = p if power_sum is None else power_sum + p
        count += 1
    if power_sum is None or count == 0:
        return {"low": 1/3, "mid": 1/3, "high": 1/3}
    p = power_sum / count
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    total = float(p[(f >= 25) & (f < 18000)].sum()) + 1e-12
    def band(lo: float, hi: float) -> float:
        m = (f >= lo) & (f < hi)
        return float(p[m].sum() / total)
    return {"low": band(25, 250), "mid": band(250, 4200), "high": band(4200, 18000)}


def rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(math.sqrt(float(np.mean(np.square(x, dtype=np.float64))) + 1e-12) + 1e-12)


def target_from_refs(library: dict[str, Any], plan: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    by_id = {str(r.get("id")): r for r in library.get("references", [])}
    ids = []
    for rid in plan.get("references", {}).values():
        rid = str(rid)
        if rid and rid not in ids and rid in by_id:
            ids.append(rid)
    profiles = []
    used = []
    for rid in ids:
        path = Path(str(by_id[rid].get("source", "")))
        if not path.exists():
            continue
        try:
            profiles.append(band_profile(load_audio(path)))
            used.append(rid)
        except Exception:
            continue
    if not profiles:
        return {"low": 0.34, "mid": 0.48, "high": 0.18}, []
    return {k: float(np.median([p[k] for p in profiles])) for k in ("low", "mid", "high")}, used


def correction_db(current: float, target: float, strength: float = 0.42) -> float:
    # Energy ratio -> dB. Keep this deliberately conservative: references guide tonal balance,
    # they do not get cloned or force the master into an extreme curve.
    raw = 10.0 * math.log10(max(target, 1e-6) / max(current, 1e-6))
    return float(np.clip(raw * strength, -2.0, 2.0))


def reference_master(master: Path, plan_path: Path, library_path: Path, out: Path | None = None) -> dict[str, Any]:
    out = out or master
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    library = json.loads(library_path.read_text(encoding="utf-8"))
    x = load_audio(master, seconds=10_000)
    before = band_profile(x)
    target, refs = target_from_refs(library, plan)
    gains = {k: correction_db(before[k], target[k]) for k in ("low", "mid", "high")}

    y = three_band_eq(x, low_db=gains["low"], mid_db=gains["mid"], high_db=gains["high"])
    # Small final glue only. Do not crush the mix to chase the reference loudness.
    y = block_compressor(y, threshold_db=-8.0, ratio=1.22, makeup_db=0.0)
    y = saturate(y, 0.008)
    peak = float(np.max(np.abs(y))) + 1e-9
    ceiling = 10 ** (-1.0 / 20.0)
    if peak > ceiling:
        y *= ceiling / peak

    backup = master.parent / "master_pre_reference.wav"
    if master.resolve() == out.resolve() and master.exists():
        shutil.copy2(master, backup)
    sf.write(out, y.T, SR, subtype="PCM_24")
    after = band_profile(y)
    report = {
        "format": "musicm8-reference-master-v1",
        "style": plan.get("style"),
        "references": refs,
        "before": before,
        "target": target,
        "eq_correction_db": {k: round(v, 3) for k, v in gains.items()},
        "after": after,
        "rms_db": round(rms_db(y), 3),
        "output": str(out),
        "backup": str(backup) if backup.exists() else None,
        "notes": "Conservative reference-guided tonal balance: maximum 2 dB correction per broad band, followed by light glue and a -1 dB peak ceiling.",
    }
    (master.parent / "reference_master_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Gently steer a Musicm8 master toward the tonal balance of its selected references.")
    p.add_argument("--master", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    r = reference_master(args.master, args.plan, args.references, args.out)
    print("✅ REFERENCE-AWARE MASTER")
    print("References:", r["references"])
    print("EQ correction dB:", r["eq_correction_db"])
    print("Before:", r["before"])
    print("After :", r["after"])


if __name__ == "__main__":
    main()
