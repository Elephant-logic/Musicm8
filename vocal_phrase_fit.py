from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def load_stereo(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    x = audio.T.astype(np.float32)
    if x.shape[0] == 1:
        x = np.vstack([x[0], x[0]])
    elif x.shape[0] > 2:
        x = x[:2]
    if int(sr) != target_sr:
        x = np.vstack([librosa.resample(ch, orig_sr=int(sr), target_sr=target_sr, res_type="kaiser_fast") for ch in x]).astype(np.float32)
    return np.nan_to_num(x)


def active_span(x: np.ndarray, sr: int) -> np.ndarray:
    mono = x.mean(axis=0)
    intervals = librosa.effects.split(mono, top_db=34, frame_length=1024, hop_length=128)
    if intervals.size == 0:
        return x
    # Do not use the model's padded 10-second placement. Keep only the actual
    # generated phrase, then place that phrase on Musicm8's intended time window.
    a = max(0, int(intervals[0, 0]) - int(0.025 * sr))
    b = min(x.shape[-1], int(intervals[-1, 1]) + int(0.045 * sr))
    return x[:, a:b]


def fit_duration(x: np.ndarray, target_n: int, sr: int) -> np.ndarray:
    target_n = max(1, int(target_n))
    x = active_span(x, sr)
    if x.shape[-1] < 32:
        return np.zeros((2, target_n), dtype=np.float32)
    rate = float(np.clip(x.shape[-1] / target_n, 0.45, 2.4))
    if abs(rate - 1.0) > 0.015:
        channels = []
        for ch in x:
            try:
                channels.append(librosa.effects.time_stretch(ch, rate=rate))
            except Exception:
                channels.append(ch)
        x = np.vstack(channels).astype(np.float32)
    if x.shape[-1] > target_n:
        x = x[:, :target_n]
    elif x.shape[-1] < target_n:
        x = np.pad(x, ((0, 0), (0, target_n - x.shape[-1])))
    fade = min(int(0.020 * sr), max(2, target_n // 6))
    if fade > 1:
        r = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        x[:, :fade] *= r[None]
        x[:, -fade:] *= r[::-1][None]
    return x.astype(np.float32)


def rebuild(vocal: Path, manifest: Path, out: Path) -> dict[str, Any]:
    base, sr = sf.read(vocal, always_2d=True, dtype="float32")
    total_n = base.shape[0]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not chunks:
        raise RuntimeError("ACE vocal manifest contains no phrase chunks")

    result = np.zeros((2, total_n), dtype=np.float32)
    repaired: list[dict[str, Any]] = []
    for row in chunks:
        src = Path(str(row.get("source", "")))
        if not src.exists():
            continue
        start = max(0.0, float(row.get("start", 0.0)))
        end = max(start + 0.08, float(row.get("end", start + 0.08)))
        a = max(0, int(round(start * sr)))
        b = min(total_n, int(round(end * sr)))
        if b <= a:
            continue
        x = load_stereo(src, int(sr))
        before_s = x.shape[-1] / float(sr)
        phrase = fit_duration(x, b - a, int(sr))
        result[:, a:b] += phrase[:, : b - a]
        repaired.append({
            "section": row.get("section"),
            "line_index": row.get("line_index"),
            "lyrics": row.get("lyrics"),
            "target_start": start,
            "target_end": end,
            "source": str(src),
            "source_duration": round(before_s, 4),
            "placed_duration": round((b - a) / float(sr), 4),
        })

    if not repaired:
        raise RuntimeError("No ACE phrase source files were available for timing rebuild")
    peak = float(np.max(np.abs(result))) + 1e-9
    if peak > 0.98:
        result *= 0.98 / peak
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, result.T, int(sr), subtype="PCM_24")
    report = {
        "format": "musicm8-vocal-phrase-fit-v1",
        "source_vocal": str(vocal),
        "manifest": str(manifest),
        "output": str(out),
        "phrases": repaired,
        "note": "ACE-Step's minimum padded generation duration is discarded. Each actual active sung phrase is trimmed, duration-fitted and placed at its exact Musicm8 song window before pitch sync.",
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Repair ACE-Step phrase timing using its original chunk outputs.")
    p.add_argument("--vocal", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    r = rebuild(args.vocal, args.manifest, args.out)
    print("✅ EXACT PHRASE TIMING REBUILD:", args.out)
    print("Rebuilt phrases:", len(r["phrases"]))


if __name__ == "__main__":
    main()
