from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def load_mono(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, in_sr = sf.read(path, always_2d=True, dtype="float32")
    mono = np.mean(y, axis=1).astype(np.float32)
    if int(in_sr) != sr:
        mono = librosa.resample(mono, orig_sr=int(in_sr), target_sr=sr, res_type="kaiser_fast").astype(np.float32)
    return np.nan_to_num(mono), sr


def score_notes(score: dict[str, Any]) -> list[dict[str, float]]:
    notes: list[dict[str, float]] = []
    for sec in score.get("sections", []):
        for n in sec.get("notes", []):
            try:
                a = float(n["start"]); b = float(n["end"]); p = float(n["pitch"])
            except Exception:
                continue
            if b > a:
                notes.append({"start": a, "end": b, "pitch": p})
    return sorted(notes, key=lambda n: n["start"])


def target_at(t: float, notes: list[dict[str, float]]) -> float | None:
    active = [n for n in notes if n["start"] - 0.03 <= t <= n["end"] + 0.03]
    if not active:
        return None
    return min(active, key=lambda n: abs((n["start"] + n["end"]) * 0.5 - t))["pitch"]


def pitch_class_error_semitones(observed: float, expected: float) -> float:
    d = (observed - expected + 6.0) % 12.0 - 6.0
    return abs(float(d))


def analyse(vocal: Path, score_path: Path) -> dict[str, Any]:
    score = json.loads(score_path.read_text(encoding="utf-8"))
    notes = score_notes(score)
    y, sr = load_mono(vocal)
    hop = 256
    f0, voiced, prob = librosa.pyin(y, fmin=70.0, fmax=900.0, sr=sr, frame_length=2048, hop_length=hop)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    errors: list[float] = []
    compared = 0
    for hz, v, pr, t in zip(f0, voiced, np.nan_to_num(prob, nan=0.0), times):
        if not bool(v) or not np.isfinite(hz) or float(pr) < 0.45:
            continue
        target = target_at(float(t), notes)
        if target is None:
            continue
        observed = 69.0 + 12.0 * math.log2(float(hz) / 440.0)
        errors.append(pitch_class_error_semitones(observed, target))
        compared += 1

    if not errors:
        return {"pass": False, "reason": "No reliable voiced frames could be compared with the written melody", "compared_frames": 0}
    e = np.asarray(errors, dtype=np.float32)
    median_cents = float(np.median(e) * 100.0)
    p75_cents = float(np.percentile(e, 75) * 100.0)
    in_tune = float(np.mean(e <= 0.85))
    # This is deliberately stricter than 'technically voiced'. A bad singer is not
    # allowed to replace a clean instrumental master.
    passed = bool(median_cents <= 72.0 and p75_cents <= 135.0 and in_tune >= 0.58 and compared >= 8)
    return {
        "pass": passed,
        "compared_frames": compared,
        "median_pitch_error_cents": round(median_cents, 2),
        "p75_pitch_error_cents": round(p75_cents, 2),
        "frames_within_85_cents": round(in_tune, 4),
        "reason": "pitch follows written melody closely enough" if passed else "vocal pitch does not reliably follow the written melody",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Reject a Musicm8 neural vocal if it is audibly off-key against the written score.")
    p.add_argument("--vocal", type=Path, required=True)
    p.add_argument("--score", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    r = analyse(args.vocal, args.score)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"format": "musicm8-vocal-quality-v1", "vocal": str(args.vocal), "score": str(args.score), **r}, indent=2), encoding="utf-8")
    print("🎤 VOCAL PITCH QUALITY")
    print(json.dumps(r, indent=2))
    if not r["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
