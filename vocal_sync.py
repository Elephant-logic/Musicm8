from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from musicm8_synth_v2 import SR


def to_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1).astype(np.float32)
    if int(sr) != SR:
        mono = librosa.resample(mono, orig_sr=int(sr), target_sr=SR, res_type="kaiser_fast").astype(np.float32)
        sr = SR
    return np.nan_to_num(mono), int(sr)


def merge_intervals(intervals: np.ndarray, sr: int, max_gap_s: float = 0.32) -> list[tuple[int, int]]:
    if intervals.size == 0:
        return []
    out: list[tuple[int, int]] = []
    max_gap = int(max_gap_s * sr)
    for a, b in intervals.tolist():
        a, b = int(a), int(b)
        if b - a < int(0.10 * sr):
            continue
        if out and a - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def split_notes_for_lines(notes: list[dict], n_lines: int) -> list[list[dict]]:
    if not notes or n_lines <= 0:
        return []
    # Newer vocal scores may carry line_index directly. Prefer it when present.
    if any("line_index" in n for n in notes):
        groups: list[list[dict]] = [[] for _ in range(n_lines)]
        for n in notes:
            li = n.get("line_index")
            if li is None:
                continue
            li = max(0, min(n_lines - 1, int(li)))
            groups[li].append(n)
        if any(groups):
            return groups

    # Backward-compatible fallback for existing v1 scores: distribute the ordered
    # notes across lyric lines while preserving note order.
    groups = [[] for _ in range(n_lines)]
    for i, note in enumerate(notes):
        gi = min(n_lines - 1, int(i * n_lines / max(1, len(notes))))
        groups[gi].append(note)
    return groups


def phrase_targets(score: dict) -> list[dict]:
    phrases: list[dict] = []
    for sec_i, sec in enumerate(score.get("sections", [])):
        lines = [str(x).strip() for x in sec.get("lines", []) if str(x).strip()]
        notes = [n for n in sec.get("notes", []) if float(n.get("end", 0.0)) > float(n.get("start", 0.0))]
        if not lines or not notes:
            continue
        groups = split_notes_for_lines(notes, len(lines))
        for line_i, (line, group) in enumerate(zip(lines, groups)):
            if not group:
                continue
            start = max(float(sec.get("start", 0.0)), min(float(n["start"]) for n in group) - 0.04)
            end = min(float(sec.get("end", 1e9)), max(float(n["end"]) for n in group) + 0.10)
            pitches = [int(n.get("pitch", 60)) for n in group]
            phrases.append({
                "section_index": sec_i,
                "section": str(sec.get("name", "section")),
                "line_index": line_i,
                "text": line,
                "start": start,
                "end": max(start + 0.20, end),
                "target_midi": float(np.median(pitches)) if pitches else 60.0,
            })
    return phrases


def section_source_phrases(mono: np.ndarray, sr: int, sec_start: float, sec_end: float, wanted: int) -> list[tuple[int, int]]:
    a = max(0, int(round(sec_start * sr)))
    b = min(len(mono), int(round(sec_end * sr)))
    if b <= a or wanted <= 0:
        return []
    part = mono[a:b]
    intervals = librosa.effects.split(part, top_db=34, frame_length=2048, hop_length=256)
    merged = merge_intervals(intervals, sr)
    merged = [(a + x, a + y) for x, y in merged]
    if not merged:
        return [(a, b)] * wanted

    if len(merged) == wanted:
        return merged

    # If the neural singer produced more/fewer activity islands than lyric lines,
    # use its complete active span and partition it in order. This keeps words in
    # the right section while the target score decides exact phrase timing.
    active_a, active_b = merged[0][0], merged[-1][1]
    total = max(1, active_b - active_a)
    return [
        (
            active_a + int(round(total * i / wanted)),
            active_a + int(round(total * (i + 1) / wanted)),
        )
        for i in range(wanted)
    ]


def median_f0_midi(y: np.ndarray, sr: int) -> float | None:
    if len(y) < int(0.20 * sr):
        return None
    try:
        f0, _, _ = librosa.pyin(y, fmin=70.0, fmax=900.0, sr=sr, frame_length=2048, hop_length=256)
        voiced = f0[np.isfinite(f0)]
        if voiced.size < 3:
            return None
        hz = float(np.median(voiced))
        return 69.0 + 12.0 * math.log2(max(1e-6, hz) / 440.0)
    except Exception:
        return None


def nearest_octave_delta(target_midi: float, source_midi: float) -> float:
    raw = target_midi - source_midi
    candidates = [raw + 12.0 * k for k in range(-3, 4)]
    best = min(candidates, key=lambda x: abs(x))
    return float(np.clip(best, -4.0, 4.0))


def fit_phrase(y: np.ndarray, target_samples: int, target_midi: float, sr: int) -> tuple[np.ndarray, float | None, float]:
    y = np.nan_to_num(np.asarray(y, dtype=np.float32))
    if y.size < 16:
        return np.zeros(target_samples, dtype=np.float32), None, 0.0

    # Remove model padding/leading silence before alignment.
    active = librosa.effects.split(y, top_db=36, frame_length=1024, hop_length=128)
    if active.size:
        y = y[int(active[0, 0]):int(active[-1, 1])]
    if y.size < 16:
        return np.zeros(target_samples, dtype=np.float32), None, 0.0

    source_midi = median_f0_midi(y, sr)
    shift = nearest_octave_delta(target_midi, source_midi) if source_midi is not None else 0.0
    if abs(shift) >= 0.15:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift).astype(np.float32)

    rate = len(y) / max(1, target_samples)
    rate = float(np.clip(rate, 0.55, 1.80))
    if abs(rate - 1.0) > 0.015:
        try:
            y = librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
        except Exception:
            pass

    if len(y) > target_samples:
        y = y[:target_samples]
    elif len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))

    fade = min(int(0.025 * sr), max(0, target_samples // 4))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return y.astype(np.float32), source_midi, shift


def main() -> None:
    p = argparse.ArgumentParser(description="Lock a generated Musicm8 vocal to the written vocal score timing and key.")
    p.add_argument("--vocal", type=Path, required=True)
    p.add_argument("--score", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.vocal.exists():
        raise FileNotFoundError(args.vocal)
    if not args.score.exists():
        raise FileNotFoundError(args.score)

    score = json.loads(args.score.read_text(encoding="utf-8"))
    mono, sr = to_mono(args.vocal)
    targets = phrase_targets(score)
    if not targets:
        raise RuntimeError("Vocal score contains no lyric phrases with melody notes")

    total_s = max([float(s.get("end", 0.0)) for s in score.get("sections", [])] + [len(mono) / sr])
    synced = np.zeros(int(math.ceil(total_s * sr)), dtype=np.float32)
    report: list[dict] = []

    by_section: dict[int, list[dict]] = {}
    for t in targets:
        by_section.setdefault(int(t["section_index"]), []).append(t)

    sections = score.get("sections", [])
    for sec_i, ts in by_section.items():
        sec = sections[sec_i]
        sources = section_source_phrases(
            mono, sr, float(sec.get("start", 0.0)), float(sec.get("end", 0.0)), len(ts)
        )
        for t, (sa, sb) in zip(ts, sources):
            ta = max(0, int(round(float(t["start"]) * sr)))
            tb = min(len(synced), int(round(float(t["end"]) * sr)))
            if tb <= ta:
                continue
            fitted, source_midi, shift = fit_phrase(mono[sa:sb], tb - ta, float(t["target_midi"]), sr)
            synced[ta:tb] += fitted[: tb - ta]
            report.append({
                **t,
                "source_start": round(sa / sr, 4),
                "source_end": round(sb / sr, 4),
                "source_midi": None if source_midi is None else round(source_midi, 3),
                "pitch_shift_semitones": round(shift, 3),
            })

    peak = float(np.max(np.abs(synced))) + 1e-9
    if peak > 0.98:
        synced *= 0.98 / peak
    stereo = np.column_stack([synced, synced])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, stereo, sr, subtype="PCM_24")
    args.out.with_suffix(".json").write_text(json.dumps({
        "format": "musicm8-vocal-sync-v1",
        "source": str(args.vocal),
        "score": str(args.score),
        "output": str(args.out),
        "phrases": report,
        "note": "Generated vocal phrases were activity-trimmed, pitch-preserving time-fitted to the written score, lightly key-corrected at phrase level, and placed at the exact melody-note windows.",
    }, indent=2), encoding="utf-8")
    print(f"✅ Score-locked vocal: {args.out}")
    print(f"Aligned lyric phrases: {len(report)}")


if __name__ == "__main__":
    main()
