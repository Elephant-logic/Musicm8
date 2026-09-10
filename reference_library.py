from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pretty_midi


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(np.asarray(x).reshape(-1)[0])
        return v if math.isfinite(v) else default
    except Exception:
        return default


def spectral_features(path: Path, sr: int = 22050) -> dict[str, float]:
    y, sr = librosa.load(path, sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size < 1024:
        return {
            "rms_db": -80.0, "spectral_centroid_hz": 0.0, "rolloff_hz": 0.0,
            "flatness": 0.0, "zcr": 0.0, "sub_ratio": 0.0, "bass_ratio": 0.0,
            "low_mid_ratio": 0.0, "high_mid_ratio": 0.0, "air_ratio": 0.0,
            "onset_rate": 0.0, "dynamic_range_db": 0.0,
        }

    n_fft = 2048
    hop = 512
    mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) + 1e-8
    power = mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_energy = power.mean(axis=1)
    total = float(band_energy.sum()) + 1e-12

    def band(lo: float, hi: float | None) -> float:
        mask = freqs >= lo
        if hi is not None:
            mask &= freqs < hi
        return float(band_energy[mask].sum() / total)

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).reshape(-1)
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=1.0)
    onset = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="time")
    duration = max(float(len(y) / sr), 1e-6)

    return {
        "rms_db": round(safe_float(np.median(rms_db), -80.0), 3),
        "spectral_centroid_hz": round(safe_float(librosa.feature.spectral_centroid(S=mag, sr=sr).mean()), 3),
        "rolloff_hz": round(safe_float(librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.85).mean()), 3),
        "flatness": round(safe_float(librosa.feature.spectral_flatness(S=mag).mean()), 5),
        "zcr": round(safe_float(librosa.feature.zero_crossing_rate(y).mean()), 5),
        "sub_ratio": round(band(20, 120), 5),
        "bass_ratio": round(band(120, 300), 5),
        "low_mid_ratio": round(band(300, 1200), 5),
        "high_mid_ratio": round(band(1200, 5000), 5),
        "air_ratio": round(band(5000, None), 5),
        "onset_rate": round(float(len(onset) / duration), 4),
        "dynamic_range_db": round(max(0.0, safe_float(np.percentile(rms_db, 95) - np.percentile(rms_db, 20))), 3),
    }


def midi_features(path: Path) -> dict[str, float | int]:
    if not path.exists():
        return {"notes": 0, "notes_per_second": 0.0, "median_pitch": 0.0, "pitch_range": 0}
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        return {"notes": 0, "notes_per_second": 0.0, "median_pitch": 0.0, "pitch_range": 0}
    duration = max(pm.get_end_time(), 1e-6)
    pitches = np.array([n.pitch for n in notes], dtype=np.float32)
    return {
        "notes": len(notes),
        "notes_per_second": round(float(len(notes) / duration), 4),
        "median_pitch": round(float(np.median(pitches)), 2),
        "pitch_range": int(pitches.max() - pitches.min()),
    }


def load_codec_token_map(codec_index: Path | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if codec_index is None or not codec_index.exists():
        return out
    root = codec_index.parent
    for line in codec_index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source = str(Path(row.get("source_audio", "")).resolve())
        token_rel = row.get("tokens")
        if source and token_rel:
            out.setdefault(source, []).append(str((root / token_rel).resolve()))
    return out


def build_library(index: Path, out: Path, codec_index: Path | None = None) -> dict[str, Any]:
    rows = [json.loads(x) for x in index.read_text(encoding="utf-8").splitlines() if x.strip()]
    token_map = load_codec_token_map(codec_index)
    references = []

    for i, row in enumerate(rows):
        source = Path(row["source"]).resolve()
        stems = {k: Path(v) for k, v in row.get("stems", {}).items()}
        midi = {k: Path(v) for k, v in row.get("midi", {}).items()}
        stem_profiles = {}
        for role, path in stems.items():
            if path.exists():
                stem_profiles[role] = spectral_features(path)
        midi_profiles = {}
        for role, path in midi.items():
            if role != "arrangement":
                midi_profiles[role] = midi_features(path)

        token_files = token_map.get(str(source), [])
        references.append({
            "id": f"ref_{i:03d}",
            "name": source.stem,
            "source": str(source),
            "bpm": float(row.get("bpm", 120.0)),
            "key": row.get("key"),
            "duration": float(row.get("duration", 0.0)),
            "stems": {k: str(v) for k, v in stems.items()},
            "midi": {k: str(v) for k, v in midi.items()},
            "sonic": stem_profiles,
            "musical": midi_profiles,
            "codec_tokens": token_files,
            "codec_token_count": len(token_files),
        })

    payload = {
        "format": "musicm8-reference-library-v1",
        "count": len(references),
        "references": references,
        "notes": (
            "MIDI/analysis describes musical structure. Stem spectral fingerprints describe timbre. "
            "codec_tokens point at the existing EnCodec acoustic representation for future neural resynthesis."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Build Musicm8 multimodal reference library.")
    p.add_argument("--index", type=Path, required=True, help="daw_dataset/index.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--codec-index", type=Path, default=None)
    args = p.parse_args()
    payload = build_library(args.index, args.out, args.codec_index)
    print(f"✅ Reference library: {args.out}")
    print(f"References: {payload['count']}")
    for ref in payload["references"]:
        print(f" - {ref['id']} {ref['name']} | {ref['bpm']:.1f} BPM | {ref.get('key') or 'key ?'} | acoustic token clips={ref['codec_token_count']}")


if __name__ == "__main__":
    main()
