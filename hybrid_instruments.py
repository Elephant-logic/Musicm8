from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from musicm8_synth_v2 import SR, filt
from reference_instrument_bank import build_bank


STYLE_BLEND = {
    "uk_garage": {"drums": 0.58, "bass": 0.42, "chords": 0.25},
    "house": {"drums": 0.52, "bass": 0.36, "chords": 0.22},
    "techno": {"drums": 0.45, "bass": 0.30, "chords": 0.14},
    "dnb": {"drums": 0.60, "bass": 0.38, "chords": 0.18},
    "trap": {"drums": 0.62, "bass": 0.46, "chords": 0.18},
    "hiphop": {"drums": 0.66, "bass": 0.42, "chords": 0.28},
    "ambient": {"drums": 0.18, "bass": 0.18, "chords": 0.34},
    "electronic": {"drums": 0.46, "bass": 0.32, "chords": 0.20},
}


def ensure_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.vstack([x, x])
    if x.shape[0] == 2:
        return x
    if x.ndim == 2 and x.shape[1] >= 2:
        return x[:, :2].T.astype(np.float32)
    mono = x.reshape(-1)
    return np.vstack([mono, mono])


def rms(x: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def load_stem(path: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    if path in cache:
        return cache[path]
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    x = audio.T.astype(np.float32)
    if int(sr) != SR:
        x = np.vstack([librosa.resample(ch, orig_sr=int(sr), target_sr=SR, res_type="kaiser_fast") for ch in x]).astype(np.float32)
    if x.shape[0] == 1:
        x = np.vstack([x[0], x[0]])
    elif x.shape[0] > 2:
        x = x[:2]
    cache[path] = np.nan_to_num(x)
    return cache[path]


def slice_meta(meta: dict[str, Any], cache: dict[str, np.ndarray]) -> np.ndarray:
    x = load_stem(str(meta["stem"]), cache)
    a = max(0, int(round(float(meta.get("start", 0.0)) * SR)))
    b = min(x.shape[-1], int(round(float(meta.get("end", 0.0)) * SR)))
    if b <= a:
        return np.zeros((2, 1), dtype=np.float32)
    clip = x[:, a:b].copy()
    mono = clip.mean(axis=0)
    intervals = librosa.effects.split(mono, top_db=42, frame_length=1024, hop_length=128)
    if intervals.size:
        aa = max(0, int(intervals[0, 0]) - int(0.008 * SR))
        bb = min(clip.shape[-1], int(intervals[-1, 1]) + int(0.020 * SR))
        clip = clip[:, aa:bb]
    return clip.astype(np.float32)


def pitch_shift_stereo(x: np.ndarray, steps: float) -> np.ndarray:
    if abs(steps) < 0.08:
        return x
    steps = float(np.clip(steps, -12.0, 12.0))
    return np.vstack([librosa.effects.pitch_shift(ch, sr=SR, n_steps=steps) for ch in x]).astype(np.float32)


def fit_clip(x: np.ndarray, target_n: int, *, preserve_attack: bool = True) -> np.ndarray:
    target_n = max(1, int(target_n))
    if x.shape[-1] < 16:
        return np.zeros((2, target_n), dtype=np.float32)
    if preserve_attack and x.shape[-1] >= target_n:
        y = x[:, :target_n].copy()
    else:
        rate = float(np.clip(x.shape[-1] / target_n, 0.45, 2.2))
        if abs(rate - 1.0) > 0.025:
            channels = []
            for ch in x:
                try:
                    channels.append(librosa.effects.time_stretch(ch, rate=rate))
                except Exception:
                    channels.append(ch)
            y = np.vstack(channels).astype(np.float32)
        else:
            y = x.copy()
        if y.shape[-1] > target_n:
            y = y[:, :target_n]
        elif y.shape[-1] < target_n:
            y = np.pad(y, ((0, 0), (0, target_n - y.shape[-1])))
    fade = min(int(0.012 * SR), max(2, target_n // 8))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[:, :fade] *= ramp[None]
        y[:, -fade:] *= ramp[::-1][None]
    return y.astype(np.float32)


def place(dst: np.ndarray, clip: np.ndarray, start_s: float, gain: float) -> None:
    a = max(0, int(round(start_s * SR)))
    if a >= dst.shape[-1]:
        return
    n = min(clip.shape[-1], dst.shape[-1] - a)
    if n > 0:
        dst[:, a:a+n] += clip[:, :n] * float(gain)


def drum_kind(pitch: int) -> str:
    if pitch in {35, 36}: return "kick"
    if pitch in {37, 38, 39, 40}: return "snare"
    if pitch in {42, 44}: return "hat"
    if pitch == 46: return "open_hat"
    if pitch in {41, 43, 45, 47, 48, 50}: return "tom"
    return "perc"


def render_drums(inst: pretty_midi.Instrument, bank_ref: dict[str, Any], n: int, cache: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    out = np.zeros((2, n), dtype=np.float32)
    pools = bank_ref.get("drums", {})
    used = 0
    counters: dict[str, int] = {}
    for note in sorted(inst.notes, key=lambda x: x.start):
        kind = drum_kind(int(note.pitch))
        pool = pools.get(kind) or pools.get("perc") or []
        if not pool:
            continue
        idx = counters.get(kind, 0) % len(pool)
        counters[kind] = counters.get(kind, 0) + 1
        clip = slice_meta(pool[idx], cache)
        # Keep drum transients intact; only shorten overly long tails.
        max_tail = 0.70 if kind in {"kick", "snare", "open_hat", "tom"} else 0.30
        clip = fit_clip(clip, min(clip.shape[-1], int(max_tail * SR)), preserve_attack=True)
        vel = (float(note.velocity) / 127.0) ** 0.72
        place(out, clip, note.start, 0.90 * vel)
        used += 1
    return out, used


def nearest_bass(pool: list[dict[str, Any]], pitch: int) -> dict[str, Any] | None:
    if not pool:
        return None
    return min(pool, key=lambda x: abs(int(x.get("pitch", pitch)) - pitch))


def render_bass(inst: pretty_midi.Instrument, bank_ref: dict[str, Any], n: int, cache: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    out = np.zeros((2, n), dtype=np.float32)
    pool = bank_ref.get("bass", [])
    used = 0
    for note in sorted(inst.notes, key=lambda x: x.start):
        meta = nearest_bass(pool, int(note.pitch))
        if not meta:
            continue
        clip = slice_meta(meta, cache)
        clip = pitch_shift_stereo(clip, int(note.pitch) - int(meta.get("pitch", note.pitch)))
        target = int((max(0.08, note.end - note.start) + 0.12) * SR)
        clip = fit_clip(clip, target, preserve_attack=False)
        clip = filt(clip, 9500.0, "low")
        vel = (float(note.velocity) / 127.0) ** 0.75
        place(out, clip, note.start, 0.78 * vel)
        used += 1
    return out, used


def chord_groups(inst: pretty_midi.Instrument, tol: float = 0.035) -> list[list[pretty_midi.Note]]:
    groups: list[list[pretty_midi.Note]] = []
    for note in sorted(inst.notes, key=lambda x: (x.start, x.pitch)):
        if not groups or abs(note.start - groups[-1][0].start) > tol:
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def chord_cost(meta: dict[str, Any], target: list[int]) -> float:
    src = [int(x) for x in meta.get("pitches", [])]
    if not src or not target:
        return 999.0
    src_int = [x - src[0] for x in src]
    tar_int = [x - target[0] for x in target]
    m = min(len(src_int), len(tar_int))
    shape = sum(abs(src_int[i] - tar_int[i]) for i in range(m)) + 4.0 * abs(len(src_int) - len(tar_int))
    return shape + 0.15 * abs(src[0] - target[0])


def render_chords(inst: pretty_midi.Instrument, bank_ref: dict[str, Any], n: int, cache: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    out = np.zeros((2, n), dtype=np.float32)
    pool = bank_ref.get("chords", [])
    used = 0
    if not pool:
        return out, used
    for group in chord_groups(inst):
        target_pitches = sorted({int(x.pitch) for x in group})
        if not target_pitches:
            continue
        meta = min(pool, key=lambda x: chord_cost(x, target_pitches))
        clip = slice_meta(meta, cache)
        src_root = int(meta.get("root", target_pitches[0]))
        clip = pitch_shift_stereo(clip, float(np.clip(target_pitches[0] - src_root, -7, 7)))
        start = min(x.start for x in group)
        end = max(x.end for x in group)
        clip = fit_clip(clip, int((max(0.15, end - start) + 0.12) * SR), preserve_attack=False)
        # Remove sub/bass information from the 'other' stem so the layer behaves like texture/room/timbre.
        clip = filt(clip, 120.0, "high")
        vel = (sum(x.velocity for x in group) / len(group) / 127.0) ** 0.72
        place(out, clip, start, 0.62 * vel)
        used += 1
    return out, used


def match_layer_level(layer: np.ndarray, synth: np.ndarray, relative_db: float) -> np.ndarray:
    lr = rms(layer)
    sr = rms(synth)
    if lr < 1e-7 or sr < 1e-7:
        return layer
    target = sr * (10 ** (relative_db / 20.0))
    gain = float(np.clip(target / lr, 0.05, 8.0))
    return (layer * gain).astype(np.float32)


def hybridize(project: Path, plan_path: Path, library_path: Path, midi_path: Path, bank_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not bank_path.exists() or bank_path.stat().st_mtime < library_path.stat().st_mtime:
        build_bank(library_path, bank_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    by_name = {(i.name or "").lower(): i for i in pm.instruments}
    stems = project / "audio_stems"
    if not stems.exists():
        raise FileNotFoundError(stems)
    backup = project / "audio_stems_synth_only"
    backup.mkdir(parents=True, exist_ok=True)

    style = str(plan.get("style", "electronic"))
    blends = dict(STYLE_BLEND.get(style, STYLE_BLEND["electronic"]))
    user_hybrid = plan.get("production", {}).get("hybrid", {})
    for role in ("drums", "bass", "chords"):
        if role in user_hybrid:
            blends[role] = float(np.clip(user_hybrid[role], 0.0, 0.8))

    refs = plan.get("references", {})
    cache: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {"format": "musicm8-hybrid-instruments-v1", "style": style, "bank": str(bank_path), "roles": {}}

    for role in ("drums", "bass", "chords"):
        path = stems / f"{role}.wav"
        inst = by_name.get(role)
        rid = refs.get(role)
        ref = bank.get("references", {}).get(str(rid))
        if not path.exists() or inst is None or not ref:
            report["roles"][role] = {"status": "skipped", "reference_id": rid}
            continue
        shutil.copy2(path, backup / path.name)
        raw, sr = sf.read(path, always_2d=True, dtype="float32")
        synth = ensure_stereo(raw)
        if int(sr) != SR:
            synth = np.vstack([librosa.resample(ch, orig_sr=int(sr), target_sr=SR, res_type="kaiser_fast") for ch in synth]).astype(np.float32)
        n = synth.shape[-1]
        if role == "drums":
            layer, used = render_drums(inst, ref, n, cache)
            relative_db = -1.5
        elif role == "bass":
            layer, used = render_bass(inst, ref, n, cache)
            relative_db = -3.0
        else:
            layer, used = render_chords(inst, ref, n, cache)
            relative_db = -5.5
        amount = float(blends.get(role, 0.0))
        layer = match_layer_level(layer, synth, relative_db)
        # The synth remains the stable, editable pitch source. Reference audio supplies transient/body/room/timbre.
        hybrid = synth * (1.0 - 0.10 * amount) + layer * amount
        peak = float(np.max(np.abs(hybrid))) + 1e-9
        if peak > 1.25:
            hybrid *= 1.25 / peak
        sf.write(path, hybrid.T, SR, subtype="PCM_24")
        report["roles"][role] = {
            "status": "hybrid",
            "reference_id": rid,
            "blend": round(amount, 3),
            "reference_events_used": used,
            "synth_backup": str(backup / path.name),
            "output": str(path),
        }

    report["notes"] = (
        "Hybrid v1 keeps Musicm8's MIDI-controlled synth as the tuning/timing backbone and layers short aligned reference-stem "
        "transients/body underneath it. Drums use reference hits, bass uses pitch-shifted note windows, and chords use short "
        "transposed texture windows. Melody stays fully synthesized/wavetable-controlled so no reference vocal words leak into it."
    )
    (project / "hybrid_instruments_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Layer reference-derived instruments under Musicm8's editable synth stems.")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--midi", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    args = p.parse_args()
    report = hybridize(args.project, args.plan, args.references, args.midi, args.bank)
    print("✅ HYBRID REFERENCE INSTRUMENTS")
    for role, info in report["roles"].items():
        print(f" - {role}: {info.get('status')} blend={info.get('blend')} events={info.get('reference_events_used')}")
    print("Report:", args.project / "hybrid_instruments_report.json")


if __name__ == "__main__":
    main()
