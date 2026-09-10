from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from musicm8_synth_v2 import SR, clamp, render_instrument, role_fx

MATCH_SR = 22050
CACHE_VERSION = "musicm8-inverse-synth-v2"
ROLE_SOURCE = {"drums": "drums", "bass": "bass", "chords": "other", "melody": "vocals"}
WAVES = ("sine", "triangle", "saw", "square", "wavetable", "fm")


def _ref(library: dict[str, Any], ref_id: str | None) -> dict[str, Any] | None:
    if not ref_id:
        return None
    return next((r for r in library.get("references", []) if r.get("id") == ref_id), None)


def _all_notes(pm: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    return [n for inst in pm.instruments for n in inst.notes]


def choose_active_window(target: np.ndarray, sr: int, midi: pretty_midi.PrettyMIDI, seconds: float) -> tuple[float, np.ndarray]:
    win = max(1, int(seconds * sr))
    if target.size <= win:
        return 0.0, np.pad(target, (0, max(0, win - target.size)))[:win].astype(np.float32)
    notes = _all_notes(midi)
    hop = max(1, int(0.20 * sr))
    sq = np.square(target.astype(np.float64))
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    best_score, best_start = -1.0, 0
    for start in range(0, target.size - win + 1, hop):
        t0, t1 = start / sr, (start + win) / sr
        note_count = sum(1 for n in notes if n.start < t1 and n.end > t0)
        if note_count == 0:
            continue
        energy = float((csum[start + win] - csum[start]) / win)
        score = math.log1p(1e5 * energy) * (1.0 + min(note_count, 24) * 0.025)
        if score > best_score:
            best_score, best_start = score, start
    if best_score < 0:
        starts = range(0, target.size - win + 1, hop)
        best_start = max(starts, key=lambda s: float((csum[s + win] - csum[s]) / win), default=0)
    return best_start / sr, target[best_start:best_start + win].astype(np.float32)


def slice_midi_instrument(midi_path: Path, start_s: float, seconds: float, role: str) -> pretty_midi.Instrument:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    src = pm.instruments[0] if pm.instruments else pretty_midi.Instrument(0, is_drum=role == "drums")
    inst = pretty_midi.Instrument(src.program, is_drum=(role == "drums") or src.is_drum, name=role)
    end_s = start_s + seconds
    for note in src.notes:
        if note.start >= end_s or note.end <= start_s:
            continue
        s = max(note.start, start_s) - start_s
        e = min(note.end, end_s) - start_s
        if e > s:
            inst.notes.append(pretty_midi.Note(velocity=int(note.velocity), pitch=int(note.pitch), start=float(s), end=float(e)))
    return inst


def _resample_curve(x: np.ndarray, n: int = 64) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros(n, dtype=np.float32)
    if x.size == 1:
        return np.full(n, float(x[0]), dtype=np.float32)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, x.size), x).astype(np.float32)


def _log_spectral_profile(y: np.ndarray, sr: int, n_fft: int, bins: int = 96) -> np.ndarray:
    hop = max(64, n_fft // 4)
    mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) + 1e-8
    profile = np.mean(librosa.amplitude_to_db(mag, ref=np.max, top_db=90.0), axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    keep = freqs >= 25.0
    freqs = freqs[keep]
    profile = profile[keep]
    if freqs.size < 2:
        return np.zeros(bins, dtype=np.float32)
    dst = np.geomspace(max(25.0, freqs[0]), min(sr * 0.49, freqs[-1]), bins)
    p = np.interp(dst, freqs, profile)
    return ((p + 90.0) / 90.0).astype(np.float32)


def descriptor(y: np.ndarray, sr: int = MATCH_SR) -> dict[str, Any]:
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1))
    if y.size < 8192:
        y = np.pad(y, (0, 8192 - y.size))
    peak = float(np.max(np.abs(y))) + 1e-8
    shape = y / peak
    profiles = {str(n): _log_spectral_profile(shape, sr, n) for n in (512, 2048, 8192)}
    mel = librosa.feature.melspectrogram(y=shape, sr=sr, n_fft=2048, hop_length=256, n_mels=64, fmin=25, fmax=min(10000, sr // 2), power=2.0)
    mel_db = librosa.power_to_db(np.maximum(mel, 1e-12), ref=np.max, top_db=90.0)
    mel_profile = ((np.mean(mel_db, axis=1) + 90.0) / 90.0).astype(np.float32)
    rms = librosa.feature.rms(y=shape, frame_length=1024, hop_length=256).reshape(-1)
    rms /= float(np.max(rms)) + 1e-8
    onset = librosa.onset.onset_strength(y=shape, sr=sr, hop_length=256)
    onset /= float(np.max(onset)) + 1e-8
    centroid = librosa.feature.spectral_centroid(y=shape, sr=sr, n_fft=2048, hop_length=256).reshape(-1)
    centroid /= max(1.0, sr * 0.5)
    abs_rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256).reshape(-1)
    loudness_db = 20.0 * math.log10(float(np.median(abs_rms)) + 1e-8)
    return {"profiles": profiles, "mel": mel_profile, "rms": _resample_curve(rms), "onset": _resample_curve(onset), "centroid": _resample_curve(centroid), "loudness_db": float(loudness_db)}


def distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    multi = np.mean([np.mean(np.abs(a["profiles"][k] - b["profiles"][k])) for k in ("512", "2048", "8192")])
    mel = float(np.mean(np.abs(a["mel"] - b["mel"])))
    rms = float(np.mean(np.abs(a["rms"] - b["rms"])))
    onset = float(np.mean(np.abs(a["onset"] - b["onset"])))
    centroid = float(np.mean(np.abs(a["centroid"] - b["centroid"])))
    loud = min(1.0, abs(float(a["loudness_db"]) - float(b["loudness_db"])) / 30.0)
    return float(0.34 * multi + 0.30 * mel + 0.12 * rms + 0.12 * onset + 0.07 * centroid + 0.05 * loud)


def sonic_for_reference(library: dict[str, Any], ref_id: str, role: str) -> dict[str, Any]:
    source_role = ROLE_SOURCE[role]
    for ref in library.get("references", []):
        if ref.get("id") == ref_id:
            return ref.get("sonic", {}).get(source_role, {})
    return {}


def learn_harmonics(target: np.ndarray, inst: pretty_midi.Instrument, sr: int = MATCH_SR, count: int = 32) -> list[float]:
    notes = [n.pitch for n in inst.notes if 24 <= n.pitch <= 100]
    if not notes:
        return [1.0] + [0.0] * (count - 1)
    f0 = 440.0 * 2 ** ((float(np.median(notes)) - 69.0) / 12.0)
    n_fft = 8192
    mag = np.abs(librosa.stft(target, n_fft=n_fft, hop_length=1024))
    profile = np.median(mag, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    amps = []
    for h in range(1, count + 1):
        f = f0 * h
        if f >= sr * 0.47:
            amps.append(0.0)
            continue
        bw = max(12.0, f * 0.025)
        mask = (freqs >= f - bw) & (freqs <= f + bw)
        amps.append(float(np.max(profile[mask])) if np.any(mask) else 0.0)
    base = max(amps[0], max(amps) * 0.08, 1e-8)
    norm = np.asarray(amps, dtype=np.float64) / base
    norm = np.clip(norm, 0.0, 1.5)
    norm *= 1.0 / np.sqrt(np.arange(1, count + 1))
    if norm[0] < 0.2:
        norm[0] = 1.0
    return [round(float(x), 5) for x in norm]


def _decay_estimate(y: np.ndarray, sr: int) -> float:
    if y.size < 256:
        return 0.12
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128).reshape(-1)
    if rms.size == 0 or float(rms.max()) <= 1e-8:
        return 0.12
    rms /= float(rms.max()) + 1e-8
    peak = int(np.argmax(rms))
    after = np.where(rms[peak:] < 0.2)[0]
    frames = int(after[0]) if after.size else min(len(rms) - peak - 1, 20)
    return clamp(frames * 128 / sr, 0.03, 0.8, 0.12)


def analyze_drum_classes(target: np.ndarray, inst: pretty_midi.Instrument, sr: int) -> dict[str, dict[str, float]]:
    groups = {"kick": [], "snare": [], "hat": []}
    for note in inst.notes:
        kind = "kick" if note.pitch in (35, 36) else "snare" if note.pitch in (38, 40) else "hat"
        start = max(0, int(note.start * sr))
        length = int((0.55 if kind != "hat" else 0.25) * sr)
        seg = target[start:min(target.size, start + length)]
        if seg.size > sr * 0.03:
            groups[kind].append(seg.astype(np.float32))
    out: dict[str, dict[str, float]] = {}
    for kind, segs in groups.items():
        if not segs:
            continue
        min_len = min(len(s) for s in segs[:12])
        stack = np.stack([s[:min_len] for s in segs[:12]])
        avg = np.median(stack, axis=0).astype(np.float32)
        mag = np.abs(librosa.stft(avg, n_fft=1024, hop_length=128)) + 1e-8
        centroid = float(librosa.feature.spectral_centroid(S=mag, sr=sr).mean())
        flat = float(librosa.feature.spectral_flatness(S=mag).mean())
        decay = _decay_estimate(avg, sr)
        if kind == "kick":
            out[kind] = {"pitch_start": clamp(115 + centroid * 0.025, 80, 240, 150), "pitch_end": clamp(35 + centroid * 0.004, 32, 75, 45), "sweep": clamp(0.018 + decay * 0.10, 0.008, 0.10, 0.028), "decay": decay, "click": clamp(flat * 3.0, 0.02, 1.0, 0.25)}
        elif kind == "snare":
            out[kind] = {"tone_hz": clamp(150 + centroid * 0.02, 120, 350, 190), "decay": decay, "noise": clamp(0.55 + flat * 1.8, 0.35, 0.98, 0.72), "brightness": clamp(centroid / 6500, 0.1, 1.0, 0.55)}
        else:
            out[kind] = {"decay": clamp(decay, 0.02, 0.22, 0.05), "open_decay": clamp(decay * 2.5, 0.12, 0.8, 0.28), "highpass": clamp(3200 + centroid * 0.55, 2800, 11000, 5200), "metallic": clamp(0.25 + flat * 1.2, 0.1, 0.95, 0.5)}
    return out


def default_patch(role: str, features: dict[str, Any], target: np.ndarray, inst: pretty_midi.Instrument) -> dict[str, Any]:
    centroid = clamp(features.get("spectral_centroid_hz", 1800), 100, 9000, 1800)
    rolloff = clamp(features.get("rolloff_hz", 6000), 300, 15000, 6000)
    flat = clamp(features.get("flatness", 0.04), 0, 0.5, 0.04)
    sub_ratio = clamp(features.get("sub_ratio", 0.15), 0, 1, 0.15)
    air = clamp(features.get("air_ratio", 0.08), 0, 1, 0.08)
    dynamic = clamp(features.get("dynamic_range_db", 10), 0, 40, 10)
    common = {"drive": clamp(0.08 + flat * 2.0, 0, 0.7, 0.15), "eq_low_db": 0.0, "eq_mid_db": 0.0, "eq_high_db": 0.0, "comp_threshold_db": -18.0, "comp_ratio": 2.0, "comp_makeup_db": 0.0}
    if role == "drums":
        patch = {**common, "gain": 0.78, "width": 0.12, "room": clamp(dynamic / 45, 0.02, 0.45, 0.14), "kick": {"pitch_start": 150, "pitch_end": 45, "sweep": 0.028, "decay": 0.20, "click": 0.25}, "snare": {"tone_hz": 190, "decay": 0.14, "noise": 0.72, "brightness": 0.55}, "hat": {"decay": 0.05, "open_decay": 0.28, "highpass": 5200, "metallic": 0.5}}
        for k, v in analyze_drum_classes(target, inst, MATCH_SR).items():
            patch[k].update(v)
        return patch
    harmonics = learn_harmonics(target, inst, MATCH_SR)
    if role == "bass":
        return {**common, "wave": "wavetable", "harmonics": harmonics, "sub": clamp(0.30 + 1.3 * sub_ratio, 0.2, 0.95, 0.6), "cutoff": clamp(rolloff * 0.48, 220, 5200, 1500), "filter_env": 0.35, "fm_mix": 0.10, "fm_ratio": 2.0, "fm_index": 1.0, "noise_mix": 0.01, "detune": 0.03, "attack": 0.004, "decay": 0.16, "sustain": 0.72, "release": 0.10, "lfo_rate": 2.0, "lfo_depth": 0.0, "width": 0.03, "gain": 0.58}
    if role == "chords":
        return {**common, "wave": "wavetable", "harmonics": harmonics, "cutoff": clamp(rolloff * 0.80, 900, 13000, 6000), "filter_env": 0.22, "fm_mix": 0.05, "fm_ratio": 2.0, "fm_index": 0.8, "noise_mix": clamp(flat * 0.2, 0, 0.08, 0.01), "detune": clamp(0.08 + air * 0.9, 0.04, 0.40, 0.16), "attack": 0.03, "decay": 0.30, "sustain": 0.64, "release": 0.38, "lfo_rate": 0.4, "lfo_depth": 0.04, "reverb": clamp(0.16 + dynamic / 70, 0.12, 0.65, 0.3), "width": 0.62, "gain": 0.31}
    return {**common, "wave": "wavetable", "harmonics": harmonics, "cutoff": clamp(rolloff * 0.92, 1200, 15000, 7000), "filter_env": 0.30, "fm_mix": 0.08, "fm_ratio": 2.0, "fm_index": 0.8, "noise_mix": 0.01, "detune": clamp(0.04 + air * 0.5, 0.02, 0.24, 0.08), "attack": 0.008, "decay": 0.18, "sustain": 0.60, "release": 0.20, "lfo_rate": 4.5, "lfo_depth": 0.02, "delay": 0.16, "reverb": 0.24, "width": 0.35, "gain": 0.27}


PARAMS: dict[str, dict[str, tuple[float, float, str]]] = {
    "bass": {"sub": (0.0, 1.0, "linear"), "cutoff": (120, 8000, "log"), "filter_env": (0, 1, "linear"), "fm_mix": (0, 0.8, "linear"), "fm_ratio": (0.5, 6.0, "linear"), "fm_index": (0, 7, "linear"), "noise_mix": (0, 0.2, "linear"), "detune": (0, 0.3, "linear"), "attack": (0.001, 0.15, "log"), "decay": (0.03, 1.0, "log"), "sustain": (0.05, 1, "linear"), "release": (0.01, 1.0, "log"), "drive": (0, 0.95, "linear"), "eq_low_db": (-8, 8, "linear"), "eq_mid_db": (-8, 8, "linear"), "eq_high_db": (-8, 8, "linear"), "comp_threshold_db": (-30, -6, "linear"), "comp_ratio": (1, 8, "linear")},
    "chords": {"cutoff": (300, 17000, "log"), "filter_env": (0, 1, "linear"), "fm_mix": (0, 0.7, "linear"), "fm_ratio": (0.5, 6.0, "linear"), "fm_index": (0, 6, "linear"), "noise_mix": (0, 0.25, "linear"), "detune": (0, 0.55, "linear"), "attack": (0.001, 1.2, "log"), "decay": (0.04, 1.5, "log"), "sustain": (0.05, 1, "linear"), "release": (0.03, 2.5, "log"), "drive": (0, 0.75, "linear"), "reverb": (0, 0.85, "linear"), "width": (0, 1, "linear"), "lfo_rate": (0.05, 8, "log"), "lfo_depth": (0, 0.5, "linear"), "eq_low_db": (-8, 8, "linear"), "eq_mid_db": (-8, 8, "linear"), "eq_high_db": (-8, 8, "linear")},
    "melody": {"cutoff": (500, 18000, "log"), "filter_env": (0, 1, "linear"), "fm_mix": (0, 0.8, "linear"), "fm_ratio": (0.5, 8.0, "linear"), "fm_index": (0, 8, "linear"), "noise_mix": (0, 0.2, "linear"), "detune": (0, 0.45, "linear"), "attack": (0.001, 0.5, "log"), "decay": (0.03, 1.2, "log"), "sustain": (0.05, 1, "linear"), "release": (0.02, 1.5, "log"), "drive": (0, 0.8, "linear"), "delay": (0, 0.7, "linear"), "reverb": (0, 0.8, "linear"), "width": (0, 0.9, "linear"), "lfo_rate": (0.1, 12, "log"), "lfo_depth": (0, 0.6, "linear"), "eq_low_db": (-8, 8, "linear"), "eq_mid_db": (-8, 8, "linear"), "eq_high_db": (-8, 8, "linear")},
}
DRUM_PARAMS: dict[str, tuple[float, float, str]] = {"kick.pitch_start": (70, 260, "linear"), "kick.pitch_end": (28, 80, "linear"), "kick.sweep": (0.008, 0.12, "log"), "kick.decay": (0.05, 0.8, "log"), "kick.click": (0, 1, "linear"), "snare.tone_hz": (110, 360, "linear"), "snare.decay": (0.04, 0.7, "log"), "snare.noise": (0.2, 1, "linear"), "snare.brightness": (0, 1, "linear"), "hat.decay": (0.015, 0.22, "log"), "hat.open_decay": (0.08, 0.8, "log"), "hat.highpass": (2500, 12000, "log"), "hat.metallic": (0, 1, "linear"), "drive": (0, 0.8, "linear"), "room": (0, 0.6, "linear"), "eq_low_db": (-8, 8, "linear"), "eq_mid_db": (-8, 8, "linear"), "eq_high_db": (-8, 8, "linear")}


def _get(patch: dict[str, Any], path: str, default: float) -> float:
    cur: Any = patch
    parts = path.split(".")
    for k in parts[:-1]:
        cur = cur.get(k, {}) if isinstance(cur, dict) else {}
    return float(cur.get(parts[-1], default)) if isinstance(cur, dict) else default


def _set(patch: dict[str, Any], path: str, value: float) -> None:
    parts = path.split(".")
    cur = patch
    for k in parts[:-1]:
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = value


def mutate_value(value: float, spec: tuple[float, float, str], strength: float, rng: random.Random) -> float:
    lo, hi, mode = spec
    if mode == "log":
        lo_l, hi_l = math.log(lo), math.log(hi)
        cur = math.log(max(lo, min(hi, float(value))))
        cur += rng.gauss(0, 0.20 * strength * (hi_l - lo_l))
        return float(math.exp(max(lo_l, min(hi_l, cur))))
    return float(max(lo, min(hi, value + rng.gauss(0, 0.16 * strength * (hi - lo)))))


def render_candidate(inst: pretty_midi.Instrument, role: str, patch: dict[str, Any], seconds: float, bpm: float) -> np.ndarray:
    n = max(1, int((seconds + 0.7) * SR))
    audio = role_fx(render_instrument(inst, role, patch, n), role, patch, bpm)
    return audio.mean(axis=0)[:int(seconds * SR)][::2].astype(np.float32)


def evaluate(inst: pretty_midi.Instrument, role: str, patch: dict[str, Any], seconds: float, bpm: float, target_desc: dict[str, Any]) -> float:
    return distance(descriptor(render_candidate(inst, role, patch, seconds, bpm)), target_desc)


def optimize_patch(role: str, initial: dict[str, Any], inst: pretty_midi.Instrument, target: np.ndarray, seconds: float, bpm: float, iterations: int, seed: int) -> tuple[dict[str, Any], float, float]:
    target_desc = descriptor(target)
    rng = random.Random(seed)
    best = json.loads(json.dumps(initial))
    best_score = evaluate(inst, role, best, seconds, bpm, target_desc)
    if role != "drums":
        for wave in WAVES:
            candidate = json.loads(json.dumps(best))
            candidate["wave"] = wave
            if wave == "fm":
                candidate["fm_mix"] = 0.0
            score = evaluate(inst, role, candidate, seconds, bpm, target_desc)
            if score < best_score:
                best_score, best = score, candidate
    initial_score = best_score
    specs = DRUM_PARAMS if role == "drums" else PARAMS[role]
    keys = list(specs)
    for i in range(max(0, iterations)):
        progress = i / max(1, iterations - 1)
        strength = 1.05 - 0.82 * progress
        candidate = json.loads(json.dumps(best))
        if i and i % 14 == 0:
            candidate = json.loads(json.dumps(initial))
            strength = 1.2
        for key in rng.sample(keys, rng.randint(1, min(4, len(keys)))):
            lo, hi, _ = specs[key]
            _set(candidate, key, mutate_value(_get(candidate, key, (lo + hi) / 2), specs[key], strength, rng))
        if role != "drums" and rng.random() < 0.14 * strength:
            candidate["wave"] = rng.choice(WAVES)
            if candidate["wave"] == "fm":
                candidate["fm_mix"] = 0.0
        score = evaluate(inst, role, candidate, seconds, bpm, target_desc)
        if score < best_score:
            best, best_score = candidate, score
    return best, float(initial_score), float(best_score)


def cache_signature(stem: Path, midi: Path, role: str) -> dict[str, Any]:
    return {"version": CACHE_VERSION, "role": role, "stem": str(stem), "stem_size": stem.stat().st_size, "stem_mtime_ns": stem.stat().st_mtime_ns, "midi": str(midi), "midi_size": midi.stat().st_size, "midi_mtime_ns": midi.stat().st_mtime_ns}


def match_role(role: str, ref: dict[str, Any], library: dict[str, Any], seconds: float, iterations: int, seed: int, cache_dir: Path | None, force: bool, preview_dir: Path) -> dict[str, Any]:
    source_role = ROLE_SOURCE[role]
    stem = Path(ref.get("stems", {}).get(source_role, ""))
    midi = Path(ref.get("midi", {}).get(role, ""))
    if not stem.exists() or not midi.exists():
        raise FileNotFoundError(f"{role}: missing reference stem or MIDI ({stem}, {midi})")
    signature = cache_signature(stem, midi, role)
    cache_path = cache_dir / f"{ref['id']}_{role}_v2.json" if cache_dir else None
    if cache_path and cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                print(f"♻️ {role}: reusing v2 matched patch for {ref['id']}")
                return cached["result"]
        except Exception:
            pass

    target_full, _ = librosa.load(stem, sr=MATCH_SR, mono=True)
    ref_midi = pretty_midi.PrettyMIDI(str(midi))
    start_s, target = choose_active_window(target_full, MATCH_SR, ref_midi, seconds)
    inst = slice_midi_instrument(midi, start_s, seconds, role)
    if not inst.notes and role != "drums":
        try:
            f0, voiced, prob = librosa.pyin(target, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C7"), sr=MATCH_SR, frame_length=2048, hop_length=256)
            good = np.isfinite(f0) & np.asarray(voiced, dtype=bool) & (np.nan_to_num(prob, nan=0.0) > 0.45)
            if np.any(good):
                pitch = int(np.clip(round(float(librosa.hz_to_midi(np.median(f0[good])))), 0, 127))
                inst.notes.append(pretty_midi.Note(velocity=92, pitch=pitch, start=0.0, end=min(seconds, 1.5)))
        except Exception:
            pass
    if not inst.notes:
        raise RuntimeError(f"{role}: no usable notes/transients in the selected reference window")
    features = sonic_for_reference(library, ref["id"], role)
    initial = default_patch(role, features, target, inst)
    print(f"🎛️ {role}: v2 match {ref['id']} @ {start_s:.2f}s ({len(inst.notes)} notes, {iterations} steps)")
    best, initial_score, best_score = optimize_patch(role, initial, inst, target, seconds, float(ref.get("bpm", 120.0)), iterations, seed)
    matched = render_candidate(inst, role, best, seconds, float(ref.get("bpm", 120.0)))
    preview_dir.mkdir(parents=True, exist_ok=True)
    ref_preview = preview_dir / f"{role}_reference.wav"
    matched_preview = preview_dir / f"{role}_matched.wav"
    sf.write(ref_preview, target, MATCH_SR, subtype="PCM_24")
    sf.write(matched_preview, matched, MATCH_SR, subtype="PCM_24")
    improvement = 100 * max(0, initial_score - best_score) / max(initial_score, 1e-8)
    result = {"reference_id": ref["id"], "reference_name": ref.get("name"), "source_role": source_role, "source_stem": str(stem), "source_midi": str(midi), "window_start_seconds": round(start_s, 4), "window_seconds": float(seconds), "midi_notes": len(inst.notes), "initial_distance": round(initial_score, 6), "best_distance": round(best_score, 6), "search_improvement_percent": round(improvement, 2), "patch": best, "reference_preview": str(ref_preview), "matched_preview": str(matched_preview), "engine": CACHE_VERSION}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"signature": signature, "result": result}, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 inverse sound designer v2: learned harmonic wavetable/FM + per-drum models + multi-resolution scoring.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=48)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    library = json.loads(args.references.read_text(encoding="utf-8"))
    seconds = max(1.5, min(6.0, float(args.seconds)))
    preview_dir = args.out.parent / "sound_matches"
    roles: dict[str, Any] = {}
    for i, role in enumerate(("drums", "bass", "chords", "melody")):
        ref_id = plan.get("references", {}).get(role)
        ref = _ref(library, ref_id)
        if ref is None:
            print(f"⚠️ {role}: no selected reference")
            continue
        try:
            roles[role] = match_role(role, ref, library, seconds, max(0, args.iterations), args.seed + i * 1009, args.cache_dir, args.force, preview_dir)
            r = roles[role]
            print(f"✅ {role}: {r['initial_distance']:.4f} -> {r['best_distance']:.4f} ({r['search_improvement_percent']:.1f}% search improvement)")
        except Exception as exc:
            print(f"⚠️ {role}: v2 inverse matching skipped ({exc})")
    payload = {"format": CACHE_VERSION, "metric_note": "v2 distance combines multi-resolution log spectrum (512/2048/8192), mel spectrum, amplitude envelope, onset envelope, spectral-centroid motion and loudness.", "roles": roles}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✅ V2 matched patches: {args.out}")
    print(f"✅ A/B previews: {preview_dir}")


if __name__ == "__main__":
    main()
