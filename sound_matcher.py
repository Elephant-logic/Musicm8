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

from musicm8_synth import (
    SR,
    echo,
    patch_from_fingerprint,
    render_instrument,
    reverb,
    sonic_for_reference,
)

MATCH_SR = 22050
CACHE_VERSION = "musicm8-inverse-synth-v1"
ROLE_SOURCE = {"drums": "drums", "bass": "bass", "chords": "other", "melody": "vocals"}

# Search space for Musicm8's current native synth/FX engine.
# The matcher learns a reusable reference patch; idea-specific producer controls are
# applied later by render_matched.py so the reference timbre and creative request
# remain separate.
PARAMS: dict[str, dict[str, tuple[float, float, str]]] = {
    "drums": {
        "brightness": (0.05, 1.0, "linear"),
        "drive": (0.0, 0.85, "linear"),
        "room": (0.0, 0.75, "linear"),
    },
    "bass": {
        "sub": (0.0, 1.0, "linear"),
        "cutoff": (120.0, 7000.0, "log"),
        "drive": (0.0, 0.95, "linear"),
        "attack": (0.001, 0.12, "log"),
        "decay": (0.03, 0.8, "log"),
        "sustain": (0.08, 1.0, "linear"),
        "release": (0.015, 0.8, "log"),
        "width": (0.0, 0.35, "linear"),
    },
    "chords": {
        "detune": (0.0, 0.5, "linear"),
        "cutoff": (350.0, 15000.0, "log"),
        "drive": (0.0, 0.7, "linear"),
        "attack": (0.001, 1.0, "log"),
        "decay": (0.04, 1.5, "log"),
        "sustain": (0.08, 1.0, "linear"),
        "release": (0.03, 2.0, "log"),
        "reverb": (0.0, 0.85, "linear"),
        "width": (0.0, 1.0, "linear"),
    },
    "melody": {
        "detune": (0.0, 0.4, "linear"),
        "cutoff": (500.0, 16000.0, "log"),
        "drive": (0.0, 0.75, "linear"),
        "attack": (0.001, 0.35, "log"),
        "decay": (0.03, 1.0, "log"),
        "sustain": (0.08, 1.0, "linear"),
        "release": (0.02, 1.2, "log"),
        "delay": (0.0, 0.65, "linear"),
        "reverb": (0.0, 0.75, "linear"),
        "width": (0.0, 0.8, "linear"),
    },
}
WAVES = ("sine", "triangle", "saw", "square")


def _ref(library: dict[str, Any], ref_id: str | None) -> dict[str, Any] | None:
    if not ref_id:
        return None
    return next((r for r in library.get("references", []) if r.get("id") == ref_id), None)


def _all_notes(pm: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    return [n for inst in pm.instruments for n in inst.notes]


def choose_active_window(
    target: np.ndarray,
    sr: int,
    midi: pretty_midi.PrettyMIDI,
    seconds: float,
) -> tuple[float, np.ndarray]:
    win = max(1, int(seconds * sr))
    if target.size <= win:
        padded = np.pad(target, (0, max(0, win - target.size)))
        return 0.0, padded[:win].astype(np.float32)

    notes = _all_notes(midi)
    hop = max(1, int(0.25 * sr))
    sq = np.square(target.astype(np.float64))
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    best_score = -1.0
    best_start = 0

    for start in range(0, target.size - win + 1, hop):
        t0, t1 = start / sr, (start + win) / sr
        note_count = sum(1 for n in notes if n.start < t1 and n.end > t0)
        if note_count == 0:
            continue
        energy = float((csum[start + win] - csum[start]) / win)
        score = math.log1p(1e5 * energy) * (1.0 + min(note_count, 16) * 0.025)
        if score > best_score:
            best_score = score
            best_start = start

    # Fallback to the loudest window if transcription has no notes.
    if best_score < 0:
        starts = range(0, target.size - win + 1, hop)
        best_start = max(
            starts,
            key=lambda s: float((csum[s + win] - csum[s]) / win),
            default=0,
        )

    return best_start / sr, target[best_start : best_start + win].astype(np.float32)


def slice_midi_instrument(
    midi_path: Path,
    start_s: float,
    seconds: float,
    role: str,
) -> pretty_midi.Instrument:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    src = pm.instruments[0] if pm.instruments else pretty_midi.Instrument(0, is_drum=role == "drums")
    inst = pretty_midi.Instrument(
        program=src.program,
        is_drum=(role == "drums") or src.is_drum,
        name=role,
    )
    end_s = start_s + seconds
    for note in src.notes:
        if note.start >= end_s or note.end <= start_s:
            continue
        s = max(note.start, start_s) - start_s
        e = min(note.end, end_s) - start_s
        if e <= s:
            continue
        inst.notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=float(s),
                end=float(e),
            )
        )
    return inst


def role_fx(audio: np.ndarray, role: str, patch: dict[str, Any], bpm: float) -> np.ndarray:
    if role == "chords":
        return reverb(audio, float(patch.get("reverb", 0.0)))
    if role == "melody":
        return reverb(
            echo(audio, float(patch.get("delay", 0.0)), bpm),
            float(patch.get("reverb", 0.0)),
        )
    if role == "drums":
        return reverb(audio, float(patch.get("room", 0.0)) * 0.45)
    return audio


def _resample_curve(x: np.ndarray, n: int = 64) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros(n, dtype=np.float32)
    if x.size == 1:
        return np.full(n, float(x[0]), dtype=np.float32)
    old = np.linspace(0.0, 1.0, x.size)
    new = np.linspace(0.0, 1.0, n)
    return np.interp(new, old, x).astype(np.float32)


def descriptor(y: np.ndarray, sr: int = MATCH_SR) -> dict[str, np.ndarray | float]:
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1))
    if y.size < 2048:
        y = np.pad(y, (0, 2048 - y.size))
    peak = float(np.max(np.abs(y))) + 1e-8
    y_shape = y / peak

    mel = librosa.feature.melspectrogram(
        y=y_shape,
        sr=sr,
        n_fft=2048,
        hop_length=256,
        n_mels=64,
        fmin=25,
        fmax=min(10000, sr // 2),
        power=2.0,
    )
    mel_db = librosa.power_to_db(np.maximum(mel, 1e-12), ref=np.max, top_db=80.0)
    spectral = (np.mean(mel_db, axis=1) + 80.0) / 80.0

    rms = librosa.feature.rms(y=y_shape, frame_length=1024, hop_length=256).reshape(-1)
    rms = rms / (float(np.max(rms)) + 1e-8)
    onset = librosa.onset.onset_strength(y=y_shape, sr=sr, hop_length=256)
    onset = onset / (float(np.max(onset)) + 1e-8)

    rms_abs = librosa.feature.rms(y=y, frame_length=1024, hop_length=256).reshape(-1)
    loudness_db = 20.0 * math.log10(float(np.median(rms_abs)) + 1e-8)
    return {
        "spectral": spectral.astype(np.float32),
        "rms": _resample_curve(rms),
        "onset": _resample_curve(onset),
        "loudness_db": float(loudness_db),
    }


def distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    spectral = float(np.mean(np.abs(a["spectral"] - b["spectral"])))
    rms = float(np.mean(np.abs(a["rms"] - b["rms"])))
    onset = float(np.mean(np.abs(a["onset"] - b["onset"])))
    # Small loudness term: timbre dominates; reference stem gain should not dictate the patch.
    loud = min(1.0, abs(float(a["loudness_db"]) - float(b["loudness_db"])) / 30.0)
    return 0.67 * spectral + 0.16 * rms + 0.12 * onset + 0.05 * loud


def render_candidate(
    inst: pretty_midi.Instrument,
    role: str,
    patch: dict[str, Any],
    seconds: float,
    bpm: float,
) -> np.ndarray:
    n = max(1, int((seconds + 0.5) * SR))
    audio = render_instrument(inst, role, patch, n)
    audio = role_fx(audio, role, patch, bpm)
    mono = audio.mean(axis=0)[: int(seconds * SR)]
    # Exact 44.1k -> 22.05k decimation keeps scoring cheap. The synth itself still renders at 44.1k.
    return mono[::2].astype(np.float32)


def evaluate(
    inst: pretty_midi.Instrument,
    role: str,
    patch: dict[str, Any],
    seconds: float,
    bpm: float,
    target_desc: dict[str, Any],
) -> float:
    y = render_candidate(inst, role, patch, seconds, bpm)
    return distance(descriptor(y), target_desc)


def mutate_value(value: float, spec: tuple[float, float, str], strength: float, rng: random.Random) -> float:
    lo, hi, mode = spec
    if mode == "log":
        lo_l, hi_l = math.log(lo), math.log(hi)
        cur = math.log(max(lo, min(hi, float(value))))
        cur += rng.gauss(0.0, 0.22 * strength * (hi_l - lo_l))
        return float(math.exp(max(lo_l, min(hi_l, cur))))
    cur = float(value)
    cur += rng.gauss(0.0, 0.18 * strength * (hi - lo))
    return float(max(lo, min(hi, cur)))


def optimize_patch(
    role: str,
    initial: dict[str, Any],
    inst: pretty_midi.Instrument,
    target: np.ndarray,
    seconds: float,
    bpm: float,
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], float, float]:
    target_desc = descriptor(target)
    rng = random.Random(seed)
    best = dict(initial)

    # Check oscillator families before numeric refinement.
    if role != "drums":
        wave_best = None
        wave_score = float("inf")
        for wave in WAVES:
            candidate = dict(best)
            candidate["wave"] = wave
            score = evaluate(inst, role, candidate, seconds, bpm, target_desc)
            if score < wave_score:
                wave_best, wave_score = candidate, score
        best = wave_best or best
        best_score = wave_score
    else:
        best_score = evaluate(inst, role, best, seconds, bpm, target_desc)

    initial_score = best_score
    specs = PARAMS[role]
    keys = list(specs)

    for i in range(max(0, iterations)):
        progress = i / max(1, iterations - 1)
        strength = 1.0 - 0.78 * progress
        candidate = dict(best)

        # Mostly local moves, with periodic broader restarts around the fingerprint patch.
        if i and i % 12 == 0:
            candidate = dict(initial)
            strength = 1.15

        mutate_n = rng.randint(1, min(3, len(keys)))
        for key in rng.sample(keys, mutate_n):
            base_value = float(candidate.get(key, initial.get(key, sum(specs[key][:2]) / 2)))
            candidate[key] = mutate_value(base_value, specs[key], strength, rng)

        if role != "drums" and rng.random() < 0.12 * strength:
            candidate["wave"] = rng.choice(WAVES)

        score = evaluate(inst, role, candidate, seconds, bpm, target_desc)
        if score < best_score:
            best, best_score = candidate, score

    return best, float(initial_score), float(best_score)


def cache_signature(stem: Path, midi: Path, role: str) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "role": role,
        "stem": str(stem),
        "stem_size": stem.stat().st_size,
        "stem_mtime_ns": stem.stat().st_mtime_ns,
        "midi": str(midi),
        "midi_size": midi.stat().st_size,
        "midi_mtime_ns": midi.stat().st_mtime_ns,
    }


def match_role(
    role: str,
    ref: dict[str, Any],
    library: dict[str, Any],
    seconds: float,
    iterations: int,
    seed: int,
    cache_dir: Path | None,
    force: bool,
    preview_dir: Path,
) -> dict[str, Any]:
    source_role = ROLE_SOURCE[role]
    stem = Path(ref.get("stems", {}).get(source_role, ""))
    midi = Path(ref.get("midi", {}).get(role, ""))
    if not stem.exists() or not midi.exists():
        raise FileNotFoundError(f"{role}: missing reference stem or MIDI ({stem}, {midi})")

    signature = cache_signature(stem, midi, role)
    cache_path = cache_dir / f"{ref['id']}_{role}.json" if cache_dir else None
    if cache_path and cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                print(f"♻️ {role}: reusing matched patch for {ref['id']}")
                return cached["result"]
        except Exception:
            pass

    target_full, _ = librosa.load(stem, sr=MATCH_SR, mono=True)
    ref_midi = pretty_midi.PrettyMIDI(str(midi))
    start_s, target = choose_active_window(target_full, MATCH_SR, ref_midi, seconds)
    inst = slice_midi_instrument(midi, start_s, seconds, role)
    if not inst.notes:
        raise RuntimeError(f"{role}: no usable MIDI notes in the selected reference window")

    features = sonic_for_reference(library, ref["id"], role)
    initial = patch_from_fingerprint(role, features, {})
    print(
        f"🎛️ {role}: matching {ref['id']} from {start_s:.2f}s "
        f"({len(inst.notes)} MIDI notes, {iterations} search steps)"
    )
    best, initial_score, best_score = optimize_patch(
        role, initial, inst, target, seconds, float(ref.get("bpm", 120.0)), iterations, seed
    )

    matched = render_candidate(inst, role, best, seconds, float(ref.get("bpm", 120.0)))
    preview_dir.mkdir(parents=True, exist_ok=True)
    sf.write(preview_dir / f"{role}_reference.wav", target, MATCH_SR, subtype="PCM_24")
    sf.write(preview_dir / f"{role}_matched.wav", matched, MATCH_SR, subtype="PCM_24")

    improvement = 100.0 * max(0.0, initial_score - best_score) / max(initial_score, 1e-8)
    # This is an internal search score, not a human-perception percentage.
    similarity = 100.0 * max(0.0, 1.0 - min(1.0, best_score))
    result = {
        "reference_id": ref["id"],
        "reference_name": ref.get("name"),
        "source_role": source_role,
        "source_stem": str(stem),
        "source_midi": str(midi),
        "window_start_seconds": round(start_s, 4),
        "window_seconds": float(seconds),
        "midi_notes": len(inst.notes),
        "initial_distance": round(initial_score, 6),
        "best_distance": round(best_score, 6),
        "search_improvement_percent": round(improvement, 2),
        "internal_similarity_score": round(similarity, 2),
        "patch": best,
        "reference_preview": str(preview_dir / f"{role}_reference.wav"),
        "matched_preview": str(preview_dir / f"{role}_matched.wav"),
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"signature": signature, "result": result}, indent=2),
            encoding="utf-8",
        )
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="Inverse-synthesis matcher: render Musicm8 patches, compare against reference stems, and iteratively improve them."
    )
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="matched_patches.json")
    p.add_argument("--iterations", type=int, default=40)
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
            print(f"⚠️ {role}: no selected reference; synth will use fingerprint defaults")
            continue
        try:
            roles[role] = match_role(
                role,
                ref,
                library,
                seconds,
                max(0, int(args.iterations)),
                args.seed + i * 1009,
                args.cache_dir,
                args.force,
                preview_dir,
            )
            r = roles[role]
            print(
                f"✅ {role}: distance {r['initial_distance']:.4f} -> {r['best_distance']:.4f} "
                f"(search improvement {r['search_improvement_percent']:.1f}%)"
            )
        except Exception as exc:
            print(f"⚠️ {role}: inverse matching skipped ({exc})")

    payload = {
        "format": "musicm8-inverse-synth-v1",
        "metric_note": (
            "Distances combine log-mel spectral shape, amplitude envelope, onset envelope and a small loudness term. "
            "internal_similarity_score is a search diagnostic, not a human perceptual accuracy claim."
        ),
        "roles": roles,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✅ Matched patches: {args.out}")
    print(f"✅ A/B previews: {preview_dir}")


if __name__ == "__main__":
    main()
