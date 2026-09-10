from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from tqdm import tqdm

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


@dataclass
class NoteEvent:
    start: float
    end: float
    pitch: int
    velocity: int


@dataclass
class Analysis:
    source: str
    bpm: float
    duration: float
    key: str | None
    stems: dict[str, str]
    midi: dict[str, str]


def slugify(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return value or "track"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, check=True)


def load_mono(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, got_sr = librosa.load(path, sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32), int(got_sr)


def estimate_bpm(path: Path) -> float:
    y, sr = load_mono(path, 22050)
    if y.size < sr:
        return 120.0
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    try:
        bpm = float(np.asarray(tempo).reshape(-1)[0])
    except Exception:
        bpm = 120.0
    if not np.isfinite(bpm) or bpm < 50 or bpm > 220:
        bpm = 120.0
    return bpm


def estimate_key(path: Path) -> str | None:
    y, sr = load_mono(path, 22050)
    if y.size < sr:
        return None
    harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    profile = chroma.mean(axis=1)
    if not np.isfinite(profile).all() or float(profile.sum()) <= 1e-6:
        return None
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    scores: list[tuple[float, str]] = []
    p = profile - profile.mean()
    for i, name in enumerate(names):
        maj = np.roll(major, i); maj = maj - maj.mean()
        minp = np.roll(minor, i); minp = minp - minp.mean()
        scores.append((float(np.dot(p, maj)), f"{name} major"))
        scores.append((float(np.dot(p, minp)), f"{name} minor"))
    return max(scores, key=lambda x: x[0])[1]


def separate_with_demucs(src: Path, out_dir: Path, device: str) -> dict[str, Path]:
    stems_dir = out_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    expected = {name: stems_dir / f"{name}.wav" for name in ("drums", "bass", "vocals", "other")}
    if all(p.exists() and p.stat().st_size > 0 for p in expected.values()):
        return expected

    tmp = out_dir / "_demucs"
    shutil.rmtree(tmp, ignore_errors=True)
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs",
        "--device", device,
        "--out", str(tmp),
        str(src),
    ]
    try:
        run(cmd)
        model_dir = tmp / "htdemucs"
        candidates = [p for p in model_dir.iterdir() if p.is_dir()] if model_dir.exists() else []
        if not candidates:
            raise RuntimeError("Demucs did not create a track directory")
        produced = candidates[0]
        for name, dst in expected.items():
            source = produced / f"{name}.wav"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return expected


def _segments_from_f0(
    f0: np.ndarray,
    voiced: np.ndarray,
    times: np.ndarray,
    min_note: float,
    max_gap: float,
) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    active_start: int | None = None
    last_voiced: int | None = None

    def flush(end_index: int | None) -> None:
        nonlocal active_start, last_voiced
        if active_start is None or last_voiced is None:
            active_start = None; last_voiced = None
            return
        lo, hi = active_start, last_voiced + 1
        vals = f0[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            start = float(times[lo])
            end = float(times[min(hi, len(times) - 1)]) if hi < len(times) else float(times[-1])
            if end - start >= min_note:
                midi = int(np.clip(round(float(librosa.hz_to_midi(np.median(vals)))), 0, 127))
                events.append(NoteEvent(start, max(end, start + min_note), midi, 92))
        active_start = None; last_voiced = None

    for i, is_voiced in enumerate(voiced):
        if bool(is_voiced) and np.isfinite(f0[i]):
            if active_start is None:
                active_start = i
            last_voiced = i
        elif active_start is not None and last_voiced is not None:
            gap = float(times[i] - times[last_voiced])
            if gap > max_gap:
                flush(i)
    flush(len(times) - 1)
    return events


def transcribe_monophonic(path: Path, fmin: float, fmax: float) -> list[NoteEvent]:
    y, sr = load_mono(path, 22050)
    if y.size < sr // 4:
        return []
    hop = 256
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=hop,
        fill_na=np.nan,
    )
    if f0 is None:
        return []
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    voiced = np.asarray(voiced_flag, dtype=bool) & (np.nan_to_num(voiced_prob, nan=0.0) > 0.55)
    return _segments_from_f0(np.asarray(f0), voiced, np.asarray(times), min_note=0.08, max_gap=0.10)


def transcribe_drums(path: Path) -> list[NoteEvent]:
    y, sr = load_mono(path, 22050)
    if y.size < sr // 4:
        return []
    hop = 256
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=False, units="frames")
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    events: list[NoteEvent] = []
    for frame in onset_frames:
        fi = int(min(max(frame, 0), stft.shape[1] - 1))
        spec = stft[:, fi]
        total = float(spec.sum()) + 1e-9
        low = float(spec[freqs < 180].sum()) / total
        high = float(spec[freqs > 5000].sum()) / total
        if low > 0.32:
            pitch, vel = 36, 110
        elif high > 0.34:
            pitch, vel = 42, 82
        else:
            pitch, vel = 38, 100
        t = float(librosa.frames_to_time(frame, sr=sr, hop_length=hop))
        events.append(NoteEvent(t, t + 0.08, pitch, vel))
    return events


def infer_chords(path: Path, bpm: float, duration: float) -> list[NoteEvent]:
    y, sr = load_mono(path, 22050)
    if y.size < sr:
        return []
    harmonic = librosa.effects.harmonic(y)
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    beat = 60.0 / max(bpm, 1.0)
    window = beat * 2.0
    triads: list[tuple[str, tuple[int, int, int]]] = []
    for root in range(12):
        triads.append(("maj", (root, (root + 4) % 12, (root + 7) % 12)))
        triads.append(("min", (root, (root + 3) % 12, (root + 7) % 12)))
    events: list[NoteEvent] = []
    t = 0.0
    last: tuple[int, str] | None = None
    while t < duration:
        mask = (times >= t) & (times < min(duration, t + window))
        if not np.any(mask):
            t += window
            continue
        profile = chroma[:, mask].mean(axis=1)
        root_best, mode_best, score_best = 0, "maj", -1e9
        for root in range(12):
            for mode, third in (("maj", 4), ("min", 3)):
                idx = [root, (root + third) % 12, (root + 7) % 12]
                score = float(profile[idx].sum() - 0.15 * (profile.sum() - profile[idx].sum()))
                if score > score_best:
                    root_best, mode_best, score_best = root, mode, score
        chord = (root_best, mode_best)
        third = 4 if mode_best == "maj" else 3
        pitches = [48 + root_best, 48 + root_best + third, 48 + root_best + 7]
        # Avoid duplicate back-to-back chord events by extending is handled naturally by MIDI sustain.
        for pitch in pitches:
            events.append(NoteEvent(t, min(duration, t + window), int(pitch), 68))
        last = chord
        t += window
    return events


def quantize(events: list[NoteEvent], bpm: float, subdivision: int = 4) -> list[NoteEvent]:
    step = (60.0 / max(bpm, 1.0)) / subdivision
    out: list[NoteEvent] = []
    for e in events:
        start = round(e.start / step) * step
        end = round(e.end / step) * step
        if end <= start:
            end = start + step
        out.append(NoteEvent(max(0.0, start), max(step, end), e.pitch, int(np.clip(e.velocity, 1, 127))))
    return out


def write_midi(
    path: Path,
    bpm: float,
    tracks: dict[str, list[NoteEvent]],
) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    programs = {
        "bass": pretty_midi.instrument_name_to_program("Electric Bass (finger)"),
        "chords": pretty_midi.instrument_name_to_program("Electric Piano 1"),
        "melody": pretty_midi.instrument_name_to_program("Lead 1 (square)"),
    }
    order = ["drums", "bass", "chords", "melody"]
    for name in order:
        events = tracks.get(name, [])
        if not events:
            continue
        inst = pretty_midi.Instrument(program=0 if name == "drums" else programs[name], is_drum=name == "drums", name=name)
        for e in events:
            inst.notes.append(pretty_midi.Note(velocity=e.velocity, pitch=e.pitch, start=e.start, end=e.end))
        pm.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def process_track(src: Path, root: Path, device: str, force: bool = False) -> Analysis:
    track_dir = root / slugify(src.stem)
    analysis_path = track_dir / "analysis.json"
    if analysis_path.exists() and not force:
        return Analysis(**json.loads(analysis_path.read_text(encoding="utf-8")))

    track_dir.mkdir(parents=True, exist_ok=True)
    stems = separate_with_demucs(src, track_dir, device)
    duration = float(librosa.get_duration(path=str(src)))
    bpm = estimate_bpm(stems.get("drums", src))
    key = estimate_key(stems.get("other", src))

    drums = quantize(transcribe_drums(stems["drums"]), bpm)
    bass = quantize(transcribe_monophonic(stems["bass"], librosa.note_to_hz("C1"), librosa.note_to_hz("C5")), bpm)
    melody = quantize(transcribe_monophonic(stems["vocals"], librosa.note_to_hz("C2"), librosa.note_to_hz("C7")), bpm)
    chords = quantize(infer_chords(stems["other"], bpm, duration), bpm)

    midi_dir = track_dir / "midi"
    combined = midi_dir / "arrangement.mid"
    tracks = {"drums": drums, "bass": bass, "chords": chords, "melody": melody}
    write_midi(combined, bpm, tracks)
    for name, events in tracks.items():
        write_midi(midi_dir / f"{name}.mid", bpm, {name: events})

    result = Analysis(
        source=str(src),
        bpm=round(float(bpm), 3),
        duration=round(duration, 3),
        key=key,
        stems={k: str(v) for k, v in stems.items()},
        midi={"arrangement": str(combined), **{k: str(midi_dir / f"{k}.mid") for k in tracks}},
    )
    analysis_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 DAW preprocessor: Demucs stems -> structure -> MIDI.")
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    files = sorted(p for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise FileNotFoundError(f"No audio found in {args.audio_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.jsonl"
    rows = []
    for src in tqdm(files, desc="DAW tracks"):
        try:
            result = process_track(src, args.out, args.device, force=args.force)
            rows.append(asdict(result))
        except Exception as exc:
            print(f"⚠️ Failed {src.name}: {exc}")
    if not rows:
        raise RuntimeError("No tracks were prepared successfully")
    index_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"✅ Prepared {len(rows)} tracks")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
