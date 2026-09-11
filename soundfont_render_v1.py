from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import soundfile as sf

SR = 44100
ROLES = ("drums", "bass", "chords", "melody")

# Zero-based General MIDI programs. MIDI-LLM still composes every note/rhythm;
# this is the producer's instrument-choice stage so a random GM program does not
# turn a garage/house track into accordion/brass/harpsichord by accident.
STYLE_PROGRAMS: dict[str, dict[str, list[int]]] = {
    "uk_garage": {"bass": [38, 39], "chords": [4, 89], "melody": [81, 80]},
    "house": {"bass": [38, 33], "chords": [4, 89], "melody": [81, 80]},
    "techno": {"bass": [38, 39], "chords": [89, 90], "melody": [81, 80]},
    "dnb": {"bass": [38, 39], "chords": [89, 4], "melody": [81, 80]},
    "trap": {"bass": [38, 39], "chords": [4, 5], "melody": [80, 81]},
    "hiphop": {"bass": [33, 38], "chords": [4, 5], "melody": [80, 81]},
    "ambient": {"bass": [33, 38], "chords": [89, 90], "melody": [88, 89]},
    "electronic": {"bass": [38, 39], "chords": [4, 89], "melody": [81, 80]},
}

# Keep electronic kits to recognisable kick/snare/clap/hat/tom/cymbal voices.
# MIDI-LLM occasionally emits bongos, whistles or other GM percussion that can
# sound like a random late beat even when its timestamp is technically on-grid.
CORE_ELECTRONIC_DRUMS = {
    35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 55, 57,
}


def choose_soundfont(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env = os.environ.get("MUSICM8_SOUNDFONT", "").strip()
    if env:
        candidates.append(Path(env))
    candidates += [
        Path("/usr/share/sounds/sf3/MuseScore_General.sf3"),
        Path("/usr/share/sounds/sf2/MuseScore_General.sf2"),
        Path("/usr/share/mscore3/sounds/MuseScore_General.sf3"),
        Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
        Path("/usr/share/sounds/sf2/FluidR3_GS.sf2"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    for base in (Path("/usr/share/sounds"), Path("/usr/share/mscore3/sounds")):
        if base.exists():
            found = sorted(list(base.rglob("*.sf3")) + list(base.rglob("*.sf2")), key=lambda p: p.stat().st_size, reverse=True)
            if found:
                return found[0]
    raise FileNotFoundError(
        "No GM SoundFont was found. In Colab install: apt-get install -y fluidsynth musescore-general-soundfont fluid-soundfont-gm"
    )


def load_stereo(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    x = x.T.astype(np.float32)
    if x.shape[0] == 1:
        x = np.vstack([x[0], x[0]])
    elif x.shape[0] > 2:
        x = x[:2]
    return np.nan_to_num(x), int(sr)


def fit(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[-1] > n:
        return x[:, :n].copy()
    if x.shape[-1] < n:
        return np.pad(x, ((0, 0), (0, n - x.shape[-1]))).astype(np.float32)
    return x.astype(np.float32)


def render_midi(fluidsynth: str, soundfont: Path, midi: Path, wav: Path) -> None:
    cmd = [
        fluidsynth,
        "-ni",
        "-g", "0.62",
        "-r", str(SR),
        "-F", str(wav),
        str(soundfont),
        str(midi),
    ]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not wav.exists():
        tail = "\n".join((proc.stdout or "").splitlines()[-80:])
        raise RuntimeError(f"FluidSynth failed for {midi}\n{tail}")


def _grid_start(t: float, bpm: float, swing: float) -> tuple[float, int]:
    step = 60.0 / max(1.0, bpm) / 4.0
    idx = max(0, int(round(float(t) / step)))
    start = idx * step + (step * swing if idx % 2 else 0.0)
    return float(start), idx


def program_for(style: str, role: str, index: int, original: int) -> int:
    bank = STYLE_PROGRAMS.get(style, STYLE_PROGRAMS["electronic"]).get(role)
    if not bank:
        return int(original)
    return int(bank[index % len(bank)])


def tighten_role_midi(path: Path, bpm: float, swing: float, role: str, style: str) -> dict[str, Any]:
    """Put all role events on one exact clock and choose genre-coherent GM voices."""
    pm = pretty_midi.PrettyMIDI(str(path))
    out = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    step = 60.0 / max(1.0, bpm) / 4.0
    half = step * 0.5
    corrected = 0
    dropped = 0
    duplicates = 0
    max_shift_ms = 0.0
    programs: list[int] = []

    for inst_index, src in enumerate(pm.instruments):
        is_drum = bool(src.is_drum or role == "drums")
        chosen_program = int(src.program) if is_drum else program_for(style, role, inst_index, int(src.program))
        programs.append(chosen_program)
        dst = pretty_midi.Instrument(program=chosen_program, is_drum=is_drum, name=src.name)
        seen: set[tuple[int, int]] = set()
        for n in sorted(src.notes, key=lambda x: (x.start, x.pitch, x.end)):
            start, idx = _grid_start(float(n.start), bpm, swing)
            shift_ms = abs(start - float(n.start)) * 1000.0
            max_shift_ms = max(max_shift_ms, shift_ms)
            if shift_ms > 0.05:
                corrected += 1

            pitch = int(n.pitch)
            if is_drum:
                if not 35 <= pitch <= 81:
                    dropped += 1
                    continue
                if style != "ambient" and pitch not in CORE_ELECTRONIC_DRUMS:
                    dropped += 1
                    continue
                end = start + min(0.12, step * 0.85)
            else:
                dur = max(half, float(n.end) - float(n.start))
                dur = max(half, round(dur / half) * half)
                end = start + dur

            key = (pitch, idx)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            velocity = int(np.clip(n.velocity, 1, 127))
            if is_drum and pitch in {42, 44, 46, 51, 52, 55, 57}:
                velocity = min(100, velocity)
            dst.notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=max(start + 0.015, end),
            ))
        if dst.notes:
            out.instruments.append(dst)

    if not out.instruments:
        raise RuntimeError(f"Timing/sound guard removed every event from {path}")

    if role != "drums":
        for inst in out.instruments:
            by_pitch: dict[int, list[pretty_midi.Note]] = {}
            for n in inst.notes:
                by_pitch.setdefault(int(n.pitch), []).append(n)
            for notes in by_pitch.values():
                notes.sort(key=lambda x: x.start)
                for a, b in zip(notes, notes[1:]):
                    if a.end > b.start:
                        a.end = max(a.start + 0.015, b.start - 0.002)

    out.write(str(path))
    return {
        "events": sum(len(i.notes) for i in out.instruments),
        "corrected": corrected,
        "duplicates_removed": duplicates,
        "unmusical_drum_events_removed": dropped,
        "max_start_correction_ms": round(max_shift_ms, 4),
        "grid": "shared 1/16 + shared swing",
        "programs": programs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Render Musicm8's learned multitrack MIDI with style-directed sampled SoundFont instruments.")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--soundfont", type=Path, default=None)
    args = p.parse_args()

    project = args.project.resolve()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    arrangement = project / "arrangement.mid"
    stems_dir = project / "midi_stems"
    if not arrangement.exists():
        raise FileNotFoundError(arrangement)
    if not stems_dir.exists():
        raise FileNotFoundError(stems_dir)

    bpm = float(plan.get("bpm", 120.0))
    style = str(plan.get("style", "electronic"))
    swing = float(np.clip(float(plan.get("groove", {}).get("swing", 0.0)), 0.0, 0.20))
    timing_report: dict[str, Any] = {
        "format": "musicm8-v10-render-clock-v2",
        "bpm": bpm,
        "style": style,
        "swing": swing,
        "roles": {},
        "note": "All learned MIDI roles are hard-locked to one sixteenth/swing clock. Style-directed GM programs prevent random instrument choices; non-core electronic percussion is removed before rendering.",
    }
    print("🥁 V10 RENDER CLOCK — hard-locking every stem to one shared groove")
    for role in ROLES:
        midi = stems_dir / f"{role}.mid"
        if midi.exists():
            timing_report["roles"][role] = tighten_role_midi(midi, bpm, swing, role, style)
            info = timing_report["roles"][role]
            print(
                f"   {role}: {info['events']} events | max correction {info['max_start_correction_ms']} ms "
                f"| programs={info['programs']}"
            )
    (project / "render_timing_report.json").write_text(json.dumps(timing_report, indent=2), encoding="utf-8")

    fluidsynth = shutil.which("fluidsynth")
    if not fluidsynth:
        raise RuntimeError("fluidsynth executable not found. The current Colab notebook installs it automatically.")
    soundfont = choose_soundfont(args.soundfont)
    print("🎹 Sample instrument bank:", soundfont)

    pm = pretty_midi.PrettyMIDI(str(arrangement))
    bars = int(plan.get("bars", 32))
    planned_end = bars * 4.0 * 60.0 / max(1.0, bpm)
    music_end = max(planned_end, float(pm.get_end_time()))
    total_n = int(round((music_end + 1.25) * SR))

    audio_dir = project / "audio_stems"
    shutil.rmtree(audio_dir, ignore_errors=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "musicm8-soundfont-render-v3",
        "source": "style-directed sample-based GM SoundFont",
        "style": style,
        "soundfont": str(soundfont),
        "sample_rate": SR,
        "roles": {},
        "timing": "All role MIDIs are hard-locked to one shared 1/16/swing clock immediately before FluidSynth; no sample slicing/time-stretching/reference-loop placement is used.",
        "timing_report": str(project / "render_timing_report.json"),
    }
    mix = np.zeros((2, total_n), dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="musicm8_sf_") as tmp:
        tmpdir = Path(tmp)
        for role in ROLES:
            midi = stems_dir / f"{role}.mid"
            out = audio_dir / f"{role}.wav"
            if not midi.exists():
                report["roles"][role] = {"status": "missing_midi"}
                continue
            temp_wav = tmpdir / f"{role}.wav"
            render_midi(fluidsynth, soundfont, midi, temp_wav)
            x, sr = load_stereo(temp_wav)
            if sr != SR:
                raise ValueError(f"FluidSynth returned {sr} Hz, expected {SR}")
            x = fit(x, total_n)
            peak = float(np.max(np.abs(x))) + 1e-9
            if peak > 0.98:
                x *= 0.98 / peak
            sf.write(out, x.T, SR, subtype="PCM_24")
            mix += x
            report["roles"][role] = {
                "status": "rendered",
                "midi": str(midi),
                "audio": str(out),
                "peak": round(float(np.max(np.abs(x))), 5),
                "programs": timing_report["roles"].get(role, {}).get("programs", []),
            }

    if not any(v.get("status") == "rendered" for v in report["roles"].values()):
        raise RuntimeError("No MIDI role stems could be rendered")

    peak = float(np.max(np.abs(mix))) + 1e-9
    if peak > 0.90:
        mix *= 0.90 / peak
    raw_master = project / "master_soundfont_raw.wav"
    sf.write(raw_master, mix.T, SR, subtype="PCM_24")
    report["raw_master"] = str(raw_master)
    (project / "soundfont_render_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("✅ SAMPLE-BASED MULTITRACK RENDER")
    print("✅ Shared clock + style-directed instrument programs + cleaned electronic drum vocabulary")
    print("No reference stem chunks. No pitch-shifted old-song audio. No hand-built oscillator instruments in the audible source.")
    print("Audio stems:", audio_dir)


if __name__ == "__main__":
    main()
