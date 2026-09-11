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
    # Last resort: search standard system sound directories only.
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


def main() -> None:
    p = argparse.ArgumentParser(description="Render Musicm8's learned multitrack MIDI using real sampled SoundFont instruments instead of the primitive oscillator engine.")
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

    fluidsynth = shutil.which("fluidsynth")
    if not fluidsynth:
        raise RuntimeError("fluidsynth executable not found. The current Colab notebook installs it automatically.")
    soundfont = choose_soundfont(args.soundfont)
    print("🎹 Sample instrument bank:", soundfont)

    pm = pretty_midi.PrettyMIDI(str(arrangement))
    bpm = float(plan.get("bpm", 120.0))
    bars = int(plan.get("bars", 32))
    planned_end = bars * 4.0 * 60.0 / max(1.0, bpm)
    music_end = max(planned_end, float(pm.get_end_time()))
    total_n = int(round((music_end + 1.25) * SR))

    audio_dir = project / "audio_stems"
    shutil.rmtree(audio_dir, ignore_errors=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "musicm8-soundfont-render-v1",
        "source": "sample-based GM SoundFont",
        "soundfont": str(soundfont),
        "sample_rate": SR,
        "roles": {},
        "timing": "MIDI event timestamps are rendered directly; no sample slicing/time-stretching/reference-loop placement is used.",
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
            # Only catch pathological SoundFont peaks. Level/balance is handled by the mix stage.
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
    print("No reference stem chunks. No pitch-shifted old-song audio. No hand-built oscillator instruments in the audible source.")
    print("Audio stems:", audio_dir)


if __name__ == "__main__":
    main()
