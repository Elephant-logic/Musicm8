from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pretty_midi


def find_soundfont(explicit: Path | None) -> Path | None:
    if explicit and explicit.exists():
        return explicit
    candidates = [
        Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
        Path("/usr/share/sounds/sf2/FluidR3_GS.sf2"),
        Path("/usr/share/sounds/sf2/TimGM6mb.sf2"),
    ]
    for root in (Path("/content/drive/MyDrive/Musicm8/soundfonts"), Path("/usr/share/sounds/sf2")):
        if root.exists():
            candidates.extend(sorted(root.glob("*.sf2")))
    return next((p for p in candidates if p.exists()), None)


def build_project_manifest(midi_path: Path, out_dir: Path) -> Path:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    _, tempi = pm.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) else 120.0
    tracks = []
    for inst in pm.instruments:
        role = (inst.name or ("drums" if inst.is_drum else "instrument")).lower()
        tracks.append({
            "name": role,
            "midi": f"midi_stems/{role.replace(' ', '_')}.mid",
            "is_drum": bool(inst.is_drum),
            "gm_program": int(inst.program),
            "suggested_vst_role": {
                "drums": "drum sampler / acoustic or electronic kit",
                "bass": "bass synth / electric bass",
                "chords": "piano / keys / pad synth",
                "melody": "lead synth / guitar / keys",
            }.get(role, "instrument"),
        })
    payload = {
        "format": "musicm8-daw-project-v1",
        "bpm": bpm,
        "time_signature": "4/4",
        "arrangement_midi": midi_path.name,
        "tracks": tracks,
        "note": "Import arrangement.mid or the individual MIDI stems into a DAW and assign your own VST instruments. The WAV is only a SoundFont preview.",
    }
    path = out_dir / "project.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def render_fluidsynth(midi: Path, out: Path, soundfont: Path) -> None:
    exe = shutil.which("fluidsynth")
    if not exe:
        raise RuntimeError("fluidsynth is not installed")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-ni", "-F", str(out), "-r", "44100", str(soundfont), str(midi)]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Render Musicm8 MIDI with a SoundFont and emit a DAW project manifest.")
    p.add_argument("--midi", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Preview WAV path")
    p.add_argument("--soundfont", type=Path, default=None)
    args = p.parse_args()

    if not args.midi.exists():
        raise FileNotFoundError(args.midi)
    project = build_project_manifest(args.midi, args.midi.parent)
    sf2 = find_soundfont(args.soundfont)
    if sf2 is None:
        print("⚠️ No SoundFont found. MIDI + project files are ready, but no WAV preview was rendered.")
        print("Project:", project)
        return
    render_fluidsynth(args.midi, args.out, sf2)
    print("✅ Preview:", args.out)
    print("✅ SoundFont:", sf2)
    print("✅ DAW project manifest:", project)


if __name__ == "__main__":
    main()
