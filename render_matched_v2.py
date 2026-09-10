from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import soundfile as sf

from musicm8_synth_v2 import SR, block_compressor, clamp, render_instrument, role_fx, saturate, set_rms, sidechain_gain


def blend(current: Any, target: Any, amount: float = 0.25) -> float:
    try:
        return (1.0 - amount) * float(current) + amount * float(target)
    except Exception:
        return float(current)


def apply_producer_controls(role: str, patch: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(patch))
    if role == "bass":
        for key in ("sub", "drive", "width"):
            if key in controls:
                out[key] = blend(out.get(key, controls[key]), controls[key], 0.28)
        if "filter_motion" in controls:
            motion = clamp(controls["filter_motion"], 0, 1, 0.35)
            out["filter_env"] = blend(out.get("filter_env", 0.3), motion, 0.40)
            out["cutoff"] = float(out.get("cutoff", 1200)) * (0.82 + 0.36 * motion)
    elif role == "drums":
        if "drive" in controls:
            out["drive"] = blend(out.get("drive", 0.1), controls["drive"], 0.25)
        if "room" in controls:
            out["room"] = blend(out.get("room", 0.1), controls["room"], 0.25)
        if "brightness" in controls:
            b = clamp(controls["brightness"], 0, 1, 0.5)
            out.setdefault("snare", {})["brightness"] = blend(out.get("snare", {}).get("brightness", 0.5), b, 0.35)
            out.setdefault("hat", {})["highpass"] = float(out.get("hat", {}).get("highpass", 5200)) * (0.82 + 0.42 * b)
    else:
        if "brightness" in controls:
            b = clamp(controls["brightness"], 0, 1, 0.5)
            out["cutoff"] = float(out.get("cutoff", 5000)) * (0.75 + 0.50 * b)
        for key in ("detune", "reverb", "delay", "width"):
            if key in controls:
                out[key] = blend(out.get(key, controls[key]), controls[key], 0.24)
    return out


def fallback_patch(role: str) -> dict[str, Any]:
    if role == "drums":
        return {"gain": 0.78, "drive": 0.10, "room": 0.12, "width": 0.10, "kick": {"pitch_start": 150, "pitch_end": 45, "sweep": 0.028, "decay": 0.20, "click": 0.25}, "snare": {"tone_hz": 190, "decay": 0.14, "noise": 0.72, "brightness": 0.55}, "hat": {"decay": 0.05, "open_decay": 0.28, "highpass": 5200, "metallic": 0.5}, "eq_low_db": 0, "eq_mid_db": 0, "eq_high_db": 0, "comp_threshold_db": -16, "comp_ratio": 2.5}
    base = {"wave": "saw", "harmonics": [1.0], "cutoff": 6000, "filter_env": 0.25, "fm_mix": 0.06, "fm_ratio": 2.0, "fm_index": 1.0, "noise_mix": 0.01, "detune": 0.08, "attack": 0.01, "decay": 0.20, "sustain": 0.65, "release": 0.18, "drive": 0.08, "width": 0.25, "gain": 0.30, "eq_low_db": 0, "eq_mid_db": 0, "eq_high_db": 0, "comp_threshold_db": -18, "comp_ratio": 2.0}
    if role == "bass":
        base.update({"sub": 0.65, "cutoff": 1800, "width": 0.03, "gain": 0.58})
    elif role == "chords":
        base.update({"detune": 0.18, "reverb": 0.32, "width": 0.60, "gain": 0.30})
    else:
        base.update({"delay": 0.16, "reverb": 0.24, "width": 0.34, "gain": 0.27})
    return base


def render_project(plan_path: Path, midi_path: Path, patches_path: Path, out_dir: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    matched = json.loads(patches_path.read_text(encoding="utf-8")) if patches_path.exists() else {"roles": {}}
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    bpm = float(plan["bpm"])
    total_s = max(pm.get_end_time() + 1.5, float(plan["bars"]) * 4 * 60 / bpm + 1)
    total_n = int(total_s * SR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = out_dir / "audio_stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    by_name = {(inst.name or "").lower(): inst for inst in pm.instruments}
    drums_inst = by_name.get("drums")
    rendered: dict[str, np.ndarray] = {}
    patch_report: dict[str, Any] = {}

    for role in ("drums", "bass", "chords", "melody"):
        inst = by_name.get(role)
        if inst is None:
            continue
        match = matched.get("roles", {}).get(role, {})
        base = match.get("patch") or fallback_patch(role)
        final_patch = apply_producer_controls(role, base, plan.get("sound_design", {}).get(role, {}))
        audio = role_fx(render_instrument(inst, role, final_patch, total_n), role, final_patch, bpm)
        rendered[role] = audio
        patch_report[role] = {"reference_id": plan.get("references", {}).get(role), "source": "inverse-synthesis-v2" if match.get("patch") else "v2-fallback", "match_distance": match.get("best_distance"), "base_patch": base, "producer_controls": plan.get("sound_design", {}).get(role, {}), "final_patch": final_patch}

    targets = {"drums": -16.0, "bass": -18.0, "chords": -24.0, "melody": -23.0}
    for role, audio in list(rendered.items()):
        rendered[role] = set_rms(audio, targets.get(role, -22.0), max_gain_db=8.0)

    sc_depth = 0.58 if plan.get("style") in {"uk_garage", "house", "techno", "dnb"} else 0.42
    sc = sidechain_gain(drums_inst, total_n, depth=sc_depth, release_s=0.18)
    for role in ("bass", "chords"):
        if role in rendered:
            rendered[role] = (rendered[role] * sc[None]).astype(np.float32)

    mix = np.zeros((2, total_n), dtype=np.float32)
    for role, audio in rendered.items():
        sf.write(stems_dir / f"{role}.wav", audio.T, SR, subtype="PCM_24")
        mix += audio

    mix = block_compressor(mix, threshold_db=-10.0, ratio=1.8, makeup_db=0.0)
    mix = saturate(mix, float(plan.get("mix", {}).get("master_drive", 0.04)) * 0.65)
    peak = float(np.max(np.abs(mix))) + 1e-9
    target = 10 ** (clamp(plan.get("mix", {}).get("target_peak_db", -1.0), -6, -0.1, -1.0) / 20)
    if peak > target:
        mix *= target / peak

    master = out_dir / "master.wav"
    sf.write(master, mix.T, SR, subtype="PCM_24")
    (out_dir / "synth_patches.json").write_text(json.dumps(patch_report, indent=2), encoding="utf-8")
    return master


def main() -> None:
    p = argparse.ArgumentParser(description="Render Musicm8 v2 inverse-matched synth/FX project.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--midi", type=Path, required=True)
    p.add_argument("--patches", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    master = render_project(args.plan, args.midi, args.patches, args.out)
    print(f"✅ Musicm8 Synth v2 master: {master}")
    print(f"✅ Audio stems: {args.out / 'audio_stems'}")
    print(f"✅ Final synth patches: {args.out / 'synth_patches.json'}")


if __name__ == "__main__":
    main()
