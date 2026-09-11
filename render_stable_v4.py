from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_matched_v2 import render_project


def clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return default


STYLE = {
    "uk_garage": {"kick": (142, 47, 0.026, 0.18, 0.28), "snare": (188, 0.12, 0.76, 0.58), "hat": (0.042, 0.22, 6500, 0.28), "bass_cut": 1550, "bass_drive": 0.15, "chord_cut": 5600, "lead_cut": 6800},
    "house": {"kick": (128, 48, 0.030, 0.22, 0.22), "snare": (180, 0.13, 0.70, 0.56), "hat": (0.045, 0.25, 6200, 0.32), "bass_cut": 1800, "bass_drive": 0.12, "chord_cut": 6200, "lead_cut": 7200},
    "techno": {"kick": (122, 46, 0.032, 0.25, 0.20), "snare": (175, 0.11, 0.72, 0.62), "hat": (0.038, 0.20, 6800, 0.34), "bass_cut": 1450, "bass_drive": 0.18, "chord_cut": 5000, "lead_cut": 7000},
    "dnb": {"kick": (155, 50, 0.022, 0.16, 0.34), "snare": (205, 0.10, 0.82, 0.70), "hat": (0.032, 0.16, 7200, 0.26), "bass_cut": 1900, "bass_drive": 0.20, "chord_cut": 6000, "lead_cut": 7600},
    "trap": {"kick": (115, 42, 0.035, 0.28, 0.16), "snare": (198, 0.12, 0.82, 0.64), "hat": (0.028, 0.18, 7600, 0.22), "bass_cut": 1050, "bass_drive": 0.11, "chord_cut": 4300, "lead_cut": 6400},
    "hiphop": {"kick": (118, 44, 0.034, 0.24, 0.18), "snare": (190, 0.14, 0.76, 0.52), "hat": (0.040, 0.20, 6100, 0.22), "bass_cut": 1350, "bass_drive": 0.11, "chord_cut": 4800, "lead_cut": 6100},
    "ambient": {"kick": (110, 45, 0.040, 0.25, 0.10), "snare": (175, 0.18, 0.60, 0.42), "hat": (0.060, 0.30, 5200, 0.18), "bass_cut": 1800, "bass_drive": 0.05, "chord_cut": 5200, "lead_cut": 5800},
    "electronic": {"kick": (132, 47, 0.030, 0.20, 0.22), "snare": (188, 0.13, 0.74, 0.58), "hat": (0.042, 0.22, 6400, 0.28), "bass_cut": 1600, "bass_drive": 0.13, "chord_cut": 5600, "lead_cut": 6800},
}


def common_tonal() -> dict[str, Any]:
    return {
        "fm_mix": 0.0,
        "fm_ratio": 2.0,
        "fm_index": 0.0,
        "noise_mix": 0.0,
        "lfo_depth": 0.0,
        "eq_low_db": 0.0,
        "eq_mid_db": 0.0,
        "eq_high_db": 0.0,
        "comp_threshold_db": -18.0,
        "comp_ratio": 1.8,
    }


def stable_patches(style: str) -> dict[str, dict[str, Any]]:
    p = STYLE.get(style, STYLE["electronic"])
    ks, ke, sw, kd, kc = p["kick"]
    st, sd, sn, sb = p["snare"]
    hd, hod, hp, hm = p["hat"]

    drums = {
        "gain": 0.76,
        "drive": 0.08,
        "room": 0.04,
        "width": 0.0,
        "kick": {"pitch_start": ks, "pitch_end": ke, "sweep": sw, "decay": kd, "click": kc},
        "snare": {"tone_hz": st, "decay": sd, "noise": sn, "brightness": sb},
        "hat": {"decay": hd, "open_decay": hod, "highpass": hp, "metallic": hm},
        "eq_low_db": 0.0,
        "eq_mid_db": 0.0,
        "eq_high_db": -0.4,
        "comp_threshold_db": -15.0,
        "comp_ratio": 2.2,
    }

    bass = {
        **common_tonal(),
        "wave": "wavetable",
        "harmonics": [1.0, 0.38, 0.16, 0.07, 0.03],
        "sub": 0.82,
        "cutoff": float(p["bass_cut"]),
        "filter_env": 0.16,
        "detune": 0.01,
        "attack": 0.004,
        "decay": 0.16,
        "sustain": 0.72,
        "release": 0.09,
        "drive": float(p["bass_drive"]),
        "width": 0.0,
        "gain": 0.50,
        "eq_low_db": 0.8,
        "eq_mid_db": -0.5,
        "eq_high_db": -1.0,
        "comp_threshold_db": -17.0,
        "comp_ratio": 2.2,
    }

    chords = {
        **common_tonal(),
        "wave": "wavetable",
        "harmonics": [1.0, 0.30, 0.15, 0.08, 0.04, 0.02],
        "cutoff": float(p["chord_cut"]),
        "filter_env": 0.12,
        "detune": 0.02,
        "attack": 0.010,
        "decay": 0.22,
        "sustain": 0.67,
        "release": 0.18,
        "drive": 0.025,
        "reverb": 0.10,
        "width": 0.0,
        "gain": 0.27,
        "eq_low_db": -1.5,
        "eq_mid_db": 0.2,
        "eq_high_db": -0.3,
    }

    melody = {
        **common_tonal(),
        "wave": "wavetable",
        "harmonics": [1.0, 0.42, 0.20, 0.09, 0.04],
        "cutoff": float(p["lead_cut"]),
        "filter_env": 0.18,
        "detune": 0.01,
        "attack": 0.004,
        "decay": 0.12,
        "sustain": 0.58,
        "release": 0.10,
        "drive": 0.02,
        "delay": 0.04,
        "reverb": 0.08,
        "width": 0.0,
        "gain": 0.24,
        "eq_low_db": -2.0,
        "eq_mid_db": 0.3,
        "eq_high_db": 0.0,
    }
    return {"drums": drums, "bass": bass, "chords": chords, "melody": melody}


def safe_plan(plan: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(plan))
    # Producer chooses style/arrangement, but the foundation render cannot reintroduce
    # delay-based widening, large detune, aggressive distortion or reference-audio layers.
    sd = out.setdefault("sound_design", {})
    sd["drums"] = {"brightness": 0.55, "drive": 0.07, "room": 0.04}
    sd["bass"] = {"sub": 0.80, "drive": 0.12, "filter_motion": 0.20, "width": 0.0}
    sd["chords"] = {"brightness": 0.50, "detune": 0.02, "reverb": 0.10, "width": 0.0}
    sd["melody"] = {"brightness": 0.56, "detune": 0.01, "delay": 0.04, "reverb": 0.08, "width": 0.0}

    mix = out.setdefault("production", {}).setdefault("mix", {})
    mix["sidechain_depth"] = clamp(mix.get("sidechain_depth", 0.35), 0.12, 0.46, 0.35)
    mix["sidechain_release"] = clamp(mix.get("sidechain_release", 0.16), 0.10, 0.24, 0.16)
    mix["bus_ratio"] = clamp(mix.get("bus_ratio", 1.35), 1.10, 1.55, 1.35)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Render a Musicm8 arrangement with stable harmonic-safe genre presets and no reference-audio timing contamination.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--midi", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    style = str(plan.get("style", "electronic"))
    clean_plan = safe_plan(plan)
    patches = stable_patches(style)

    args.out.mkdir(parents=True, exist_ok=True)
    stable_plan_path = args.out / "plan_stable_v4.json"
    stable_patch_path = args.out / "stable_patches_v4.json"
    matched_path = args.out / "stable_matched_v4.json"
    stable_plan_path.write_text(json.dumps(clean_plan, indent=2), encoding="utf-8")
    stable_patch_path.write_text(json.dumps({"format": "musicm8-stable-patches-v4", "style": style, "patches": patches}, indent=2), encoding="utf-8")
    matched_path.write_text(json.dumps({
        "format": "musicm8-stable-render-v4",
        "roles": {role: {"patch": patch, "source": "curated-stable-genre-preset-v4"} for role, patch in patches.items()},
    }, indent=2), encoding="utf-8")

    master = render_project(stable_plan_path, args.midi, matched_path, args.out)
    report = {
        "format": "musicm8-stable-render-report-v4",
        "style": style,
        "master": str(master),
        "timing_source": "strict MIDI grid",
        "pitched_reference_audio": False,
        "reference_drum_audio": False,
        "inverse_spectral_patch_search": False,
        "delay_based_stereo_width": False,
        "note": "Foundation mode removes reference stem layers, free inverse sound matching and delay-based widening. Every audible onset comes from one MIDI clock and stable harmonic-safe source patches.",
    }
    (args.out / "stable_render_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("✅ STABLE FOUNDATION RENDER V4")
    print("Style:", style)
    print("✅ No reference-audio layers")
    print("✅ No continuous FM / aggressive detune")
    print("✅ No delay-based stereo widening")
    print("✅ All audible events follow the strict MIDI grid")
    print("Master base:", master)


if __name__ == "__main__":
    main()
