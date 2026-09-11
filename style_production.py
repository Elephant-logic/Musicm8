from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        x = float(v)
        return max(lo, min(hi, x))
    except Exception:
        return default


def blend(a: Any, b: Any, amount: float, default: float) -> float:
    return (1.0 - amount) * clamp(a, -1e9, 1e9, default) + amount * clamp(b, -1e9, 1e9, default)


PROFILES: dict[str, dict[str, Any]] = {
    "uk_garage": {
        "groove": {"swing": 0.19, "drum_density": 0.80, "bass_density": 0.68, "chord_density": 0.58, "melody_density": 0.34},
        "arrangement": {"phrase_bars": 4, "humanize_ms": 7, "drum_variation": 0.58, "bass_syncopation": 0.78, "chord_rhythm": "stabs", "melody_rest": 0.34, "transition_strength": 0.72},
        "harmony": {"sevenths": 0.78, "ninths": 0.42, "voice_leading": 0.92},
        "sound_design": {
            "drums": {"brightness": 0.62, "drive": 0.20, "room": 0.11},
            "bass": {"sub": 0.72, "drive": 0.34, "filter_motion": 0.52, "width": 0.04},
            "chords": {"brightness": 0.56, "detune": 0.16, "reverb": 0.30, "width": 0.72},
            "melody": {"brightness": 0.62, "detune": 0.07, "delay": 0.24, "reverb": 0.24, "width": 0.42},
        },
        "mix": {"drums_db": -17.0, "bass_db": -19.5, "chords_db": -23.0, "melody_db": -24.0, "sidechain_depth": 0.47, "sidechain_release": 0.16, "bus_ratio": 1.45, "bus_drive": 0.025, "low_tilt_db": -0.9, "mid_tilt_db": 0.7, "high_tilt_db": 0.9},
    },
    "house": {
        "groove": {"swing": 0.045, "drum_density": 0.84, "bass_density": 0.62, "chord_density": 0.62, "melody_density": 0.34},
        "arrangement": {"phrase_bars": 8, "humanize_ms": 4, "drum_variation": 0.42, "bass_syncopation": 0.48, "chord_rhythm": "offbeat", "melody_rest": 0.38, "transition_strength": 0.78},
        "harmony": {"sevenths": 0.58, "ninths": 0.26, "voice_leading": 0.88},
        "sound_design": {"drums": {"brightness": 0.60, "drive": 0.15, "room": 0.10}, "bass": {"sub": 0.62, "drive": 0.24, "filter_motion": 0.42, "width": 0.03}, "chords": {"brightness": 0.58, "detune": 0.13, "reverb": 0.24, "width": 0.62}, "melody": {"brightness": 0.66, "detune": 0.06, "delay": 0.18, "reverb": 0.20, "width": 0.38}},
        "mix": {"drums_db": -16.5, "bass_db": -19.0, "chords_db": -23.5, "melody_db": -24.5, "sidechain_depth": 0.55, "sidechain_release": 0.20, "bus_ratio": 1.55, "bus_drive": 0.020, "low_tilt_db": -0.6, "mid_tilt_db": 0.5, "high_tilt_db": 0.8},
    },
    "techno": {
        "groove": {"swing": 0.02, "drum_density": 0.90, "bass_density": 0.72, "chord_density": 0.30, "melody_density": 0.22},
        "arrangement": {"phrase_bars": 8, "humanize_ms": 2, "drum_variation": 0.36, "bass_syncopation": 0.54, "chord_rhythm": "pulse", "melody_rest": 0.56, "transition_strength": 0.88},
        "harmony": {"sevenths": 0.28, "ninths": 0.14, "voice_leading": 0.82},
        "sound_design": {"drums": {"brightness": 0.66, "drive": 0.30, "room": 0.08}, "bass": {"sub": 0.64, "drive": 0.42, "filter_motion": 0.62, "width": 0.02}, "chords": {"brightness": 0.46, "detune": 0.10, "reverb": 0.42, "width": 0.76}, "melody": {"brightness": 0.70, "detune": 0.05, "delay": 0.28, "reverb": 0.30, "width": 0.46}},
        "mix": {"drums_db": -16.0, "bass_db": -18.5, "chords_db": -25.0, "melody_db": -25.0, "sidechain_depth": 0.50, "sidechain_release": 0.14, "bus_ratio": 1.7, "bus_drive": 0.045, "low_tilt_db": -0.4, "mid_tilt_db": 0.3, "high_tilt_db": 0.7},
    },
    "dnb": {
        "groove": {"swing": 0.03, "drum_density": 0.94, "bass_density": 0.80, "chord_density": 0.42, "melody_density": 0.36},
        "arrangement": {"phrase_bars": 4, "humanize_ms": 3, "drum_variation": 0.72, "bass_syncopation": 0.86, "chord_rhythm": "pads", "melody_rest": 0.34, "transition_strength": 0.92},
        "harmony": {"sevenths": 0.62, "ninths": 0.34, "voice_leading": 0.90},
        "sound_design": {"drums": {"brightness": 0.76, "drive": 0.24, "room": 0.07}, "bass": {"sub": 0.76, "drive": 0.48, "filter_motion": 0.64, "width": 0.05}, "chords": {"brightness": 0.54, "detune": 0.17, "reverb": 0.40, "width": 0.80}, "melody": {"brightness": 0.70, "detune": 0.05, "delay": 0.18, "reverb": 0.26, "width": 0.50}},
        "mix": {"drums_db": -15.5, "bass_db": -18.5, "chords_db": -24.5, "melody_db": -24.0, "sidechain_depth": 0.50, "sidechain_release": 0.10, "bus_ratio": 1.55, "bus_drive": 0.030, "low_tilt_db": -0.8, "mid_tilt_db": 0.8, "high_tilt_db": 1.1},
    },
    "trap": {
        "groove": {"swing": 0.07, "drum_density": 0.74, "bass_density": 0.72, "chord_density": 0.42, "melody_density": 0.38},
        "arrangement": {"phrase_bars": 4, "humanize_ms": 5, "drum_variation": 0.62, "bass_syncopation": 0.74, "chord_rhythm": "half_time", "melody_rest": 0.36, "transition_strength": 0.78},
        "harmony": {"sevenths": 0.36, "ninths": 0.18, "voice_leading": 0.84},
        "sound_design": {"drums": {"brightness": 0.66, "drive": 0.18, "room": 0.05}, "bass": {"sub": 0.88, "drive": 0.28, "filter_motion": 0.22, "width": 0.01}, "chords": {"brightness": 0.42, "detune": 0.10, "reverb": 0.36, "width": 0.66}, "melody": {"brightness": 0.60, "detune": 0.04, "delay": 0.20, "reverb": 0.32, "width": 0.44}},
        "mix": {"drums_db": -17.0, "bass_db": -17.5, "chords_db": -25.0, "melody_db": -23.5, "sidechain_depth": 0.30, "sidechain_release": 0.18, "bus_ratio": 1.4, "bus_drive": 0.020, "low_tilt_db": -0.3, "mid_tilt_db": 0.4, "high_tilt_db": 0.7},
    },
    "hiphop": {
        "groove": {"swing": 0.12, "drum_density": 0.70, "bass_density": 0.56, "chord_density": 0.56, "melody_density": 0.30},
        "arrangement": {"phrase_bars": 4, "humanize_ms": 10, "drum_variation": 0.54, "bass_syncopation": 0.56, "chord_rhythm": "laid_back", "melody_rest": 0.46, "transition_strength": 0.58},
        "harmony": {"sevenths": 0.72, "ninths": 0.42, "voice_leading": 0.94},
        "sound_design": {"drums": {"brightness": 0.50, "drive": 0.20, "room": 0.08}, "bass": {"sub": 0.64, "drive": 0.24, "filter_motion": 0.20, "width": 0.02}, "chords": {"brightness": 0.40, "detune": 0.12, "reverb": 0.28, "width": 0.64}, "melody": {"brightness": 0.54, "detune": 0.05, "delay": 0.12, "reverb": 0.24, "width": 0.40}},
        "mix": {"drums_db": -16.5, "bass_db": -19.0, "chords_db": -23.0, "melody_db": -24.0, "sidechain_depth": 0.20, "sidechain_release": 0.22, "bus_ratio": 1.35, "bus_drive": 0.018, "low_tilt_db": -0.4, "mid_tilt_db": 0.5, "high_tilt_db": 0.3},
    },
    "ambient": {
        "groove": {"swing": 0.00, "drum_density": 0.24, "bass_density": 0.34, "chord_density": 0.76, "melody_density": 0.24},
        "arrangement": {"phrase_bars": 8, "humanize_ms": 14, "drum_variation": 0.20, "bass_syncopation": 0.20, "chord_rhythm": "pads", "melody_rest": 0.62, "transition_strength": 0.42},
        "harmony": {"sevenths": 0.82, "ninths": 0.72, "voice_leading": 0.98},
        "sound_design": {"drums": {"brightness": 0.38, "drive": 0.05, "room": 0.35}, "bass": {"sub": 0.50, "drive": 0.08, "filter_motion": 0.36, "width": 0.10}, "chords": {"brightness": 0.46, "detune": 0.24, "reverb": 0.72, "width": 0.92}, "melody": {"brightness": 0.52, "detune": 0.12, "delay": 0.42, "reverb": 0.66, "width": 0.82}},
        "mix": {"drums_db": -24.0, "bass_db": -23.0, "chords_db": -19.5, "melody_db": -22.5, "sidechain_depth": 0.08, "sidechain_release": 0.40, "bus_ratio": 1.15, "bus_drive": 0.008, "low_tilt_db": -0.5, "mid_tilt_db": 0.2, "high_tilt_db": 0.8},
    },
}
PROFILES["electronic"] = {
    "groove": {"swing": 0.04, "drum_density": 0.72, "bass_density": 0.60, "chord_density": 0.54, "melody_density": 0.38},
    "arrangement": {"phrase_bars": 4, "humanize_ms": 5, "drum_variation": 0.48, "bass_syncopation": 0.50, "chord_rhythm": "mixed", "melody_rest": 0.40, "transition_strength": 0.68},
    "harmony": {"sevenths": 0.48, "ninths": 0.24, "voice_leading": 0.88},
    "sound_design": {"drums": {"brightness": 0.58, "drive": 0.16, "room": 0.11}, "bass": {"sub": 0.66, "drive": 0.26, "filter_motion": 0.38, "width": 0.04}, "chords": {"brightness": 0.52, "detune": 0.15, "reverb": 0.32, "width": 0.68}, "melody": {"brightness": 0.62, "detune": 0.07, "delay": 0.20, "reverb": 0.26, "width": 0.44}},
    "mix": {"drums_db": -17.0, "bass_db": -19.5, "chords_db": -23.5, "melody_db": -24.0, "sidechain_depth": 0.36, "sidechain_release": 0.18, "bus_ratio": 1.4, "bus_drive": 0.020, "low_tilt_db": -0.7, "mid_tilt_db": 0.5, "high_tilt_db": 0.7},
}


def reference_character(library: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    by_id = {str(r.get("id")): r for r in library.get("references", [])}
    out: dict[str, Any] = {}
    for role, rid in plan.get("references", {}).items():
        ref = by_id.get(str(rid))
        sonic = (ref or {}).get("sonic", {}).get("other" if role in {"chords", "melody"} else role, {})
        if sonic:
            out[role] = {k: sonic.get(k) for k in ("spectral_centroid_hz", "sub_ratio", "bass_ratio", "high_mid_ratio", "air_ratio", "onset_rate", "dynamic_range_db")}
    return out


def enrich(plan: dict[str, Any], library: dict[str, Any]) -> dict[str, Any]:
    style = str(plan.get("style", "electronic"))
    profile = PROFILES.get(style, PROFILES["electronic"])
    out = json.loads(json.dumps(plan))

    groove = dict(out.get("groove", {}))
    for k, v in profile["groove"].items():
        groove[k] = round(blend(groove.get(k, v), v, 0.58, v), 4)
    out["groove"] = groove

    sound = dict(out.get("sound_design", {}))
    for role, controls in profile["sound_design"].items():
        current = dict(sound.get(role, {}))
        for k, v in controls.items():
            current[k] = round(blend(current.get(k, v), v, 0.52, v), 4)
        sound[role] = current
    out["sound_design"] = sound

    mix = dict(out.get("mix", {}))
    mix["master_drive"] = round(blend(mix.get("master_drive", profile["mix"]["bus_drive"]), profile["mix"]["bus_drive"], 0.55, profile["mix"]["bus_drive"]), 4)
    out["mix"] = mix

    out["production"] = {
        "format": "musicm8-production-director-v1",
        "style": style,
        "groove": profile["groove"],
        "arrangement": profile["arrangement"],
        "harmony": profile["harmony"],
        "mix": profile["mix"],
        "reference_character": reference_character(library, out),
        "note": "Genre-specific production knowledge applied after the AI producer plan. This controls phrasing, harmony extensions, groove, humanization, sound design and mix behavior.",
    }
    out["format"] = "musicm8-producer-plan-v2"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Apply genre-aware production knowledge to a Musicm8 AI plan.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    library = json.loads(args.references.read_text(encoding="utf-8"))
    result = enrich(plan, library)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("✅ Production director:", args.out)
    print("Style:", result.get("style"))
    print("Groove:", result.get("groove"))
    print("Production:", result.get("production", {}).get("arrangement"))


if __name__ == "__main__":
    main()
