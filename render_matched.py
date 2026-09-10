from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import soundfile as sf

from musicm8_synth import (
    SR,
    clamp,
    echo,
    patch_from_fingerprint,
    render_instrument,
    reverb,
    saturate,
    sonic_for_reference,
)


def blend(current: Any, target: Any, amount: float = 0.28) -> float:
    try:
        a = float(current)
        b = float(target)
        return (1.0 - amount) * a + amount * b
    except Exception:
        return float(current)


def apply_producer_controls(role: str, patch: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    """Nudge a matched reference patch toward the AI request without destroying its reference timbre."""
    out = dict(patch)
    if role == "bass":
        for key in ("sub", "drive", "width"):
            if key in controls:
                out[key] = blend(out.get(key, controls[key]), controls[key], 0.30)
        if "filter_motion" in controls:
            out["cutoff"] = float(out.get("cutoff", 1200.0)) * (
                0.78 + 0.48 * clamp(controls["filter_motion"], 0, 1, 0.35)
            )
    elif role == "drums":
        for key in ("brightness", "drive", "room"):
            if key in controls:
                out[key] = blend(out.get(key, controls[key]), controls[key], 0.25)
    else:
        if "brightness" in controls:
            out["cutoff"] = float(out.get("cutoff", 5000.0)) * (
                0.72 + 0.56 * clamp(controls["brightness"], 0, 1, 0.5)
            )
        for key in ("detune", "reverb", "delay", "width"):
            if key in controls:
                out[key] = blend(out.get(key, controls[key]), controls[key], 0.25)
    return out


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


def render_project(
    plan_path: Path,
    midi_path: Path,
    library_path: Path,
    patches_path: Path,
    out_dir: Path,
) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    library = json.loads(library_path.read_text(encoding="utf-8"))
    matched = json.loads(patches_path.read_text(encoding="utf-8")) if patches_path.exists() else {"roles": {}}
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    bpm = float(plan["bpm"])
    total_s = max(pm.get_end_time() + 1.5, float(plan["bars"]) * 4 * 60 / bpm + 1)
    total_n = int(total_s * SR)

    out_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = out_dir / "audio_stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    by_name = {(inst.name or "").lower(): inst for inst in pm.instruments}
    mix = np.zeros((2, total_n), dtype=np.float32)
    patch_report: dict[str, Any] = {}

    for role in ("drums", "bass", "chords", "melody"):
        inst = by_name.get(role)
        if inst is None:
            continue
        ref_id = plan.get("references", {}).get(role)
        match = matched.get("roles", {}).get(role, {})
        features = sonic_for_reference(library, ref_id, role)

        if match.get("patch"):
            base_patch = dict(match["patch"])
            source = "inverse-synthesis-match"
        else:
            base_patch = patch_from_fingerprint(role, features, {})
            source = "spectral-fingerprint-fallback"

        final_patch = apply_producer_controls(
            role,
            base_patch,
            plan.get("sound_design", {}).get(role, {}),
        )
        audio = render_instrument(inst, role, final_patch, total_n)
        audio = role_fx(audio, role, final_patch, bpm)
        sf.write(stems_dir / f"{role}.wav", audio.T, SR, subtype="PCM_24")
        mix += audio

        patch_report[role] = {
            "reference_id": ref_id,
            "source": source,
            "match_distance": match.get("best_distance"),
            "base_reference_patch": base_patch,
            "producer_controls": plan.get("sound_design", {}).get(role, {}),
            "final_patch": final_patch,
        }

    mix = saturate(mix, float(plan.get("mix", {}).get("master_drive", 0.05)))
    peak = float(np.max(np.abs(mix))) + 1e-9
    target = 10 ** (
        clamp(plan.get("mix", {}).get("target_peak_db", -1), -6, -0.1, -1) / 20
    )
    mix *= target / peak

    master = out_dir / "master.wav"
    sf.write(master, mix.T, SR, subtype="PCM_24")
    (out_dir / "synth_patches.json").write_text(
        json.dumps(patch_report, indent=2),
        encoding="utf-8",
    )
    return master


def main() -> None:
    p = argparse.ArgumentParser(description="Render Musicm8 with inverse-matched native synth patches.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--midi", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--patches", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    master = render_project(args.plan, args.midi, args.references, args.patches, args.out)
    print(f"✅ Musicm8 inverse-matched master: {master}")
    print(f"✅ Audio stems: {args.out / 'audio_stems'}")
    print(f"✅ Final synth patches: {args.out / 'synth_patches.json'}")


if __name__ == "__main__":
    main()
