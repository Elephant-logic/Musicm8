from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_matched_v2 import render_project

SAFE_FM_RATIOS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return default


def nearest_ratio(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        x = 2.0
    return min(SAFE_FM_RATIOS, key=lambda r: abs(r - x))


def sanitize_patch(role: str, patch: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(patch))
    if role == "drums":
        out["drive"] = clamp(out.get("drive", 0.1), 0.0, 0.35, 0.1)
        out["room"] = clamp(out.get("room", 0.1), 0.0, 0.30, 0.1)
        return out

    # The old inverse search was allowed to choose continuous FM ratios and very
    # aggressive modulation. That can match a spectrum while sounding metallic,
    # inharmonic and subjectively out of tune. Keep only musically stable ratios.
    out["fm_ratio"] = nearest_ratio(out.get("fm_ratio", 2.0))
    if role == "bass":
        out["fm_mix"] = clamp(out.get("fm_mix", 0.05), 0.0, 0.16, 0.05)
        out["fm_index"] = clamp(out.get("fm_index", 1.0), 0.0, 2.2, 1.0)
        out["detune"] = clamp(out.get("detune", 0.04), 0.0, 0.10, 0.04)
        out["noise_mix"] = clamp(out.get("noise_mix", 0.01), 0.0, 0.04, 0.01)
        out["drive"] = clamp(out.get("drive", 0.10), 0.0, 0.32, 0.10)
        out["width"] = clamp(out.get("width", 0.03), 0.0, 0.08, 0.03)
        out["cutoff"] = clamp(out.get("cutoff", 1800), 180.0, 5200.0, 1800.0)
    elif role == "chords":
        if str(out.get("wave", "wavetable")).lower() == "fm":
            out["wave"] = "wavetable"
        out["fm_mix"] = clamp(out.get("fm_mix", 0.04), 0.0, 0.10, 0.04)
        out["fm_index"] = clamp(out.get("fm_index", 0.8), 0.0, 1.6, 0.8)
        out["detune"] = clamp(out.get("detune", 0.12), 0.0, 0.18, 0.12)
        out["noise_mix"] = clamp(out.get("noise_mix", 0.01), 0.0, 0.04, 0.01)
        out["drive"] = clamp(out.get("drive", 0.06), 0.0, 0.20, 0.06)
        out["reverb"] = clamp(out.get("reverb", 0.28), 0.0, 0.48, 0.28)
    else:
        if str(out.get("wave", "wavetable")).lower() == "fm":
            out["wave"] = "wavetable"
        out["fm_mix"] = clamp(out.get("fm_mix", 0.05), 0.0, 0.12, 0.05)
        out["fm_index"] = clamp(out.get("fm_index", 0.8), 0.0, 1.8, 0.8)
        out["detune"] = clamp(out.get("detune", 0.06), 0.0, 0.10, 0.06)
        out["noise_mix"] = clamp(out.get("noise_mix", 0.01), 0.0, 0.03, 0.01)
        out["drive"] = clamp(out.get("drive", 0.06), 0.0, 0.18, 0.06)
        out["delay"] = clamp(out.get("delay", 0.14), 0.0, 0.30, 0.14)
        out["reverb"] = clamp(out.get("reverb", 0.20), 0.0, 0.38, 0.20)
    return out


def make_safe_inputs(plan_path: Path, patches_path: Path, out_dir: Path) -> tuple[Path, Path]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    matched = json.loads(patches_path.read_text(encoding="utf-8")) if patches_path.exists() else {"roles": {}}

    for role, row in matched.get("roles", {}).items():
        if isinstance(row, dict) and isinstance(row.get("patch"), dict):
            row["patch"] = sanitize_patch(str(role), row["patch"])
            row["tuning_safe_sanitized"] = True

    # Producer controls are also bounded so they cannot undo the safety pass.
    sd = plan.setdefault("sound_design", {})
    for role in ("bass", "chords", "melody"):
        controls = sd.setdefault(role, {})
        if role == "bass":
            controls["width"] = clamp(controls.get("width", 0.03), 0.0, 0.08, 0.03)
            controls["drive"] = clamp(controls.get("drive", 0.20), 0.0, 0.35, 0.20)
        else:
            controls["detune"] = clamp(controls.get("detune", 0.08), 0.0, 0.16 if role == "chords" else 0.10, 0.08)
            if "reverb" in controls:
                controls["reverb"] = clamp(controls.get("reverb", 0.25), 0.0, 0.50, 0.25)

    # Critical change: only unpitched reference drums may be layered. Previous
    # bass/chord reference clips carried their original harmonic content and could
    # clash with the new key even after root transposition.
    production = plan.setdefault("production", {})
    hybrid = production.setdefault("hybrid", {})
    hybrid["bass"] = 0.0
    hybrid["chords"] = 0.0

    safe_plan = out_dir / "plan_tuning_safe.json"
    safe_patches = out_dir / "matched_patches_tuning_safe.json"
    safe_plan.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    safe_patches.write_text(json.dumps(matched, indent=2), encoding="utf-8")
    return safe_plan, safe_patches


def drum_only_hybridize(out_dir: Path, plan_path: Path, midi_path: Path) -> None:
    try:
        work = out_dir.parents[1]
        references = work / "reference_library" / "library.json"
        bank = work / "reference_instrument_bank.json"
        if not references.exists():
            print("⚠️ Reference drum layer skipped: library not found")
            return
        from hybrid_instruments import hybridize
        report = hybridize(out_dir, plan_path, references, midi_path, bank)
        print("✅ TUNING-SAFE HYBRID LAYER")
        for role, info in report.get("roles", {}).items():
            print(f"   {role}: {info.get('status')} blend={info.get('blend')} events={info.get('reference_events_used')}")
        print("   Tonal reference audio: DISABLED (bass/chords remain MIDI-controlled synths)")
    except Exception as exc:
        print(f"⚠️ Reference drum layer failed safely ({exc}); keeping synth stems")


def main() -> None:
    p = argparse.ArgumentParser(description="Render Musicm8 with strict harmonic/timbre safety before mixing.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--midi", type=Path, required=True)
    p.add_argument("--patches", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    safe_plan, safe_patches = make_safe_inputs(args.plan, args.patches, args.out)
    master = render_project(safe_plan, args.midi, safe_patches, args.out)
    drum_only_hybridize(args.out, safe_plan, args.midi)
    print("✅ TUNING-SAFE SYNTH RENDER:", master)
    print("✅ Continuous/in-harmonic FM ratios constrained")
    print("✅ Pitched reference sample layers disabled")


if __name__ == "__main__":
    main()
