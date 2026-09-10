from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def run(cmd: list[str], cwd: Path) -> None:
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 AI producer v3: references -> producer AI -> coherent MIDI -> Synth v2 inverse sound design -> mix/master.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--idea", required=True)
    p.add_argument("--bars", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ai-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--match-iters", type=int, default=48)
    p.add_argument("--match-seconds", type=float, default=3.0)
    p.add_argument("--no-ai", action="store_true")
    p.add_argument("--skip-sound-match", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--force-sound-match", action="store_true")
    args = p.parse_args()

    root = args.root.resolve()
    repo = args.repo.resolve()
    audio = root / "audio"
    work = root / "work"
    dataset = work / "daw_dataset"
    reference_dir = work / "reference_library"
    library = reference_dir / "library.json"
    patch_cache = work / "sound_patch_cache_v2"
    old_codec_index = work / "tokens-encodec24" / "index.jsonl"
    project = work / "ai_projects" / "latest"
    plan = project / "plan.json"
    arrangement = project / "arrangement.mid"
    matched = project / "matched_patches.json"

    for path in (audio, work, reference_dir, patch_cache, project):
        path.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the Colab AI producer workflow.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Idea:", args.idea)
    print("Sound engine: Musicm8 Synth v2")

    print("\n=== 1/6 ANALYSE REFERENCE SONGS ===")
    extract_cmd = [sys.executable, "-u", "daw_extract.py", "--audio-dir", audio, "--out", dataset, "--device", "cuda"]
    if args.force_extract:
        extract_cmd.append("--force")
    run(extract_cmd, repo)
    index = dataset / "index.jsonl"
    if not index.exists():
        raise RuntimeError(f"Missing DAW reference index: {index}")

    print("\n=== 2/6 BUILD MULTIMODAL REFERENCE LIBRARY ===")
    lib_cmd = [sys.executable, "-u", "reference_library.py", "--index", index, "--out", library]
    if old_codec_index.exists():
        lib_cmd += ["--codec-index", old_codec_index]
    run(lib_cmd, repo)

    print("\n=== 3/6 AI PRODUCER BRAIN ===")
    brain_cmd = [sys.executable, "-u", "producer_ai.py", "--idea", args.idea, "--references", library, "--out", plan, "--model", args.ai_model, "--device", "cuda", "--bars", str(args.bars)]
    if args.no_ai:
        brain_cmd.append("--no-ai")
    run(brain_cmd, repo)

    print("\n=== 4/6 COMPOSE EDITABLE MIDI ===")
    run([sys.executable, "-u", "plan_to_midi.py", "--plan", plan, "--out", project, "--seed", str(args.seed)], repo)

    print("\n=== 5/6 SYNTH V2 INVERSE SOUND DESIGN ===")
    if args.skip_sound_match:
        matched.write_text(json.dumps({"format": "musicm8-inverse-synth-v2", "roles": {}, "note": "Sound matching skipped; Synth v2 renderer uses fallback patches."}, indent=2), encoding="utf-8")
        print("⏭️ Sound matching skipped.")
    else:
        match_cmd = [sys.executable, "-u", "sound_matcher_v2.py", "--plan", plan, "--references", library, "--out", matched, "--iterations", str(max(0, args.match_iters)), "--seconds", str(args.match_seconds), "--seed", str(args.seed), "--cache-dir", patch_cache]
        if args.force_sound_match:
            match_cmd.append("--force")
        run(match_cmd, repo)

    print("\n=== 6/6 SYNTH V2 + FX + SIDECHAIN + MASTER ===")
    run([sys.executable, "-u", "render_matched_v2.py", "--plan", plan, "--midi", arrangement, "--patches", matched, "--out", project], repo)

    payload = {
        "format": "musicm8-ai-project-v3",
        "engine": "musicm8-synth-v2",
        "idea": args.idea,
        "plan": str(plan),
        "reference_library": str(library),
        "arrangement_midi": str(arrangement),
        "midi_stems": str(project / "midi_stems"),
        "matched_patches": str(matched),
        "sound_match_previews": str(project / "sound_matches"),
        "audio_stems": str(project / "audio_stems"),
        "synth_patches": str(project / "synth_patches.json"),
        "master": str(project / "master.wav"),
        "legacy_codec_tokens": str(old_codec_index) if old_codec_index.exists() else None,
        "sound_design": {"learned_harmonic_wavetables": True, "fm_layer": True, "noise_texture_layer": True, "per_drum_voice_analysis": True, "multi_resolution_match": [512, 2048, 8192], "eq": "3-band", "compressor": True, "kick_sidechain": True},
        "note": "Musicm8 v3 keeps the producer AI and reference library, but the sound engine is Synth v2. Pitched references are converted into reusable harmonic/wavetable fingerprints and searched with FM, filtering, envelopes, modulation, EQ, compression and FX. Drum references are analysed separately as kick/snare/hat voices. The final mix uses DAW-style stem balance and kick-triggered sidechain. Existing EnCodec tokens remain linked for a future neural residual layer."
    }
    (project / "project.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n✅ MUSICM8 AI PRODUCER V3 COMPLETE")
    print("Plan:", plan)
    print("MIDI:", arrangement)
    print("MIDI stems:", project / "midi_stems")
    print("Matched v2 patches:", matched)
    print("A/B sound matches:", project / "sound_matches")
    print("Audio stems:", project / "audio_stems")
    print("Synth patches:", project / "synth_patches.json")
    print("Master:", project / "master.wav")


if __name__ == "__main__":
    main()
