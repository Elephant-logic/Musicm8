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
    p = argparse.ArgumentParser(description="Musicm8 AI producer: references -> AI plan -> MIDI -> own synth/FX.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--idea", required=True)
    p.add_argument("--bars", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ai-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--no-ai", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    args = p.parse_args()

    root = args.root.resolve(); repo = args.repo.resolve(); audio = root / "audio"; work = root / "work"
    dataset = work / "daw_dataset"; reference_dir = work / "reference_library"; library = reference_dir / "library.json"
    old_codec_index = work / "tokens-encodec24" / "index.jsonl"
    project = work / "ai_projects" / "latest"; plan = project / "plan.json"; arrangement = project / "arrangement.mid"
    audio.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True); reference_dir.mkdir(parents=True, exist_ok=True); project.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the Colab AI producer workflow.")
    print("GPU:", torch.cuda.get_device_name(0)); print("Idea:", args.idea)

    print("\n=== 1/5 ANALYSE REFERENCE SONGS ===")
    extract_cmd = [sys.executable, "-u", "daw_extract.py", "--audio-dir", audio, "--out", dataset, "--device", "cuda"]
    if args.force_extract: extract_cmd.append("--force")
    run(extract_cmd, repo)
    index = dataset / "index.jsonl"
    if not index.exists(): raise RuntimeError(f"Missing DAW reference index: {index}")

    print("\n=== 2/5 BUILD MULTIMODAL REFERENCE LIBRARY ===")
    lib_cmd = [sys.executable, "-u", "reference_library.py", "--index", index, "--out", library]
    if old_codec_index.exists(): lib_cmd += ["--codec-index", old_codec_index]
    run(lib_cmd, repo)

    print("\n=== 3/5 AI PRODUCER BRAIN ===")
    brain_cmd = [sys.executable, "-u", "producer_ai.py", "--idea", args.idea, "--references", library, "--out", plan, "--model", args.ai_model, "--device", "cuda", "--bars", str(args.bars)]
    if args.no_ai: brain_cmd.append("--no-ai")
    run(brain_cmd, repo)

    print("\n=== 4/5 COMPOSE EDITABLE MIDI ===")
    run([sys.executable, "-u", "plan_to_midi.py", "--plan", plan, "--out", project, "--seed", str(args.seed)], repo)

    print("\n=== 5/5 MUSICM8 SYNTH + FX ===")
    run([sys.executable, "-u", "musicm8_synth.py", "--plan", plan, "--midi", arrangement, "--references", library, "--out", project], repo)

    payload = {
        "format": "musicm8-ai-project-v1", "idea": args.idea, "plan": str(plan), "reference_library": str(library),
        "arrangement_midi": str(arrangement), "midi_stems": str(project / "midi_stems"), "audio_stems": str(project / "audio_stems"),
        "synth_patches": str(project / "synth_patches.json"), "master": str(project / "master.wav"),
        "legacy_codec_tokens": str(old_codec_index) if old_codec_index.exists() else None,
        "note": "The local producer brain uses the reference library to plan a new song. MIDI stays editable. Musicm8's internal synth derives patches from measured stem spectra and the AI steers sound-design and FX parameters. Existing EnCodec tokens stay linked for future neural resynthesis."
    }
    (project / "project.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n✅ MUSICM8 AI PRODUCER COMPLETE")
    print("Plan:", plan); print("MIDI:", arrangement); print("MIDI stems:", project / "midi_stems"); print("Audio stems:", project / "audio_stems"); print("Synth patches:", project / "synth_patches.json"); print("Master:", project / "master.wav")


if __name__ == "__main__": main()
