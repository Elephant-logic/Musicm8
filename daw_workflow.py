from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def run(cmd: list[str], cwd: Path) -> None:
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def checkpoint_step(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False).get("step", 0))
    except Exception:
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 hybrid DAW pipeline: stems -> MIDI -> symbolic model -> editable project.")
    p.add_argument("--root", type=Path, required=True, help="Musicm8 Drive folder")
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--bars", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-extract", action="store_true")
    args = p.parse_args()

    repo = args.repo.resolve()
    root = args.root.resolve()
    audio = root / "audio"
    work = root / "work"
    dataset = work / "daw_dataset"
    model_dir = work / "daw_symbolic"
    checkpoint = model_dir / "latest.pt"
    project = work / "daw_projects" / "latest"
    midi = project / "arrangement.mid"
    preview = project / "preview.wav"

    audio.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    project.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the Colab DAW workflow. Switch Runtime to GPU and rerun.")
    print("GPU:", torch.cuda.get_device_name(0))

    # Step 1: source separation + musical analysis + editable MIDI training data.
    extract_cmd = [
        sys.executable, "-u", "daw_extract.py",
        "--audio-dir", audio,
        "--out", dataset,
        "--device", "cuda",
    ]
    if args.force_extract:
        extract_cmd.append("--force")
    print("\n=== 1/4 STEMS + MIDI ANALYSIS ===")
    run(extract_cmd, repo)

    index = dataset / "index.jsonl"
    if not index.exists():
        raise RuntimeError(f"DAW dataset index missing: {index}")
    rows = [json.loads(x) for x in index.read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"Prepared tracks: {len(rows)}")

    # Step 2: train a much smaller symbolic arranger instead of predicting raw codec audio.
    prior = checkpoint_step(checkpoint)
    print("\n=== 2/4 SYMBOLIC ARRANGER ===")
    print(f"Checkpoint step: {prior} / target {args.steps}")
    if prior < args.steps:
        cmd = [
            sys.executable, "-u", "train_symbolic.py",
            "--index", index,
            "--out", model_dir,
            "--steps", str(args.steps),
            "--batch-size", "4",
            "--max-seq-len", "768",
            "--save-every", "250",
            "--device", "cuda",
        ]
        if checkpoint.exists():
            cmd += ["--resume", checkpoint]
        run(cmd, repo)
    else:
        print("✅ Symbolic checkpoint already trained; skipping.")

    # Step 3: generate a valid, editable multitrack MIDI arrangement.
    print("\n=== 3/4 GENERATE EDITABLE MIDI ===")
    shutil.rmtree(project / "midi_stems", ignore_errors=True)
    run([
        sys.executable, "-u", "generate_symbolic.py",
        "--checkpoint", checkpoint,
        "--out", midi,
        "--bars", str(args.bars),
        "--temperature", "0.85",
        "--top-k", "20",
        "--seed", str(args.seed),
        "--device", "cuda",
    ], repo)

    # Step 4: SoundFont is only a preview. The MIDI stems are meant for real VSTs in a DAW.
    print("\n=== 4/4 DAW BUNDLE + PREVIEW ===")
    run([
        sys.executable, "-u", "render_daw.py",
        "--midi", midi,
        "--out", preview,
    ], repo)

    print("\n✅ MUSICM8 DAW WORKFLOW COMPLETE")
    print("Training stems/data:", dataset)
    print("Symbolic model:", checkpoint)
    print("Arrangement MIDI:", midi)
    print("Individual MIDI stems:", project / "midi_stems")
    print("DAW manifest:", project / "project.json")
    if preview.exists():
        print("SoundFont preview:", preview)
    print("\nImport the MIDI stems into Ableton/FL Studio/Logic/Reaper and assign your VSTs. "
          "The model now learns arrangement/note structure; instruments and mixing stay editable like a DAW.")


if __name__ == "__main__":
    main()
