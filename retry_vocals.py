from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Retry only the Musicm8 vocal stage using the already-generated instrumental and lyrics.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--style", default="expressive contemporary lead vocal, intimate verses, emotional hook, clear lyrics")
    p.add_argument("--language", default="en")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = args.root.resolve()
    repo = args.repo.resolve()
    project = root / "work" / "ai_projects" / "latest"
    lyrics = project / "lyrics.txt"
    plan = project / "plan.json"
    raw_vocal = project / "vocals" / "neural_lead_raw.wav"
    instrumental = project / "master_instrumental.wav"
    master = project / "master.wav"

    if not project.exists():
        raise FileNotFoundError(f"Missing Musicm8 project: {project}")
    if not lyrics.exists():
        raise FileNotFoundError(f"Missing lyrics: {lyrics}")
    if not plan.exists():
        raise FileNotFoundError(f"Missing plan: {plan}")

    # Prefer the preserved clean instrumental if a previous vocal mix existed.
    backing = instrumental if instrumental.exists() else master
    if not backing.exists():
        raise FileNotFoundError(f"Missing instrumental/master: {backing}")

    raw_vocal.parent.mkdir(parents=True, exist_ok=True)
    print("🎤 MUSICM8 VOCAL-ONLY RETRY")
    print("Backing:", backing)
    print("Lyrics :", lyrics)
    print("Plan   :", plan)

    run([
        sys.executable, "-u", "neural_vocals.py",
        "--repo", repo,
        "--root", root,
        "--backing", backing,
        "--lyrics", lyrics,
        "--plan", plan,
        "--out", raw_vocal,
        "--style", args.style,
        "--language", args.language,
        "--seed", str(args.seed),
        "--steps", str(args.steps),
    ], repo)

    # Restore the clean instrumental as master before adding the newly-created vocal.
    if backing != master:
        shutil.copy2(backing, master)

    run([
        sys.executable, "-u", "mix_vocals.py",
        "--project", project,
        "--vocal", raw_vocal,
        "--plan", plan,
        "--out", master,
    ], repo)

    status_path = project / "vocals" / "vocal_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    print("\n✅ VOCAL RETRY COMPLETE")
    print("Method:", status.get("method", "ACE-Step"))
    print("Raw vocal:", raw_vocal)
    print("Vocal mix:", project / "vocals" / "vocal_mix.wav")
    print("Final song:", master)


if __name__ == "__main__":
    main()
