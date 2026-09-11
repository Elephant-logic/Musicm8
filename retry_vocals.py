from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, *, show_failure_log: Path | None = None) -> None:
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if proc.returncode != 0:
        print("\n" + "=" * 70)
        print("❌ COMMAND FAILED — REAL BACKEND DIAGNOSTICS")
        print("=" * 70)
        if show_failure_log and show_failure_log.exists():
            lines = show_failure_log.read_text(encoding="utf-8", errors="ignore").splitlines()
            print(f"Backend log: {show_failure_log}")
            print("\n--- LAST 180 LOG LINES ---")
            print("\n".join(lines[-180:]))
            print("--- END LOG ---")
        else:
            print("No backend log was found.")
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}. "
            "The real backend error is printed above; do not rely on the outer CalledProcessError."
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Retry only the Musicm8 vocal stage using the existing instrumental, lyrics and melody.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--style", default="expressive contemporary lead vocal, intimate verses, emotional hook, clear lyrics")
    p.add_argument("--language", default="en")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse-raw", action="store_true", help="Do not regenerate ACE-Step; resync the existing raw vocal only.")
    args = p.parse_args()

    root = args.root.resolve()
    repo = args.repo.resolve()
    project = root / "work" / "ai_projects" / "latest"
    lyrics_txt = project / "lyrics.txt"
    lyrics_json = project / "lyrics.json"
    plan = project / "plan.json"
    melody_midi = project / "midi_stems" / "melody.mid"
    vocal_score = project / "vocal_score.json"
    raw_vocal = project / "vocals" / "neural_lead_raw.wav"
    synced_vocal = project / "vocals" / "neural_lead_synced.wav"
    instrumental = project / "master_instrumental.wav"
    master = project / "master.wav"
    backend_log = project / "vocals" / "vocal_backend.log"

    for path in (project, lyrics_txt, lyrics_json, plan, melody_midi):
        if not path.exists():
            raise FileNotFoundError(path)

    backing = instrumental if instrumental.exists() else master
    if not backing.exists():
        raise FileNotFoundError(f"Missing instrumental/master: {backing}")

    old_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(repo) + (os.pathsep + old_pythonpath if old_pythonpath else "")

    raw_vocal.parent.mkdir(parents=True, exist_ok=True)
    print("🎤 MUSICM8 VOCAL SYNC RETRY")
    print("Backing:", backing)
    print("Lyrics :", lyrics_txt)
    print("Melody :", melody_midi)
    print("Plan   :", plan)
    print("Log    :", backend_log)
    print("T4 compatibility hook:", repo / "sitecustomize.py")

    # Rebuild the score using the current v2 alignment so every note knows which
    # lyric line it belongs to. This is fast and fixes old projects automatically.
    run([
        sys.executable, "-u", "vocal_score.py",
        "--plan", plan,
        "--lyrics", lyrics_json,
        "--melody-midi", melody_midi,
        "--out", vocal_score,
    ], repo)

    print("\n📝 LYRICS")
    print(lyrics_txt.read_text(encoding="utf-8", errors="ignore"))

    if not args.reuse_raw:
        run([
            sys.executable, "-u", "neural_vocals.py",
            "--repo", repo,
            "--root", root,
            "--backing", backing,
            "--lyrics", lyrics_txt,
            "--plan", plan,
            "--out", raw_vocal,
            "--style", args.style,
            "--language", args.language,
            "--seed", str(args.seed),
            "--steps", str(args.steps),
        ], repo, show_failure_log=backend_log)
    elif not raw_vocal.exists():
        raise FileNotFoundError(f"--reuse-raw requested but raw vocal is missing: {raw_vocal}")
    else:
        print("♻️ Reusing existing raw neural vocal; ACE-Step will NOT run again.")

    # The neural singer is creative and can drift against the arrangement. The score
    # is authoritative: trim model padding, map phrases in lyric order, preserve pitch
    # while time-fitting each phrase to its written note window, and gently correct
    # each phrase toward the melody's key/average pitch.
    run([
        sys.executable, "-u", "vocal_sync.py",
        "--vocal", raw_vocal,
        "--score", vocal_score,
        "--out", synced_vocal,
    ], repo)

    if backing != master:
        shutil.copy2(backing, master)

    run([
        sys.executable, "-u", "mix_vocals.py",
        "--project", project,
        "--vocal", synced_vocal,
        "--plan", plan,
        "--out", master,
    ], repo)

    status_path = project / "vocals" / "vocal_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    print("\n✅ VOCAL SYNC COMPLETE")
    print("Neural method:", status.get("method", "ACE-Step"))
    print("Raw vocal:", raw_vocal)
    print("Score-locked vocal:", synced_vocal)
    print("Vocal mix:", project / "vocals" / "vocal_mix.wav")
    print("Final song:", master)


if __name__ == "__main__":
    main()
