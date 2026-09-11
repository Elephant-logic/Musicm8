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
        raise RuntimeError(f"Command failed with exit code {proc.returncode}. The real backend error is printed above.")


def main() -> None:
    p = argparse.ArgumentParser(description="Retry only Musicm8 v8 score-controlled SoulX vocals on the existing instrumental.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--style", default="expressive contemporary lead vocal")
    p.add_argument("--language", default="en")
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--voice-reference", type=Path, default=None)
    p.add_argument("--reuse-raw", action="store_true", help="Reuse the existing SoulX raw vocal and rerun only QA/mixing.")
    args = p.parse_args()

    root = args.root.resolve()
    repo = args.repo.resolve()
    project = root / "work" / "ai_projects" / "latest"
    lyrics_txt = project / "lyrics.txt"
    lyrics_json = project / "lyrics.json"
    plan = project / "plan.json"
    vocal_score = project / "vocal_score.json"
    vocal_midi = project / "vocal_melody.mid"
    raw_vocal = project / "vocals" / "neural_lead_raw.wav"
    synced_vocal = project / "vocals" / "neural_lead_synced.wav"
    pitch_quality = project / "vocals" / "vocal_quality.json"
    word_quality = project / "vocals" / "vocal_word_quality.json"
    instrumental = project / "master_instrumental.wav"
    master = project / "master.wav"
    backend_log = project / "vocals" / "vocal_backend.log"

    for path in (project, lyrics_txt, lyrics_json, plan):
        if not path.exists():
            raise FileNotFoundError(path)
    backing = instrumental if instrumental.exists() else master
    if not backing.exists():
        raise FileNotFoundError(f"Missing instrumental/master: {backing}")
    if args.voice_reference is not None and not args.voice_reference.exists():
        raise FileNotFoundError(args.voice_reference)

    old_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(repo) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    raw_vocal.parent.mkdir(parents=True, exist_ok=True)

    print("🎤 MUSICM8 V8 VOCAL-ONLY RETRY")
    print("Backing:", backing)
    print("Lyrics :", lyrics_txt)
    print("Plan   :", plan)
    print("Voice reference:", args.voice_reference or "default SoulX prompt singer")
    print("\n================ 📝 LYRICS USED ================\n")
    print(lyrics_txt.read_text(encoding="utf-8", errors="ignore"))
    print("=================================================\n")

    run([
        sys.executable, "-u", "vocal_lead_score.py",
        "--plan", plan,
        "--lyrics", lyrics_json,
        "--midi-out", vocal_midi,
        "--score-out", vocal_score,
    ], repo)

    if not args.reuse_raw:
        for stale in (raw_vocal, synced_vocal):
            if stale.exists():
                stale.unlink()
        cmd = [
            sys.executable, "-u", "soulx_vocals.py",
            "--repo", repo,
            "--root", root,
            "--score", vocal_score,
            "--out", raw_vocal,
            "--svc-steps", str(max(8, min(40, args.steps))),
        ]
        if args.voice_reference is not None:
            cmd += ["--voice-reference", args.voice_reference]
        run(cmd, repo, show_failure_log=backend_log)
    elif not raw_vocal.exists():
        raise FileNotFoundError(f"--reuse-raw requested but raw SoulX vocal is missing: {raw_vocal}")
    else:
        print("♻️ Reusing existing SoulX raw vocal; no singer generation will run.")

    shutil.copy2(raw_vocal, synced_vocal)

    try:
        run([sys.executable, "-u", "vocal_quality_gate.py", "--vocal", synced_vocal, "--score", vocal_score, "--report", pitch_quality], repo)
        run([sys.executable, "-u", "vocal_word_gate.py", "--vocal", synced_vocal, "--lyrics", lyrics_txt, "--report", word_quality], repo)
    except Exception:
        if instrumental.exists():
            shutil.copy2(instrumental, master)
        print("❌ Vocal failed pitch/word QA. The instrumental master has been restored.")
        raise

    if instrumental.exists():
        shutil.copy2(instrumental, master)
    run([sys.executable, "-u", "mix_vocals.py", "--project", project, "--vocal", synced_vocal, "--plan", plan, "--out", master], repo)

    status_path = project / "vocals" / "vocal_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    print("\n✅ SCORE-CONTROLLED VOCAL COMPLETE")
    print("Method:", status.get("method", "SoulX-Singer"))
    print("Vocal melody:", vocal_midi)
    print("Raw/score vocal:", raw_vocal)
    print("Pitch QA:", pitch_quality)
    print("Word QA:", word_quality)
    print("Final song:", master)


if __name__ == "__main__":
    main()
