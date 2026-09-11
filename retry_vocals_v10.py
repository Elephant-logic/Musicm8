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
        if show_failure_log and show_failure_log.exists():
            print("\n--- BACKEND LOG TAIL ---")
            print("\n".join(show_failure_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-180:]))
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def main() -> None:
    p = argparse.ArgumentParser(description="Retry only v10 vocals against the existing learned AI instrumental.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--voice-reference", type=Path, default=None)
    p.add_argument("--reuse-raw", action="store_true")
    args = p.parse_args()

    root = args.root.resolve()
    repo = args.repo.resolve()
    project = root / "work" / "ai_projects" / "latest"
    plan = project / "plan.json"
    lyrics_json = project / "lyrics.json"
    lyrics_txt = project / "lyrics.txt"
    instrumental = project / "master_instrumental.wav"
    master = project / "master.wav"
    harmony_midi = project / "midi_stems" / "chords.mid"
    melody_midi = project / "midi_stems" / "melody.mid"
    vocal_score = project / "vocal_score.json"
    vocal_midi = project / "vocal_melody.mid"
    raw_vocal = project / "vocals" / "neural_lead_raw.wav"
    synced_vocal = project / "vocals" / "neural_lead_synced.wav"
    pitch_quality = project / "vocals" / "vocal_quality.json"
    word_quality = project / "vocals" / "vocal_word_quality.json"
    backend_log = project / "vocals" / "vocal_backend.log"

    for path in (plan, lyrics_json, lyrics_txt, instrumental):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.voice_reference is not None and not args.voice_reference.exists():
        raise FileNotFoundError(args.voice_reference)

    old_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(repo) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    raw_vocal.parent.mkdir(parents=True, exist_ok=True)

    print("🎤 MUSICM8 V10 VOCAL-ONLY RETRY")
    print("Backing:", instrumental)
    print("Harmony MIDI:", harmony_midi if harmony_midi.exists() else "song-key fallback")
    print("AI lead MIDI:", melody_midi if melody_midi.exists() else "contour fallback")
    print("\n================ 📝 LYRICS USED ================\n")
    print(lyrics_txt.read_text(encoding="utf-8", errors="ignore"))
    print("=================================================\n")

    score_cmd = [
        sys.executable, "-u", "vocal_lead_score_ai.py",
        "--plan", plan,
        "--lyrics", lyrics_json,
        "--midi-out", vocal_midi,
        "--score-out", vocal_score,
    ]
    if harmony_midi.exists():
        score_cmd += ["--harmony-midi", harmony_midi]
    if melody_midi.exists():
        score_cmd += ["--melody-midi", melody_midi]
    run(score_cmd, repo)

    if not args.reuse_raw:
        for stale in (raw_vocal, synced_vocal, pitch_quality, word_quality):
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
        raise FileNotFoundError(f"--reuse-raw requested but raw vocal is missing: {raw_vocal}")
    else:
        print("♻️ Reusing existing SoulX raw vocal")

    shutil.copy2(raw_vocal, synced_vocal)
    try:
        run([sys.executable, "-u", "vocal_quality_gate.py", "--vocal", synced_vocal, "--score", vocal_score, "--report", pitch_quality], repo)
        run([sys.executable, "-u", "vocal_word_gate.py", "--vocal", synced_vocal, "--lyrics", lyrics_txt, "--report", word_quality, "--model", "openai/whisper-base.en", "--min-recall", "0.30"], repo)
    except Exception:
        shutil.copy2(instrumental, master)
        print("❌ Vocal failed pitch/word QA. Instrumental restored.")
        raise

    shutil.copy2(instrumental, master)
    run([sys.executable, "-u", "mix_vocals.py", "--project", project, "--vocal", synced_vocal, "--plan", plan, "--out", master], repo)

    final_status = project / "final_status.json"
    status = json.loads(final_status.read_text(encoding="utf-8")) if final_status.exists() else {}
    status.update({"final_kind": "song_with_vocals", "vocal_ok": True, "vocal_error": None})
    final_status.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n✅ V10 VOCAL COMPLETE")
    print("Score follows learned AI harmony/lead:", vocal_score)
    print("Final song:", master)


if __name__ == "__main__":
    main()
