from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def run(cmd: list[str], cwd: Path) -> None:
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_optional(cmd: list[str], cwd: Path, label: str) -> tuple[bool, str | None]:
    try:
        run(cmd, cwd)
        return True, None
    except Exception as exc:
        print(f"⚠️ {label} failed: {exc}")
        return False, str(exc)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 AI producer v9: strict clock + strict key + stable instruments + score/phoneme lead vocals.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--idea", required=True)
    p.add_argument("--bars", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ai-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    # Kept for notebook/backwards compatibility. v9 foundation mode intentionally
    # does not use free inverse spectral patch search in the audible render.
    p.add_argument("--match-iters", type=int, default=0)
    p.add_argument("--match-seconds", type=float, default=3.0)
    p.add_argument("--force-sound-match", action="store_true")
    p.add_argument("--vocal-style", default="expressive contemporary lead vocal")
    p.add_argument("--vocal-language", default="en")
    p.add_argument("--vocal-steps", type=int, default=24)
    p.add_argument("--lyrics-file", type=Path, default=None)
    p.add_argument("--voice-reference", type=Path, default=None, help="Optional clean voice reference you own or have permission to use.")
    p.add_argument("--no-ai", action="store_true")
    p.add_argument("--no-vocals", action="store_true")
    p.add_argument("--skip-sound-match", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    args = p.parse_args()

    root = args.root.resolve()
    repo = args.repo.resolve()
    audio = root / "audio"
    work = root / "work"
    dataset = work / "daw_dataset"
    reference_dir = work / "reference_library"
    library = reference_dir / "library.json"
    old_codec_index = work / "tokens-encodec24" / "index.jsonl"
    project = work / "ai_projects" / "latest"
    plan = project / "plan.json"
    lyrics_json = project / "lyrics.json"
    lyrics_txt = project / "lyrics.txt"
    arrangement = project / "arrangement.mid"
    vocal_score = project / "vocal_score.json"
    vocal_midi = project / "vocal_melody.mid"
    raw_vocal = project / "vocals" / "neural_lead_raw.wav"
    synced_vocal = project / "vocals" / "neural_lead_synced.wav"
    pitch_quality = project / "vocals" / "vocal_quality.json"
    word_quality = project / "vocals" / "vocal_word_quality.json"
    vocal_status_path = project / "vocals" / "vocal_status.json"
    final_status_path = project / "final_status.json"
    matched = project / "matched_patches.json"

    for path in (audio, work, reference_dir, project, raw_vocal.parent):
        path.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the Colab AI producer workflow.")
    if args.voice_reference is not None and not args.voice_reference.exists():
        raise FileNotFoundError(args.voice_reference)

    old_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(repo) + (os.pathsep + old_pythonpath if old_pythonpath else "")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Idea:", args.idea)
    print("Song seed:", args.seed)
    print("Musicm8 v9 FOUNDATION MODE")
    print("  one global MIDI clock")
    print("  no random timing jitter")
    print("  no reference-audio instrument layers")
    print("  no inverse-spectral patches in the audible render")
    print("  SoulX words+phonemes+score lead vocal")

    print("\n=== 1/12 ANALYSE REFERENCE SONGS ===")
    cmd = [sys.executable, "-u", "daw_extract.py", "--audio-dir", audio, "--out", dataset, "--device", "cuda"]
    if args.force_extract:
        cmd.append("--force")
    run(cmd, repo)
    index = dataset / "index.jsonl"
    if not index.exists():
        raise RuntimeError(f"Missing DAW reference index: {index}")

    print("\n=== 2/12 BUILD REFERENCE LIBRARY (STYLE/PRODUCTION ONLY) ===")
    cmd = [sys.executable, "-u", "reference_library.py", "--index", index, "--out", library]
    if old_codec_index.exists():
        cmd += ["--codec-index", old_codec_index]
    run(cmd, repo)

    print("\n=== 3/12 AI PRODUCER BRAIN ===")
    cmd = [sys.executable, "-u", "producer_ai.py", "--idea", args.idea, "--references", library, "--out", plan, "--model", args.ai_model, "--device", "cuda", "--bars", str(args.bars)]
    if args.no_ai:
        cmd.append("--no-ai")
    run(cmd, repo)

    print("\n=== 4/12 GENRE PRODUCTION DIRECTOR ===")
    run([sys.executable, "-u", "style_production.py", "--plan", plan, "--references", library, "--out", plan], repo)

    print("\n=== 5/12 LYRICS ===")
    if args.no_vocals:
        lyrics_json.write_text(json.dumps({"format": "musicm8-lyrics-v2", "title": "Instrumental", "language": args.vocal_language, "idea": args.idea, "hook": "", "source": "disabled", "sections": []}, indent=2), encoding="utf-8")
        lyrics_txt.write_text("[Instrumental]\n", encoding="utf-8")
        print("⏭️ Vocals disabled.")
    else:
        cmd = [sys.executable, "-u", "lyrics_ai.py", "--idea", args.idea, "--plan", plan, "--out", lyrics_json, "--model", args.ai_model, "--device", "cuda", "--language", args.vocal_language]
        if args.no_ai:
            cmd.append("--no-ai")
        if args.lyrics_file is not None and args.lyrics_file.exists() and args.lyrics_file.read_text(encoding="utf-8").strip():
            cmd += ["--lyrics-file", args.lyrics_file]
        run(cmd, repo)
        print("\n📝 EXACT LYRICS REQUESTED\n")
        print(lyrics_txt.read_text(encoding="utf-8", errors="ignore"))

    print("\n=== 6/12 COMPOSE + KEY GUARD + GLOBAL TIMING GRID ===")
    run([sys.executable, "-u", "plan_to_midi_v3.py", "--plan", plan, "--out", project, "--seed", str(args.seed)], repo)
    run([sys.executable, "-u", "midi_tuning_guard.py", "--plan", plan, "--midi", arrangement], repo)
    run([sys.executable, "-u", "timing_grid_guard.py", "--plan", plan, "--midi", arrangement], repo)

    print("\n=== 7/12 STABLE GENRE INSTRUMENTS ===")
    # Keep a compatibility marker so older tooling never reuses stale inverse patches.
    matched.write_text(json.dumps({
        "format": "musicm8-v9-foundation",
        "roles": {},
        "note": "Audible v9 render uses curated stable genre patches. Inverse spectral matching/reference stem layers are disabled until timing and source quality are proven.",
    }, indent=2), encoding="utf-8")
    if args.force_sound_match and not args.skip_sound_match:
        print("ℹ️ --force-sound-match was requested, but v9 foundation mode does not put experimental matched patches into the audible master.")
    run([sys.executable, "-u", "render_stable_v4.py", "--plan", plan, "--midi", arrangement, "--out", project], repo)

    print("\n=== 8/12 CONTROLLED MIX / MASTER ===")
    run([sys.executable, "-u", "mix_polish_v3.py", "--project", project, "--plan", plan, "--out", project / "master.wav"], repo)
    # Reference-master coloration is intentionally disabled in foundation mode.
    instrumental = project / "master_instrumental.wav"
    if (project / "master.wav").exists():
        shutil.copy2(project / "master.wav", instrumental)

    vocal_ok = False
    vocal_error: str | None = None
    final_kind = "instrumental"

    # Remove stale vocal artefacts/status from previous runs so the notebook can never
    # display an old singer as if it belonged to the new song.
    for stale in (raw_vocal, synced_vocal, pitch_quality, word_quality, vocal_status_path):
        if stale.exists():
            stale.unlink()

    if not args.no_vocals:
        print("\n=== 9/12 BUILD WORD-BY-WORD VOCAL SCORE ===")
        run([
            sys.executable, "-u", "vocal_lead_score.py",
            "--plan", plan,
            "--lyrics", lyrics_json,
            "--midi-out", vocal_midi,
            "--score-out", vocal_score,
        ], repo)

        print("\n=== 10/12 SOULX SCORE/PHONEME LEAD SINGER ===")
        cmd = [
            sys.executable, "-u", "soulx_vocals.py",
            "--repo", repo,
            "--root", root,
            "--score", vocal_score,
            "--out", raw_vocal,
            "--svc-steps", str(max(8, min(40, args.vocal_steps))),
        ]
        if args.voice_reference is not None:
            cmd += ["--voice-reference", args.voice_reference]
        vocal_ok, vocal_error = run_optional(cmd, repo, "SoulX lead vocal generation")

        print("\n=== 11/12 VOCAL PITCH + WORD QA ===")
        if vocal_ok and raw_vocal.exists():
            shutil.copy2(raw_vocal, synced_vocal)
            vocal_ok, vocal_error = run_optional([sys.executable, "-u", "vocal_quality_gate.py", "--vocal", synced_vocal, "--score", vocal_score, "--report", pitch_quality], repo, "Vocal pitch quality gate")
            if vocal_ok:
                vocal_ok, vocal_error = run_optional([sys.executable, "-u", "vocal_word_gate.py", "--vocal", synced_vocal, "--lyrics", lyrics_txt, "--report", word_quality, "--model", "openai/whisper-base.en", "--min-recall", "0.30"], repo, "Vocal word intelligibility gate")
        if not vocal_ok:
            print("❌ LEAD VOCAL NOT ACCEPTED")
            print("The requested lyrics are preserved, but the final mix will NOT pretend an instrumental is a successful vocal song.")

        print("\n=== 12/12 VOCAL PRODUCTION + FINAL ===")
        if vocal_ok and synced_vocal.exists():
            shutil.copy2(instrumental, project / "master.wav")
            run([sys.executable, "-u", "mix_vocals.py", "--project", project, "--vocal", synced_vocal, "--plan", plan, "--out", project / "master.wav"], repo)
            final_kind = "song_with_vocals"
        else:
            shutil.copy2(instrumental, project / "master.wav")
            final_kind = "instrumental_only_vocal_failed"
    else:
        final_kind = "instrumental_requested"

    lyrics_source = read_json(lyrics_json).get("source")
    vocal_backend = read_json(vocal_status_path)
    final_status = {
        "format": "musicm8-final-status-v9",
        "final_kind": final_kind,
        "vocal_requested": not args.no_vocals,
        "vocal_ok": bool(vocal_ok),
        "vocal_error": vocal_error,
        "vocal_backend": vocal_backend,
        "pitch_qa": read_json(pitch_quality),
        "word_qa": read_json(word_quality),
        "lyrics": str(lyrics_txt),
        "instrumental": str(instrumental),
        "master": str(project / "master.wav"),
    }
    final_status_path.write_text(json.dumps(final_status, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "format": "musicm8-ai-project-v9",
        "engine": "musicm8-stable-foundation-v4",
        "producer": "ai+genre-production-director",
        "composer": "phrase-aware-v3+tuning-guard+strict-global-grid",
        "mix_engine": "musicm8-mix-polish-v3 foundation mode",
        "vocal_engine": "SoulX-Singer score+phoneme + optional authorized SVC + pitch/word QA" if not args.no_vocals else None,
        "idea": args.idea,
        "seed": args.seed,
        "plan": str(plan),
        "lyrics_json": str(lyrics_json),
        "lyrics_text": str(lyrics_txt),
        "lyrics_source": lyrics_source,
        "user_lyrics_file": str(args.lyrics_file) if args.lyrics_file else None,
        "voice_reference": str(args.voice_reference) if args.voice_reference else None,
        "vocal_score": str(vocal_score) if vocal_score.exists() else None,
        "vocal_melody": str(vocal_midi) if vocal_midi.exists() else None,
        "raw_neural_vocal": str(raw_vocal) if raw_vocal.exists() else None,
        "synced_neural_vocal": str(synced_vocal) if synced_vocal.exists() else None,
        "vocal_status": "ok" if vocal_ok else ("disabled" if args.no_vocals else "rejected_or_failed"),
        "vocal_error": vocal_error,
        "final_kind": final_kind,
        "reference_library": str(library),
        "arrangement_midi": str(arrangement),
        "midi_stems": str(project / "midi_stems"),
        "tuning_guard_report": str(project / "tuning_guard_report.json"),
        "timing_grid_report": str(project / "timing_grid_report.json"),
        "stable_render_report": str(project / "stable_render_report.json"),
        "audio_stems": str(project / "audio_stems"),
        "polished_audio_stems": str(project / "audio_stems_polished"),
        "master_instrumental": str(instrumental),
        "master": str(project / "master.wav"),
        "final_status": str(final_status_path),
    }
    (project / "project.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n============================================================")
    print("✅ MUSICM8 V9 RENDER COMPLETE")
    print("Final kind:", final_kind)
    print("Seed:", args.seed)
    print("Instrumental:", instrumental)
    if final_kind == "song_with_vocals":
        print("✅ FINAL SONG CONTAINS QA-PASSED WORDS:", project / "master.wav")
    elif not args.no_vocals:
        print("❌ NO VOCAL FINAL WAS CREATED — master.wav is the instrumental safety copy")
        print("Check:", final_status_path)
        print("Vocal backend log:", project / "vocals" / "vocal_backend.log")
    else:
        print("Final instrumental:", project / "master.wav")
    print("============================================================")

    if lyrics_txt.exists():
        print("\n================ 📝 LYRICS REQUESTED ================\n")
        print(lyrics_txt.read_text(encoding="utf-8", errors="ignore"))
        print("======================================================\n")


if __name__ == "__main__":
    main()
