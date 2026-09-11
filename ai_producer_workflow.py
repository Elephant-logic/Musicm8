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
        print("Project music, lyrics and intermediate files are still saved.")
        return False, str(exc)


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 AI producer v8: strict-key music + score/phoneme-conditioned SoulX lead vocals.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--idea", required=True)
    p.add_argument("--bars", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ai-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--match-iters", type=int, default=48)
    p.add_argument("--match-seconds", type=float, default=3.0)
    p.add_argument("--vocal-style", default="expressive contemporary lead vocal")
    p.add_argument("--vocal-language", default="en")
    p.add_argument("--vocal-steps", type=int, default=32)
    p.add_argument("--lyrics-file", type=Path, default=None)
    p.add_argument("--voice-reference", type=Path, default=None, help="Optional clean authorized voice reference for SoulX-Singer-SVC.")
    p.add_argument("--no-ai", action="store_true")
    p.add_argument("--no-vocals", action="store_true")
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
    lyrics_json = project / "lyrics.json"
    lyrics_txt = project / "lyrics.txt"
    arrangement = project / "arrangement.mid"
    matched = project / "matched_patches.json"
    vocal_score = project / "vocal_score.json"
    vocal_midi = project / "vocal_melody.mid"
    raw_vocal = project / "vocals" / "neural_lead_raw.wav"
    synced_vocal = project / "vocals" / "neural_lead_synced.wav"
    pitch_quality = project / "vocals" / "vocal_quality.json"
    word_quality = project / "vocals" / "vocal_word_quality.json"

    for path in (audio, work, reference_dir, patch_cache, project, raw_vocal.parent):
        path.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the Colab AI producer workflow.")

    old_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(repo) + (os.pathsep + old_pythonpath if old_pythonpath else "")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Idea:", args.idea)
    print("Song seed:", args.seed)
    print("Musicm8 v8: strict tuning + tuning-safe sound design + score/phoneme-controlled SoulX vocals")

    print("\n=== 1/13 ANALYSE REFERENCE SONGS ===")
    cmd = [sys.executable, "-u", "daw_extract.py", "--audio-dir", audio, "--out", dataset, "--device", "cuda"]
    if args.force_extract:
        cmd.append("--force")
    run(cmd, repo)
    index = dataset / "index.jsonl"
    if not index.exists():
        raise RuntimeError(f"Missing DAW reference index: {index}")

    print("\n=== 2/13 BUILD MULTIMODAL REFERENCE LIBRARY ===")
    cmd = [sys.executable, "-u", "reference_library.py", "--index", index, "--out", library]
    if old_codec_index.exists():
        cmd += ["--codec-index", old_codec_index]
    run(cmd, repo)

    print("\n=== 3/13 AI PRODUCER BRAIN ===")
    cmd = [sys.executable, "-u", "producer_ai.py", "--idea", args.idea, "--references", library, "--out", plan, "--model", args.ai_model, "--device", "cuda", "--bars", str(args.bars)]
    if args.no_ai:
        cmd.append("--no-ai")
    run(cmd, repo)

    print("\n=== 4/13 GENRE PRODUCTION DIRECTOR ===")
    run([sys.executable, "-u", "style_production.py", "--plan", plan, "--references", library, "--out", plan], repo)

    print("\n=== 5/13 LYRICS ===")
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
        print("\n📝 EXACT LYRICS THAT WILL BE SUNG\n")
        print(lyrics_txt.read_text(encoding="utf-8", errors="ignore"))

    print("\n=== 6/13 COMPOSE + STRICT TUNING GUARD ===")
    run([sys.executable, "-u", "plan_to_midi_v3.py", "--plan", plan, "--out", project, "--seed", str(args.seed)], repo)
    run([sys.executable, "-u", "midi_tuning_guard.py", "--plan", plan, "--midi", arrangement], repo)

    print("\n=== 7/13 REFERENCE-INSPIRED SOUND DESIGN ===")
    if args.skip_sound_match:
        matched.write_text(json.dumps({"format": "musicm8-inverse-synth-v2", "roles": {}, "note": "Sound matching skipped; renderer uses tuning-safe fallback patches."}, indent=2), encoding="utf-8")
        print("⏭️ Sound matching skipped.")
    else:
        cmd = [sys.executable, "-u", "sound_matcher_v2.py", "--plan", plan, "--references", library, "--out", matched, "--iterations", str(max(0, args.match_iters)), "--seconds", str(args.match_seconds), "--seed", str(args.seed), "--cache-dir", patch_cache]
        if args.force_sound_match:
            cmd.append("--force")
        run(cmd, repo)

    print("\n=== 8/13 TUNING-SAFE SYNTH + REFERENCE DRUMS + FX ===")
    run([sys.executable, "-u", "render_safe_v3.py", "--plan", plan, "--midi", arrangement, "--patches", matched, "--out", project], repo)

    print("\n=== 9/13 GENRE-AWARE MIX + MASTER ===")
    run([sys.executable, "-u", "mix_polish_v3.py", "--project", project, "--plan", plan, "--out", project / "master.wav"], repo)
    if (repo / "reference_master.py").exists():
        run([sys.executable, "-u", "reference_master.py", "--master", project / "master.wav", "--plan", plan, "--references", library, "--out", project / "master.wav"], repo)
    instrumental = project / "master_instrumental.wav"
    if (project / "master.wav").exists():
        shutil.copy2(project / "master.wav", instrumental)

    vocal_ok = False
    vocal_error = None
    if not args.no_vocals:
        print("\n=== 10/13 BUILD DICTION-FIRST VOCAL SCORE ===")
        run([
            sys.executable, "-u", "vocal_lead_score.py",
            "--plan", plan,
            "--lyrics", lyrics_json,
            "--midi-out", vocal_midi,
            "--score-out", vocal_score,
        ], repo)

        print("\n=== 11/13 SOULX SCORE/PHONEME-CONDITIONED LEAD SINGER ===")
        for stale in (raw_vocal, synced_vocal):
            if stale.exists():
                stale.unlink()
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

        print("\n=== 12/13 PITCH + WORD INTELLIGIBILITY QA ===")
        if vocal_ok and raw_vocal.exists():
            shutil.copy2(raw_vocal, synced_vocal)
            vocal_ok, vocal_error = run_optional([sys.executable, "-u", "vocal_quality_gate.py", "--vocal", synced_vocal, "--score", vocal_score, "--report", pitch_quality], repo, "Vocal pitch quality gate")
            if vocal_ok:
                vocal_ok, vocal_error = run_optional([sys.executable, "-u", "vocal_word_gate.py", "--vocal", synced_vocal, "--lyrics", lyrics_txt, "--report", word_quality], repo, "Vocal word intelligibility gate")
            if not vocal_ok:
                print("❌ Vocal rejected. Musicm8 will keep the clean instrumental rather than mix an off-key or unintelligible lead.")
        else:
            print("⏭️ No SoulX vocal available for QA.")

        print("\n=== 13/13 GENRE VOCAL PROCESSING + FINAL MIX ===")
        if vocal_ok and synced_vocal.exists():
            if instrumental.exists():
                shutil.copy2(instrumental, project / "master.wav")
            run([sys.executable, "-u", "mix_vocals.py", "--project", project, "--vocal", synced_vocal, "--plan", plan, "--out", project / "master.wav"], repo)
        else:
            if instrumental.exists():
                shutil.copy2(instrumental, project / "master.wav")
            print("⏭️ Final master is instrumental because the lead vocal did not pass QA.")

    lyrics_source = None
    if lyrics_json.exists():
        try:
            lyrics_source = json.loads(lyrics_json.read_text(encoding="utf-8")).get("source")
        except Exception:
            pass

    payload = {
        "format": "musicm8-ai-project-v8",
        "engine": "musicm8-tuning-safe-synth-v3",
        "producer": "ai+genre-production-director",
        "composer": "phrase-aware-genre-v3+tuning-guard-v1",
        "mix_engine": "musicm8-mix-polish-v3+reference-master",
        "vocal_engine": "SoulX-Singer score+phoneme control + optional SoulX-SVC clone + pitch/word QA" if not args.no_vocals else None,
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
        "vocal_pitch_quality": str(pitch_quality) if pitch_quality.exists() else None,
        "vocal_word_quality": str(word_quality) if word_quality.exists() else None,
        "vocal_status": "ok" if vocal_ok else ("disabled" if args.no_vocals else "rejected_or_failed"),
        "vocal_error": vocal_error,
        "reference_library": str(library),
        "arrangement_midi": str(arrangement),
        "midi_stems": str(project / "midi_stems"),
        "tuning_guard_report": str(project / "tuning_guard_report.json"),
        "matched_patches": str(matched),
        "safe_matched_patches": str(project / "matched_patches_tuning_safe.json"),
        "audio_stems": str(project / "audio_stems"),
        "polished_audio_stems": str(project / "audio_stems_polished"),
        "composition_report": str(project / "composition_report.json"),
        "mix_report": str(project / "mix_report.json"),
        "master_instrumental": str(instrumental),
        "master": str(project / "master.wav"),
        "legacy_codec_tokens": str(old_codec_index) if old_codec_index.exists() else None,
    }
    (project / "project.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n✅ MUSICM8 AI PRODUCER V8 COMPLETE")
    print("Seed:", args.seed)
    print("Lyrics source:", lyrics_source)
    print("Vocal status:", payload["vocal_status"])
    print("Instrumental:", instrumental)
    print("Final:", project / "master.wav")
    if lyrics_txt.exists():
        print("\n================ 📝 LYRICS USED ================\n")
        print(lyrics_txt.read_text(encoding="utf-8", errors="ignore"))
        print("=================================================\n")


if __name__ == "__main__":
    main()
