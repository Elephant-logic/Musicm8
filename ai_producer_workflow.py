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


def run_optional(cmd: list[str], cwd: Path, label: str) -> tuple[bool, str | None]:
    try:
        run(cmd, cwd)
        return True, None
    except Exception as exc:
        print(f"⚠️ {label} failed: {exc}")
        print("Instrumental, lyrics and vocal score are still saved. You can rerun later without losing the project.")
        return False, str(exc)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Musicm8 AI producer v4: references -> producer AI -> original lyrics -> hierarchical MIDI -> "
            "inverse sound design -> polished instrumental -> aligned vocal score -> ACE-Step neural vocals -> final mix."
        )
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--idea", required=True)
    p.add_argument("--bars", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ai-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--match-iters", type=int, default=48)
    p.add_argument("--match-seconds", type=float, default=3.0)
    p.add_argument("--vocal-style", default="expressive contemporary lead vocal, intimate verses, stronger hook, clear words")
    p.add_argument("--vocal-language", default="en")
    p.add_argument("--vocal-steps", type=int, default=40)
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
    vocal_score = project / "vocal_score.json"
    arrangement = project / "arrangement.mid"
    matched = project / "matched_patches.json"
    raw_vocal = project / "vocals" / "neural_lead_raw.wav"

    for path in (audio, work, reference_dir, patch_cache, project, raw_vocal.parent):
        path.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the Colab AI producer workflow.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Idea:", args.idea)
    print("Musicm8: Producer AI + Synth v2 + Neural Vocals")

    print("\n=== 1/11 ANALYSE REFERENCE SONGS ===")
    extract_cmd = [
        sys.executable, "-u", "daw_extract.py",
        "--audio-dir", audio,
        "--out", dataset,
        "--device", "cuda",
    ]
    if args.force_extract:
        extract_cmd.append("--force")
    run(extract_cmd, repo)
    index = dataset / "index.jsonl"
    if not index.exists():
        raise RuntimeError(f"Missing DAW reference index: {index}")

    print("\n=== 2/11 BUILD MULTIMODAL REFERENCE LIBRARY ===")
    lib_cmd = [
        sys.executable, "-u", "reference_library.py",
        "--index", index,
        "--out", library,
    ]
    if old_codec_index.exists():
        lib_cmd += ["--codec-index", old_codec_index]
    run(lib_cmd, repo)

    print("\n=== 3/11 AI PRODUCER BRAIN ===")
    brain_cmd = [
        sys.executable, "-u", "producer_ai.py",
        "--idea", args.idea,
        "--references", library,
        "--out", plan,
        "--model", args.ai_model,
        "--device", "cuda",
        "--bars", str(args.bars),
    ]
    if args.no_ai:
        brain_cmd.append("--no-ai")
    run(brain_cmd, repo)

    if args.no_vocals:
        print("\n=== 4/11 LYRICS ===")
        lyrics_json.write_text(
            json.dumps({"format": "musicm8-lyrics-v1", "title": "Instrumental", "language": args.vocal_language, "idea": args.idea, "hook": "", "sections": []}, indent=2),
            encoding="utf-8",
        )
        lyrics_txt.write_text("[Instrumental]\n", encoding="utf-8")
        print("⏭️ Vocals disabled; this run is instrumental.")
    else:
        print("\n=== 4/11 ORIGINAL AI LYRICS ===")
        lyric_cmd = [
            sys.executable, "-u", "lyrics_ai.py",
            "--idea", args.idea,
            "--plan", plan,
            "--out", lyrics_json,
            "--model", args.ai_model,
            "--device", "cuda",
            "--language", args.vocal_language,
        ]
        if args.no_ai:
            lyric_cmd.append("--no-ai")
        run(lyric_cmd, repo)

    print("\n=== 5/11 HIERARCHICAL CHORD-AWARE COMPOSITION ===")
    run(
        [
            sys.executable, "-u", "plan_to_midi_v2.py",
            "--plan", plan,
            "--out", project,
            "--seed", str(args.seed),
        ],
        repo,
    )

    print("\n=== 6/11 SYNTH V2 INVERSE SOUND DESIGN ===")
    if args.skip_sound_match:
        matched.write_text(
            json.dumps(
                {
                    "format": "musicm8-inverse-synth-v2",
                    "roles": {},
                    "note": "Sound matching skipped; Synth v2 renderer uses fallback patches.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("⏭️ Sound matching skipped.")
    else:
        match_cmd = [
            sys.executable, "-u", "sound_matcher_v2.py",
            "--plan", plan,
            "--references", library,
            "--out", matched,
            "--iterations", str(max(0, args.match_iters)),
            "--seconds", str(args.match_seconds),
            "--seed", str(args.seed),
            "--cache-dir", patch_cache,
        ]
        if args.force_sound_match:
            match_cmd.append("--force")
        run(match_cmd, repo)

    print("\n=== 7/11 SYNTH V2 + FX + SIDECHAIN ===")
    run(
        [
            sys.executable, "-u", "render_matched_v2.py",
            "--plan", plan,
            "--midi", arrangement,
            "--patches", matched,
            "--out", project,
        ],
        repo,
    )

    print("\n=== 8/11 TONAL BALANCE + TRUE STEREO + MASTER POLISH ===")
    run(
        [
            sys.executable, "-u", "mix_polish_v2.py",
            "--project", project,
            "--out", project / "master.wav",
        ],
        repo,
    )

    vocal_ok = False
    vocal_error: str | None = None
    if not args.no_vocals:
        print("\n=== 9/11 ALIGN LYRICS TO VOCAL MELODY ===")
        melody_midi = project / "midi_stems" / "melody.mid"
        if melody_midi.exists():
            run(
                [
                    sys.executable, "-u", "vocal_score.py",
                    "--plan", plan,
                    "--lyrics", lyrics_json,
                    "--melody-midi", melody_midi,
                    "--out", vocal_score,
                ],
                repo,
            )
        else:
            print("⚠️ No melody MIDI was produced, so vocal score alignment is unavailable.")

        print("\n=== 10/11 NEURAL SINGING VOICE (ACE-STEP 1.5) ===")
        if raw_vocal.exists():
            raw_vocal.unlink()
        vocal_cmd = [
            sys.executable, "-u", "neural_vocals.py",
            "--repo", repo,
            "--root", root,
            "--backing", project / "master.wav",
            "--lyrics", lyrics_txt,
            "--plan", plan,
            "--out", raw_vocal,
            "--style", args.vocal_style,
            "--language", args.vocal_language,
            "--seed", str(args.seed),
            "--steps", str(args.vocal_steps),
        ]
        vocal_ok, vocal_error = run_optional(vocal_cmd, repo, "Neural vocal generation")

        print("\n=== 11/11 VOCAL FX + DOUBLES + FINAL MIX ===")
        if vocal_ok and raw_vocal.exists():
            run(
                [
                    sys.executable, "-u", "mix_vocals.py",
                    "--project", project,
                    "--vocal", raw_vocal,
                    "--plan", plan,
                    "--out", project / "master.wav",
                ],
                repo,
            )
        else:
            print("⏭️ No neural vocal available; keeping the polished instrumental master.")
    else:
        print("\n=== 9-11/11 VOCALS DISABLED ===")

    payload = {
        "format": "musicm8-ai-project-v4",
        "engine": "musicm8-synth-v2",
        "composer": "hierarchical-chord-aware-v2",
        "mix_engine": "musicm8-mix-polish-v1",
        "vocal_engine": "ACE-Step-1.5 base Lego vocals" if not args.no_vocals else None,
        "idea": args.idea,
        "plan": str(plan),
        "lyrics_json": str(lyrics_json),
        "lyrics_text": str(lyrics_txt),
        "vocal_score": str(vocal_score) if vocal_score.exists() else None,
        "raw_neural_vocal": str(raw_vocal) if raw_vocal.exists() else None,
        "vocals_dir": str(project / "vocals") if (project / "vocals").exists() else None,
        "vocal_status": "ok" if vocal_ok else ("disabled" if args.no_vocals else "failed"),
        "vocal_error": vocal_error,
        "reference_library": str(library),
        "arrangement_midi": str(arrangement),
        "midi_stems": str(project / "midi_stems"),
        "matched_patches": str(matched),
        "sound_match_previews": str(project / "sound_matches"),
        "audio_stems": str(project / "audio_stems"),
        "polished_audio_stems": str(project / "audio_stems_polished"),
        "synth_patches": str(project / "synth_patches.json"),
        "mix_report": str(project / "mix_report.json"),
        "vocal_mix_report": str(project / "vocal_mix_report.json") if (project / "vocal_mix_report.json").exists() else None,
        "master_instrumental": str(project / "master_instrumental.wav") if (project / "master_instrumental.wav").exists() else str(project / "master.wav"),
        "master": str(project / "master.wav"),
        "legacy_codec_tokens": str(old_codec_index) if old_codec_index.exists() else None,
        "note": (
            "Musicm8 v4 adds a dedicated vocal system. The producer AI writes an original section-aware lyric sheet, "
            "lyrics are aligned to the generated melody as an editable vocal score, and ACE-Step 1.5's base-model Lego task "
            "generates a vocals track in the context of Musicm8's own backing. A DAW-style vocal stage then high-passes, EQs, "
            "de-esses, compresses, adds subtle stereo doubles/reverb/delay, ducks the backing lightly, and writes the complete master. "
            "ACE-Step runs in an isolated Python 3.12 environment because the main Colab runtime may use Python 3.13; model weights are cached in Drive."
        ),
    }
    (project / "project.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n✅ MUSICM8 AI PRODUCER V4 COMPLETE")
    print("Plan:", plan)
    print("Lyrics:", lyrics_txt)
    print("Vocal score:", vocal_score if vocal_score.exists() else "not available")
    print("MIDI:", arrangement)
    print("Matched sound patches:", matched)
    print("Instrument stems:", project / "audio_stems_polished")
    if raw_vocal.exists():
        print("Raw neural vocal:", raw_vocal)
        print("Processed vocals:", project / "vocals")
    print("Instrumental master:", project / "master_instrumental.wav" if (project / "master_instrumental.wav").exists() else project / "master.wav")
    print("Final song master:", project / "master.wav")
    print("Project:", project / "project.json")


if __name__ == "__main__":
    main()
