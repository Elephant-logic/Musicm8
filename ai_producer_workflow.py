from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch


class StageFailure(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_streamed(cmd: list[Any], cwd: Path, stage: str, failure_path: Path) -> None:
    argv = [str(x) for x in cmd]
    print(f"\n$ {' '.join(argv)}", flush=True)
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=180)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip("\n"))
    rc = proc.wait()
    if rc != 0:
        payload = {
            "format": "musicm8-workflow-failure-v1",
            "stage": stage,
            "exit_code": rc,
            "command": argv,
            "tail": list(tail),
        }
        write_json(failure_path, payload)
        print("\n============================================================")
        print("❌ MUSICM8 FAILED AT:", stage)
        print("============================================================")
        if tail:
            print("\n".join(list(tail)[-80:]))
        print("Failure report:", failure_path)
        raise StageFailure(f"{stage} failed with exit code {rc}")


def run_optional(cmd: list[Any], cwd: Path, stage: str, failure_path: Path) -> tuple[bool, str | None]:
    try:
        run_streamed(cmd, cwd, stage, failure_path)
        return True, None
    except Exception as exc:
        print(f"⚠️ {stage} failed: {exc}")
        return False, str(exc)


def clear_stale(project: Path) -> None:
    stale = [
        project / "composition_report.json",
        project / "composition_failure.json",
        project / "soundfont_render_report.json",
        project / "render_timing_report.json",
        project / "workflow_failure.json",
        project / "final_status.json",
        project / "project.json",
        project / "vocals" / "vocal_status.json",
        project / "vocals" / "vocal_quality.json",
        project / "vocals" / "vocal_word_quality.json",
        project / "vocals" / "vocal_backend.log",
    ]
    for path in stale:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 v10.1 resilient AI producer: learned section composition + sampled instruments + score/phoneme vocals.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--idea", required=True)
    p.add_argument("--bars", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ai-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--composer-candidates", type=int, default=2)
    p.add_argument("--soundfont", type=Path, default=None)
    p.add_argument("--match-iters", type=int, default=0)
    p.add_argument("--match-seconds", type=float, default=3.0)
    p.add_argument("--force-sound-match", action="store_true")
    p.add_argument("--skip-sound-match", action="store_true")
    p.add_argument("--vocal-style", default="expressive contemporary lead vocal")
    p.add_argument("--vocal-language", default="en")
    p.add_argument("--vocal-steps", type=int, default=24)
    p.add_argument("--lyrics-file", type=Path, default=None)
    p.add_argument("--voice-reference", type=Path, default=None)
    p.add_argument("--no-ai", action="store_true")
    p.add_argument("--no-vocals", action="store_true")
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
    vocals_dir = project / "vocals"

    for path in (audio, work, reference_dir, project, vocals_dir):
        path.mkdir(parents=True, exist_ok=True)
    clear_stale(project)

    failure_path = project / "workflow_failure.json"
    plan = project / "plan.json"
    lyrics_json = project / "lyrics.json"
    lyrics_txt = project / "lyrics.txt"
    arrangement = project / "arrangement.mid"
    composition_report = project / "composition_report.json"
    soundfont_report = project / "soundfont_render_report.json"
    vocal_score = project / "vocal_score.json"
    vocal_midi = project / "vocal_melody.mid"
    raw_vocal = vocals_dir / "neural_lead_raw.wav"
    synced_vocal = vocals_dir / "neural_lead_synced.wav"
    pitch_quality = vocals_dir / "vocal_quality.json"
    word_quality = vocals_dir / "vocal_word_quality.json"
    vocal_status_path = vocals_dir / "vocal_status.json"
    final_status_path = project / "final_status.json"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the Colab AI producer workflow.")
    if args.voice_reference is not None and not args.voice_reference.exists():
        raise FileNotFoundError(args.voice_reference)

    inherited = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(repo) + (os.pathsep + inherited if inherited else "")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Idea:", args.idea)
    print("Song seed:", args.seed)
    print("Musicm8 v10.1 — RESILIENT LEARNED PRODUCER")
    print("  MIDI-LLM composes each song section")
    print("  weak AI roles get targeted AI rescue rather than a rule-note fallback")
    print("  all pitched material is harmony-locked after generation")
    print("  all stems share one groove clock")
    print("  SoulX uses words + phonemes + exact vocal score")

    try:
        print("\n=== 1/12 ANALYSE REFERENCE SONGS ===")
        cmd = [sys.executable, "-u", "daw_extract.py", "--audio-dir", audio, "--out", dataset, "--device", "cuda"]
        if args.force_extract:
            cmd.append("--force")
        run_streamed(cmd, repo, "1/12 reference analysis", failure_path)
        index = dataset / "index.jsonl"
        if not index.exists():
            raise FileNotFoundError(index)

        print("\n=== 2/12 BUILD REFERENCE LIBRARY ===")
        cmd = [sys.executable, "-u", "reference_library.py", "--index", index, "--out", library]
        if old_codec_index.exists():
            cmd += ["--codec-index", old_codec_index]
        run_streamed(cmd, repo, "2/12 reference library", failure_path)

        print("\n=== 3/12 AI PRODUCER BRIEF ===")
        cmd = [
            sys.executable, "-u", "producer_ai.py",
            "--idea", args.idea, "--references", library, "--out", plan,
            "--model", args.ai_model, "--device", "cuda", "--bars", str(args.bars),
        ]
        if args.no_ai:
            cmd.append("--no-ai")
        run_streamed(cmd, repo, "3/12 producer brief", failure_path)

        print("\n=== 4/12 GENRE PRODUCTION DIRECTOR ===")
        run_streamed(
            [sys.executable, "-u", "style_production.py", "--plan", plan, "--references", library, "--out", plan],
            repo, "4/12 production director", failure_path,
        )

        print("\n=== 5/12 LYRICS ===")
        if args.no_vocals:
            write_json(lyrics_json, {
                "format": "musicm8-lyrics-v3", "title": "Instrumental", "language": args.vocal_language,
                "idea": args.idea, "hook": "", "source": "disabled", "sections": [],
            })
            lyrics_txt.write_text("[Instrumental]\n", encoding="utf-8")
        else:
            cmd = [
                sys.executable, "-u", "lyrics_ai_v3.py",
                "--idea", args.idea, "--plan", plan, "--out", lyrics_json,
                "--model", args.ai_model, "--device", "cuda", "--language", args.vocal_language,
            ]
            if args.no_ai:
                cmd.append("--no-ai")
            if args.lyrics_file is not None and args.lyrics_file.exists() and args.lyrics_file.read_text(encoding="utf-8").strip():
                cmd += ["--lyrics-file", args.lyrics_file]
            run_streamed(cmd, repo, "5/12 lyrics", failure_path)

        print("\n=== 6/12 LEARNED SECTION COMPOSER + AI ROLE RESCUE ===")
        run_streamed([
            sys.executable, "-u", "midi_llm_composer_v3.py",
            "--root", root, "--plan", plan, "--idea", args.idea, "--out", project,
            "--seed", str(args.seed), "--candidates", str(max(1, min(3, args.composer_candidates))),
        ], repo, "6/12 learned composer", failure_path)
        if not arrangement.exists() or not composition_report.exists():
            raise RuntimeError("V3 composer returned without arrangement.mid/composition_report.json")

        print("\n=== 7/12 STYLE-DIRECTED SAMPLED INSTRUMENT RENDER ===")
        render_cmd: list[Any] = [sys.executable, "-u", "soundfont_render_v1.py", "--project", project, "--plan", plan]
        if args.soundfont is not None:
            render_cmd += ["--soundfont", args.soundfont]
        run_streamed(render_cmd, repo, "7/12 instrument render", failure_path)

        print("\n=== 8/12 CONTROLLED STEM MIX / MASTER ===")
        shutil.rmtree(project / "audio_stems_polished", ignore_errors=True)
        run_streamed(
            [sys.executable, "-u", "mix_polish_v3.py", "--project", project, "--plan", plan, "--out", project / "master.wav"],
            repo, "8/12 stem mix", failure_path,
        )
        instrumental = project / "master_instrumental.wav"
        if not (project / "master.wav").exists():
            raise FileNotFoundError(project / "master.wav")
        shutil.copy2(project / "master.wav", instrumental)

        vocal_ok = False
        vocal_error: str | None = None
        final_kind = "instrumental_requested" if args.no_vocals else "instrumental_only_vocal_failed"
        for stale in (raw_vocal, synced_vocal, pitch_quality, word_quality, vocal_status_path):
            stale.unlink(missing_ok=True)

        if not args.no_vocals:
            print("\n=== 9/12 VOCAL SCORE FROM ACTUAL AI HARMONY/LEAD ===")
            score_cmd: list[Any] = [
                sys.executable, "-u", "vocal_lead_score_ai.py",
                "--plan", plan, "--lyrics", lyrics_json,
                "--midi-out", vocal_midi, "--score-out", vocal_score,
            ]
            harmony_midi = project / "midi_stems" / "chords.mid"
            melody_midi = project / "midi_stems" / "melody.mid"
            if harmony_midi.exists():
                score_cmd += ["--harmony-midi", harmony_midi]
            if melody_midi.exists():
                score_cmd += ["--melody-midi", melody_midi]
            run_streamed(score_cmd, repo, "9/12 vocal score", failure_path)

            print("\n=== 10/12 SOULX SCORE/PHONEME SINGER ===")
            singer_cmd: list[Any] = [
                sys.executable, "-u", "soulx_vocals.py",
                "--repo", repo, "--root", root, "--score", vocal_score, "--out", raw_vocal,
                "--svc-steps", str(max(8, min(40, args.vocal_steps))),
            ]
            if args.voice_reference is not None:
                singer_cmd += ["--voice-reference", args.voice_reference]
            vocal_ok, vocal_error = run_optional(singer_cmd, repo, "10/12 SoulX singer", failure_path)

            print("\n=== 11/12 VOCAL PITCH + WORD QA ===")
            if vocal_ok and raw_vocal.exists():
                shutil.copy2(raw_vocal, synced_vocal)
                vocal_ok, vocal_error = run_optional(
                    [sys.executable, "-u", "vocal_quality_gate.py", "--vocal", synced_vocal, "--score", vocal_score, "--report", pitch_quality],
                    repo, "11/12 vocal pitch QA", failure_path,
                )
                if vocal_ok:
                    vocal_ok, vocal_error = run_optional(
                        [sys.executable, "-u", "vocal_word_gate.py", "--vocal", synced_vocal, "--lyrics", lyrics_txt,
                         "--report", word_quality, "--model", "openai/whisper-base.en", "--min-recall", "0.55"],
                        repo, "11/12 lyric intelligibility QA", failure_path,
                    )

            print("\n=== 12/12 VOCAL PRODUCTION + FINAL SONG ===")
            if vocal_ok and synced_vocal.exists():
                shutil.copy2(instrumental, project / "master.wav")
                run_streamed(
                    [sys.executable, "-u", "mix_vocals.py", "--project", project, "--vocal", synced_vocal, "--plan", plan, "--out", project / "master.wav"],
                    repo, "12/12 vocal mix", failure_path,
                )
                final_kind = "song_with_vocals"
            else:
                shutil.copy2(instrumental, project / "master.wav")
                print("❌ Vocal was not accepted; master.wav is explicitly the instrumental safety copy.")
        else:
            shutil.copy2(instrumental, project / "master.wav")

        final_status = {
            "format": "musicm8-final-status-v10.1",
            "final_kind": final_kind,
            "seed": args.seed,
            "vocal_requested": not args.no_vocals,
            "vocal_ok": bool(vocal_ok),
            "vocal_error": vocal_error,
            "vocal_backend": read_json(vocal_status_path),
            "pitch_qa": read_json(pitch_quality),
            "word_qa": read_json(word_quality),
            "composer": read_json(composition_report),
            "instrument_renderer": read_json(soundfont_report),
            "lyrics": str(lyrics_txt),
            "instrumental": str(instrumental),
            "master": str(project / "master.wav"),
        }
        write_json(final_status_path, final_status)

        payload = {
            "format": "musicm8-ai-project-v10.1",
            "producer": "Qwen producer brief + genre production director",
            "composer": "MIDI-LLM section composer v3 + targeted learned role rescue",
            "composer_report": str(composition_report),
            "engine": "style-directed sampled GM SoundFont renderer",
            "mix_engine": "Musicm8 timing-safe stem mix",
            "vocal_engine": "SoulX score/phoneme chunks + pitch/word QA" if not args.no_vocals else None,
            "idea": args.idea,
            "seed": args.seed,
            "plan": str(plan),
            "lyrics_json": str(lyrics_json),
            "lyrics_text": str(lyrics_txt),
            "lyrics_source": read_json(lyrics_json).get("source"),
            "arrangement_midi": str(arrangement),
            "midi_stems": str(project / "midi_stems"),
            "audio_stems": str(project / "audio_stems"),
            "vocal_score": str(vocal_score) if vocal_score.exists() else None,
            "vocal_melody": str(vocal_midi) if vocal_midi.exists() else None,
            "raw_neural_vocal": str(raw_vocal) if raw_vocal.exists() else None,
            "synced_neural_vocal": str(synced_vocal) if synced_vocal.exists() else None,
            "final_kind": final_kind,
            "master_instrumental": str(instrumental),
            "master": str(project / "master.wav"),
            "rule_note_composer_used": False,
        }
        write_json(project / "project.json", payload)

        print("\n============================================================")
        print("✅ MUSICM8 V10.1 COMPLETE")
        print("Seed:", args.seed)
        print("Composer role totals:", read_json(composition_report).get("role_notes"))
        print("Final kind:", final_kind)
        print("Instrumental:", instrumental)
        if final_kind == "song_with_vocals":
            print("✅ FINAL SONG CONTAINS QA-PASSED VOCALS:", project / "master.wav")
        elif not args.no_vocals:
            print("❌ NO VOCAL FINAL — instrumental safety copy retained")
        print("============================================================")

    except Exception as exc:
        if not failure_path.exists():
            write_json(failure_path, {
                "format": "musicm8-workflow-failure-v1",
                "stage": "workflow/python",
                "error": repr(exc),
                "seed": args.seed,
            })
        print("\n❌ MUSICM8 WORKFLOW STOPPED")
        print("Reason:", exc)
        print("Failure report:", failure_path)
        raise


if __name__ == "__main__":
    main()
