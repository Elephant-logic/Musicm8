from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf


def key_name(root: int, mode: str) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[int(root) % 12]} {'Major' if str(mode).lower() == 'major' else 'minor'}"


def main() -> None:
    p = argparse.ArgumentParser(description="Run ACE-Step 1.5 base-model Lego vocals for Musicm8.")
    p.add_argument("--ace-root", type=Path, required=True)
    p.add_argument("--backing", type=Path, required=True)
    p.add_argument("--lyrics", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--style", default="expressive contemporary lead vocal, clear lyrics, musical phrasing")
    p.add_argument("--language", default="en")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=36)
    args = p.parse_args()

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    lyrics = args.lyrics.read_text(encoding="utf-8").strip()
    if not lyrics:
        raise RuntimeError("Lyrics are empty")
    info = sf.info(str(args.backing))
    duration = max(10.0, min(600.0, float(info.duration)))

    caption = (
        f"{plan.get('style', 'electronic')} song, {args.style}. "
        f"Follow the supplied backing track, sing the supplied lyrics naturally, "
        f"stay in {key_name(int(plan.get('key_root', 0)), str(plan.get('mode', 'minor')))}, "
        f"and match {float(plan.get('bpm', 120.0)):.0f} BPM. "
        "Return a lead vocal track only, with no drums, bass or accompaniment."
    )[:510]

    print("🎤 Initializing ACE-Step base model for vocal Lego generation...")
    handler = AceStepHandler()
    handler.initialize_service(
        project_root=str(args.ace_root),
        config_path="acestep-v15-base",
        device="cuda",
    )

    params = GenerationParams(
        task_type="lego",
        src_audio=str(args.backing),
        instruction="Generate the vocals track based on the audio context:",
        caption=caption,
        lyrics=lyrics[:4096],
        instrumental=False,
        vocal_language=args.language,
        bpm=int(round(float(plan.get("bpm", 120.0)))),
        keyscale=key_name(int(plan.get("key_root", 0)), str(plan.get("mode", "minor"))),
        timesignature="4",
        duration=duration,
        repainting_start=0.0,
        repainting_end=-1,
        inference_steps=max(16, min(80, int(args.steps))),
        guidance_scale=7.5,
        shift=3.0,
        seed=int(args.seed),
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_lyrics=False,
        use_cot_language=False,
    )
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=False,
        seeds=[int(args.seed)],
        audio_format="wav",
    )

    generated_dir = args.out.parent / "ace_step_output"
    generated_dir.mkdir(parents=True, exist_ok=True)
    result = generate_music(handler, None, params, config, save_dir=str(generated_dir))
    if not result.success or not result.audios:
        raise RuntimeError(result.error or result.status_message or "ACE-Step returned no vocal audio")

    source = Path(result.audios[0]["path"])
    if not source.exists():
        raise FileNotFoundError(source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.out)

    meta = {
        "format": "musicm8-neural-vocal-v1",
        "backend": "ACE-Step-1.5",
        "model": "acestep-v15-base",
        "task": "lego:vocals",
        "source": str(source),
        "output": str(args.out),
        "duration": duration,
        "bpm": int(round(float(plan.get("bpm", 120.0)))),
        "keyscale": key_name(int(plan.get("key_root", 0)), str(plan.get("mode", "minor"))),
        "language": args.language,
        "seed": int(args.seed),
        "inference_steps": int(args.steps),
        "caption": caption,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("✅ Neural lead vocal:", args.out)


if __name__ == "__main__":
    main()
