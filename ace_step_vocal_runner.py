from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import soundfile as sf
import torch


def key_name(root: int, mode: str) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[int(root) % 12]} {'Major' if str(mode).lower() == 'major' else 'minor'}"


def audio_rms_db(path: Path) -> float:
    audio, _ = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.size == 0:
        return -120.0
    rms = math.sqrt(float((audio.astype("float64") ** 2).mean()) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)


def main() -> None:
    p = argparse.ArgumentParser(description="Run ACE-Step 1.5 for Musicm8 vocals.")
    p.add_argument("--ace-root", type=Path, required=True)
    p.add_argument("--backing", type=Path, required=True)
    p.add_argument("--lyrics", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--style", default="expressive contemporary lead vocal, clear lyrics, musical phrasing")
    p.add_argument("--language", default="en")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=36)
    p.add_argument("--mode", choices=["lego", "cover"], default="lego")
    args = p.parse_args()

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    if not args.backing.exists():
        raise FileNotFoundError(args.backing)
    if not args.lyrics.exists():
        raise FileNotFoundError(args.lyrics)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    lyrics = args.lyrics.read_text(encoding="utf-8").strip()
    if not lyrics:
        raise RuntimeError("Lyrics are empty")

    info = sf.info(str(args.backing))
    duration = max(10.0, min(600.0, float(info.duration)))
    bpm = int(round(float(plan.get("bpm", 120.0))))
    keyscale = key_name(int(plan.get("key_root", 0)), str(plan.get("mode", "minor")))

    if args.mode == "lego":
        caption = (
            f"{plan.get('style', 'electronic')} lead vocals, {args.style}. "
            f"Sing the supplied lyrics over the supplied backing, follow its phrasing and pulse, "
            f"stay in {keyscale} at {bpm} BPM. Vocal track only; no instruments."
        )[:510]
        task_type = "lego"
        instruction = "Generate the vocals track based on the audio context:"
        cover_strength = 0.8
    else:
        caption = (
            f"{plan.get('style', 'electronic')} song with {args.style}. "
            f"Keep the supplied backing's structure, tempo and harmony and add a clear sung lead vocal "
            f"performing the supplied lyrics in {keyscale} at {bpm} BPM."
        )[:510]
        task_type = "cover"
        instruction = "Generate a vocal song based on the supplied audio and lyrics:"
        cover_strength = 0.9

    # Colab T4 is a 16GB-class GPU. ACE-Step's own configuration recommends CPU
    # offload in the 12-16GB tier, so use it automatically instead of risking a
    # silent/OOM failure during the vocal stage.
    gpu_gb = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_gb = props.total_memory / (1024 ** 3)
    low_vram = 0 < gpu_gb < 17.0
    print(f"🎤 Initializing ACE-Step base model ({args.mode}); GPU VRAM={gpu_gb:.1f}GB; CPU offload={low_vram}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(args.ace_root),
        config_path="acestep-v15-base",
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=low_vram,
        offload_dit_to_cpu=False,
        quantization=None,
        use_mlx_dit=False,
    )
    print("ACE-Step init:", status)
    if not ok:
        raise RuntimeError(f"ACE-Step model initialization failed: {status}")

    params = GenerationParams(
        task_type=task_type,
        src_audio=str(args.backing),
        instruction=instruction,
        caption=caption,
        lyrics=lyrics[:4096],
        instrumental=False,
        vocal_language=args.language,
        bpm=bpm,
        keyscale=keyscale,
        timesignature="4",
        duration=duration,
        repainting_start=0.0,
        repainting_end=-1,
        audio_cover_strength=cover_strength,
        inference_steps=max(24, min(80, int(args.steps))),
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

    generated_dir = args.out.parent / f"ace_step_{args.mode}_output"
    generated_dir.mkdir(parents=True, exist_ok=True)
    result = generate_music(handler, None, params, config, save_dir=str(generated_dir))
    if not result.success or not result.audios:
        raise RuntimeError(result.error or result.status_message or "ACE-Step returned no audio")

    source = Path(result.audios[0].get("path", ""))
    if not source.exists():
        raise FileNotFoundError(f"ACE-Step result path missing: {source}")

    rms = audio_rms_db(source)
    if bool(result.audios[0].get("silent", False)) or rms < -65.0:
        raise RuntimeError(f"ACE-Step returned effectively silent audio ({rms:.1f} dB RMS)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.out)

    meta = {
        "format": "musicm8-neural-vocal-v2",
        "backend": "ACE-Step-1.5",
        "model": "acestep-v15-base",
        "task": task_type,
        "mode": args.mode,
        "source": str(source),
        "output": str(args.out),
        "duration": duration,
        "bpm": bpm,
        "keyscale": keyscale,
        "language": args.language,
        "seed": int(args.seed),
        "inference_steps": int(args.steps),
        "gpu_vram_gb": round(gpu_gb, 2),
        "cpu_offload": low_vram,
        "rms_db": round(rms, 3),
        "caption": caption,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"✅ ACE-Step {args.mode} audio:", args.out)
    print(f"Audio RMS: {rms:.1f} dB")


if __name__ == "__main__":
    main()
