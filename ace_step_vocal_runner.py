from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
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


def parse_lyrics(text: str) -> list[dict]:
    blocks: list[dict] = []
    current: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            if current is not None:
                blocks.append(current)
            current = {"name": line[1:-1].strip().lower(), "lines": []}
        else:
            if current is None:
                current = {"name": "song", "lines": []}
            current["lines"].append(line)
    if current is not None:
        blocks.append(current)
    return blocks


def lyric_lines_for_sections(plan: dict, lyrics: str) -> list[list[str]]:
    blocks = parse_lyrics(lyrics)
    queues: dict[str, list[list[str]]] = {}
    for block in blocks:
        queues.setdefault(block["name"], []).append(list(block["lines"]))

    result: list[list[str]] = []
    fallback_blocks = [list(b["lines"]) for b in blocks if b["lines"]]
    fallback_i = 0
    for sec in plan.get("sections", []):
        name = str(sec.get("name", "section")).lower()
        lines: list[str] = []
        candidates = [name]
        if name == "final":
            candidates += ["chorus", "drop"]
        if "pre" in name:
            candidates += ["build"]
        for candidate in candidates:
            q = queues.get(candidate, [])
            if q:
                lines = q.pop(0)
                break
        if not lines and name not in {"intro", "outro", "instrumental"} and fallback_i < len(fallback_blocks):
            lines = fallback_blocks[fallback_i]
            fallback_i += 1
        result.append(lines)
    return result


def linear_resample(audio: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return audio.astype(np.float32)
    n_in = audio.shape[0]
    n_out = max(1, int(round(n_in * out_sr / in_sr)))
    old = np.linspace(0.0, 1.0, n_in, endpoint=False)
    new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    out = np.stack([np.interp(new, old, audio[:, ch]) for ch in range(audio.shape[1])], axis=1)
    return out.astype(np.float32)


def to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2].astype(np.float32)


def fade_edges(audio: np.ndarray, sr: int, seconds: float = 0.05) -> np.ndarray:
    y = audio.copy()
    n = min(int(sr * seconds), max(0, len(y) // 3))
    if n > 1:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return y


def chunk_specs(plan: dict, lyrics: str, bpm: float, max_seconds: float = 15.0) -> list[dict]:
    bar_seconds = 4.0 * 60.0 / max(30.0, bpm)
    sections = list(plan.get("sections", []))
    lyric_sets = lyric_lines_for_sections(plan, lyrics)
    specs: list[dict] = []
    cursor = 0.0

    for sec_i, sec in enumerate(sections):
        bars = max(1, int(sec.get("bars", 1)))
        sec_duration = bars * bar_seconds
        sec_start = cursor
        sec_end = sec_start + sec_duration
        cursor = sec_end
        lines = lyric_sets[sec_i] if sec_i < len(lyric_sets) else []
        if not lines:
            continue

        pieces = max(1, int(math.ceil(sec_duration / max_seconds)))
        # Do not create more lyric chunks than useful lyric phrases unless the section is very long.
        pieces = max(1, min(pieces, max(1, len(lines))))
        for i in range(pieces):
            a = sec_start + sec_duration * i / pieces
            b = sec_start + sec_duration * (i + 1) / pieces
            lo = int(round(len(lines) * i / pieces))
            hi = int(round(len(lines) * (i + 1) / pieces))
            chunk_lines = lines[lo:hi] or [lines[min(lo, len(lines) - 1)]]
            specs.append(
                {
                    "section": str(sec.get("name", "section")),
                    "start": a,
                    "end": b,
                    "lyrics": "\n".join(chunk_lines),
                }
            )

    if not specs:
        # Last-resort single chunk: still bounded on T4.
        duration = min(max_seconds, cursor if cursor > 0 else 12.0)
        plain = "\n".join(line for line in lyrics.splitlines() if not line.strip().startswith("["))
        specs = [{"section": "song", "start": 0.0, "end": duration, "lyrics": plain[:1200]}]
    return specs


def main() -> None:
    p = argparse.ArgumentParser(description="Run ACE-Step 1.5 for Musicm8 vocals using T4-safe section chunks.")
    p.add_argument("--ace-root", type=Path, required=True)
    p.add_argument("--backing", type=Path, required=True)
    p.add_argument("--lyrics", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--style", default="expressive contemporary lead vocal, clear lyrics, musical phrasing")
    p.add_argument("--language", default="en")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--mode", choices=["lego", "cover"], default="lego")
    p.add_argument("--max-chunk-seconds", type=float, default=15.0)
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

    backing, backing_sr = sf.read(str(args.backing), always_2d=True, dtype="float32")
    backing = to_stereo(backing)
    total_duration = len(backing) / float(backing_sr)
    bpm = int(round(float(plan.get("bpm", 120.0))))
    keyscale = key_name(int(plan.get("key_root", 0)), str(plan.get("mode", "minor")))

    gpu_gb = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_gb = props.total_memory / (1024 ** 3)
    low_vram = 0 < gpu_gb < 17.0
    print(f"🎤 ACE-Step {args.mode}: GPU VRAM={gpu_gb:.1f}GB; CPU offload={low_vram}")

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

    specs = chunk_specs(plan, lyrics, bpm, max_seconds=max(10.0, min(18.0, args.max_chunk_seconds)))
    print(f"🎤 Vocal generation split into {len(specs)} section chunk(s) for T4 stability")

    generated_dir = args.out.parent / f"ace_step_{args.mode}_chunks"
    generated_dir.mkdir(parents=True, exist_ok=True)
    stitched = np.zeros_like(backing, dtype=np.float32)
    rendered_meta: list[dict] = []

    for idx, spec in enumerate(specs):
        start = max(0.0, float(spec["start"]))
        end = min(total_duration, float(spec["end"]))
        if end <= start + 0.2:
            continue
        a = int(round(start * backing_sr))
        b = int(round(end * backing_sr))
        clip = backing[a:b]
        true_duration = len(clip) / float(backing_sr)
        model_duration = max(10.0, true_duration)
        if model_duration > true_duration:
            pad = int(round((model_duration - true_duration) * backing_sr))
            clip_for_model = np.pad(clip, ((0, pad), (0, 0)))
        else:
            clip_for_model = clip

        clip_path = generated_dir / f"{idx:02d}_backing.wav"
        sf.write(clip_path, clip_for_model, backing_sr, subtype="PCM_16")
        lyric_text = str(spec["lyrics"]).strip()
        if not lyric_text:
            continue

        if args.mode == "lego":
            caption = (
                f"{plan.get('style', 'electronic')} lead vocals, {args.style}. "
                f"Sing these supplied lyrics naturally over the supplied backing, in {keyscale} at {bpm} BPM. "
                "Lead vocal track only, no instruments."
            )[:510]
            task_type = "lego"
            instruction = "Generate the vocals track based on the audio context:"
            cover_strength = 0.8
        else:
            caption = (
                f"{plan.get('style', 'electronic')} song with {args.style}. "
                f"Preserve the supplied backing's harmony and pulse and add a clear sung lead vocal in {keyscale} at {bpm} BPM."
            )[:510]
            task_type = "cover"
            instruction = "Generate a vocal song based on the supplied audio and lyrics:"
            cover_strength = 0.92

        params = GenerationParams(
            task_type=task_type,
            src_audio=str(clip_path),
            instruction=instruction,
            caption=caption,
            lyrics=lyric_text[:1800],
            instrumental=False,
            vocal_language=args.language,
            bpm=bpm,
            keyscale=keyscale,
            timesignature="4",
            duration=model_duration,
            repainting_start=0.0,
            repainting_end=-1,
            audio_cover_strength=cover_strength,
            inference_steps=max(24, min(64, int(args.steps))),
            guidance_scale=7.5,
            shift=3.0,
            seed=int(args.seed) + idx * 17,
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_lyrics=False,
            use_cot_language=False,
        )
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[int(args.seed) + idx * 17],
            audio_format="wav",
        )

        print(f"\n🎤 Chunk {idx+1}/{len(specs)} [{spec['section']}] {start:.1f}-{end:.1f}s")
        print("Lyrics:", lyric_text.replace("\n", " / "))
        out_dir = generated_dir / f"{idx:02d}_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = generate_music(handler, None, params, config, save_dir=str(out_dir))
        if not result.success or not result.audios:
            raise RuntimeError(result.error or result.status_message or f"ACE-Step returned no audio for chunk {idx}")

        source = Path(result.audios[0].get("path", ""))
        if not source.exists():
            raise FileNotFoundError(f"ACE-Step result path missing for chunk {idx}: {source}")
        audio, sr = sf.read(str(source), always_2d=True, dtype="float32")
        audio = to_stereo(audio)
        audio = linear_resample(audio, int(sr), backing_sr)
        wanted = b - a
        if len(audio) < wanted:
            audio = np.pad(audio, ((0, wanted - len(audio)), (0, 0)))
        audio = fade_edges(audio[:wanted], backing_sr)
        stitched[a:b] += audio
        chunk_rms = 20.0 * math.log10(math.sqrt(float((audio.astype("float64") ** 2).mean()) + 1e-12) + 1e-12)
        rendered_meta.append({"section": spec["section"], "start": start, "end": end, "lyrics": lyric_text, "source": str(source), "rms_db": round(chunk_rms, 3)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rendered_meta:
        raise RuntimeError("ACE-Step completed without rendering any vocal chunks")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(stitched))) + 1e-9
    if peak > 0.98:
        stitched *= 0.98 / peak
    sf.write(args.out, stitched, backing_sr, subtype="PCM_24")

    rms = audio_rms_db(args.out)
    if rms < -65.0:
        raise RuntimeError(f"ACE-Step stitched output is effectively silent ({rms:.1f} dB RMS)")

    meta = {
        "format": "musicm8-neural-vocal-v3",
        "backend": "ACE-Step-1.5",
        "model": "acestep-v15-base",
        "task": args.mode,
        "mode": args.mode,
        "chunked": True,
        "chunks": rendered_meta,
        "output": str(args.out),
        "duration": total_duration,
        "bpm": bpm,
        "keyscale": keyscale,
        "language": args.language,
        "seed": int(args.seed),
        "inference_steps": int(args.steps),
        "gpu_vram_gb": round(gpu_gb, 2),
        "cpu_offload": low_vram,
        "rms_db": round(rms, 3),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"✅ ACE-Step {args.mode} stitched audio:", args.out)
    print(f"Audio RMS: {rms:.1f} dB")


if __name__ == "__main__":
    main()
