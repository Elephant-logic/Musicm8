from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio

from codec import codec_from_info
from model import AudioTokenTransformer, ModelConfig
from planner import heuristic_plan, load_plan, plan_to_conditions


def choose_device(name: str) -> torch.device:
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = AudioTokenTransformer(cfg).to(device)
    model.load_trainable_state_dict(ckpt["model"])
    model.eval()
    codec_info = ckpt.get("data_meta", {}).get("codec")
    if not codec_info:
        raise ValueError("V2 generation requires checkpoint data_meta.codec")
    codec = codec_from_info(codec_info, device)
    return ckpt, model, codec


def main() -> None:
    p = argparse.ArgumentParser(description="Generate/continue music with Musicm8 V2.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--seconds", type=float, default=12.0, help="Total output length.")
    p.add_argument("--plan", type=Path, default=None)
    p.add_argument("--bpm", type=float, default=None)
    p.add_argument("--key", default=None)
    p.add_argument("--prompt-audio", type=Path, default=None, help="Optional audio prefix for continuation.")
    p.add_argument("--semantic-checkpoint", type=Path, default=None, help="Optional text->semantic LM checkpoint for two-stage generation.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=250)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("sample.wav"))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = p.parse_args()

    device = choose_device(args.device)
    _, model, codec = load_model(args.checkpoint, device)
    frames = max(1, round(args.seconds * codec.info.frame_rate))
    plan = load_plan(args.plan) if args.plan else heuristic_plan(args.prompt, args.seconds, args.bpm, args.key)
    plan.duration = args.seconds
    cond = plan_to_conditions(plan, frames, codec.info.frame_rate)
    if args.semantic_checkpoint:
        from semantic_lm import SemanticLM, SemanticLMConfig
        sem_ck = torch.load(args.semantic_checkpoint, map_location="cpu", weights_only=False)
        sem = SemanticLM(SemanticLMConfig(**sem_ck["config"])).to(device)
        sem.load_trainable_state_dict(sem_ck["model"]); sem.eval()
        sem_rate = float(sem_ck["semantic_frame_rate"])
        sem_steps = max(1, round(args.seconds * sem_rate))
        sem_ids = sem.generate(args.prompt, sem_steps, seed=args.seed)[0].cpu()
        aligned = torch.nn.functional.interpolate(sem_ids.float().view(1,1,-1), size=frames, mode="nearest").long().view(1,-1)
        cond["semantic_ids"] = aligned.clamp_max(model.cfg.condition["semantic_vocab"] - 1)

    prompt_codes = None
    if args.prompt_audio:
        wav, sr = torchaudio.load(args.prompt_audio)
        prompt_codes = codec.encode(wav, sr).unsqueeze(0)
        if prompt_codes.shape[-1] >= frames:
            raise ValueError("prompt audio must be shorter than --seconds total output")

    print(f"Generating {args.seconds:.2f}s with {codec.info.name}, Q={codec.info.num_codebooks}, {codec.info.frame_rate:.2f} fps")
    codes = model.generate(
        args.prompt, frames, cond, prompt_codes=prompt_codes,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        cfg_scale=args.cfg_scale, seed=args.seed,
    )
    wav = codec.decode(codes)
    target_samples = round(args.seconds * codec.info.sample_rate)
    wav = wav[..., :target_samples]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out, wav, codec.info.sample_rate)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
