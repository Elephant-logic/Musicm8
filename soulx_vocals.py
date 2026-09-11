from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SOULX_REPO = "https://github.com/Soul-AILab/SoulX-Singer.git"
SOULX_COMMIT = "81aeb3ae772c70093c3de74dc23c92d983801ae4"
SOULX_MODEL_REPO = "Soul-AILab/SoulX-Singer"
ENV_VERSION = "musicm8-soulx-env-v2"


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, log: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE if log else None,
        stderr=subprocess.STDOUT if log else None,
    )
    text = proc.stdout or ""
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write("\n$ " + " ".join(cmd) + "\n")
            f.write(text)
        if text:
            print(text[-10000:], flush=True)
    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-80:])
        raise RuntimeError(f"Command failed with exit {proc.returncode}: {' '.join(cmd)}\n--- backend tail ---\n{tail}")
    return proc


def ensure_uv() -> str:
    exe = shutil.which("uv")
    if exe:
        return exe
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv>=0.8"], check=True)
    exe = shutil.which("uv")
    if not exe:
        raise RuntimeError("uv installed but executable was not found")
    return exe


def ensure_source(root: Path, log: Path) -> None:
    if not (root / ".git").exists():
        shutil.rmtree(root, ignore_errors=True)
        run(["git", "clone", "--no-checkout", SOULX_REPO, root], log=log)
    print("♻️ Pinning SoulX-Singer source to verified revision", SOULX_COMMIT[:12])
    run(["git", "-C", root, "fetch", "--depth", "1", "origin", SOULX_COMMIT], log=log)
    run(["git", "-C", root, "reset", "--hard", SOULX_COMMIT], log=log)


def ensure_environment(uv: str, venv: Path, env: dict[str, str], log: Path) -> Path:
    py = venv / "bin" / "python"
    stamp = venv / ".musicm8_env_version"
    if py.exists() and stamp.exists() and stamp.read_text(encoding="utf-8", errors="ignore").strip() == ENV_VERSION:
        print("♻️ Reusing Musicm8 SoulX Python 3.10 environment")
        return py

    print("Preparing isolated SoulX-Singer Python 3.10 environment...")
    run([uv, "python", "install", "3.10"], env=env, log=log)
    shutil.rmtree(venv, ignore_errors=True)
    run([uv, "venv", "--python", "3.10", venv], env=env, log=log)

    torch_index = "https://download.pytorch.org/whl/cu121"
    run([
        uv, "pip", "install", "--python", py,
        "--index-url", torch_index,
        "--extra-index-url", "https://pypi.org/simple",
        "torch==2.2.0", "torchaudio==2.2.0",
    ], env=env, log=log)

    packages = [
        "accelerate==1.11.0", "beartype==0.22.9", "einops==0.8.2", "g2p_en==2.1.0",
        "huggingface_hub>=0.20.0", "librosa==0.11.0", "loralib==0.1.2", "mido==1.3.3",
        "ml_collections==1.1.0", "nltk==3.9.2", "numpy<2.0.0", "omegaconf==2.3.0",
        "packaging==24.2", "pretty_midi==0.2.11", "pyloudnorm==0.2.0",
        "rotary_embedding_torch==0.8.9", "scikit_learn==1.7.2", "scipy==1.15.3",
        "soundfile==0.13.1", "tqdm==4.67.1", "transformers==4.41.2",
    ]
    run([uv, "pip", "install", "--python", py, *packages], env=env, log=log)
    run([
        py, "-c",
        "import nltk; "
        "nltk.download('averaged_perceptron_tagger_eng', quiet=True); "
        "nltk.download('averaged_perceptron_tagger', quiet=True); "
        "nltk.download('cmudict', quiet=True); "
        "import torch, torchaudio, g2p_en, omegaconf; "
        "print('SoulX env ready', torch.__version__, torchaudio.__version__)"
    ], env=env, log=log)
    stamp.write_text(ENV_VERSION, encoding="utf-8")
    return py


def ensure_model(py: Path, models_dir: Path, filename: str, env: dict[str, str], log: Path) -> Path:
    target = models_dir / filename
    if target.exists() and target.stat().st_size > 10_000_000:
        print(f"♻️ Reusing SoulX model: {target}")
        return target
    models_dir.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import hf_hub_download; "
        f"p=hf_hub_download(repo_id={SOULX_MODEL_REPO!r}, filename={filename!r}, local_dir={str(models_dir)!r}); "
        "print(p)"
    )
    run([py, "-c", code], env=env, log=log)
    if not target.exists():
        raise FileNotFoundError(f"SoulX model download did not create {target}")
    return target


def main() -> None:
    p = argparse.ArgumentParser(description="Musicm8 score-controlled lead singer using SoulX-Singer, with optional authorized voice conversion.")
    p.add_argument("--repo", type=Path, required=True, help="Musicm8 repo")
    p.add_argument("--root", type=Path, required=True, help="Musicm8 Drive root")
    p.add_argument("--score", type=Path, required=True, help="Musicm8 vocal_score.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--voice-reference", type=Path, default=None, help="Optional clean voice reference you own/have permission to use; best 5-30 s dry singing.")
    p.add_argument("--svc-steps", type=int, default=24)
    p.add_argument("--soulx-root", type=Path, default=Path("/content/SoulX-Singer"))
    args = p.parse_args()

    for path in (args.repo, args.root, args.score):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.voice_reference is not None and not args.voice_reference.exists():
        raise FileNotFoundError(args.voice_reference)

    work = args.root / "work"
    models_dir = work / "soulx_models" / "SoulX-Singer"
    venv = Path("/content/musicm8_soulx_venv")
    uv_cache = Path("/content/musicm8_soulx_uv_cache")
    uv_python = Path("/content/musicm8_soulx_uv_python")
    hf_cache = Path("/content/musicm8_soulx_hf_cache")
    mpl_config = Path("/content/musicm8_soulx_mpl")
    for path in (uv_cache, uv_python, hf_cache, mpl_config, args.out.parent):
        path.mkdir(parents=True, exist_ok=True)
    log = args.out.parent / "vocal_backend.log"
    log.write_text("Musicm8 SoulX vocal backend log\n", encoding="utf-8")

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(uv_cache)
    env["UV_PYTHON_INSTALL_DIR"] = str(uv_python)
    env["HF_HOME"] = str(hf_cache)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(mpl_config)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    status = {
        "format": "musicm8-vocal-backend-status-v9",
        "backend": "SoulX-Singer",
        "source_revision": SOULX_COMMIT,
        "mode": "score-conditioned-clip-safe-chunks",
        "success": False,
        "voice_clone": bool(args.voice_reference),
        "voice_reference": str(args.voice_reference) if args.voice_reference else None,
        "score": str(args.score),
        "log": str(log),
        "errors": [],
    }
    status_path = args.out.parent / "vocal_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    try:
        uv = ensure_uv()
        ensure_source(args.soulx_root, log)
        py = ensure_environment(uv, venv, env, log)
        model = ensure_model(py, models_dir, "model.pt", env, log)

        metadata = args.out.parent / "soulx_target_metadata.json"
        run([py, args.repo / "soulx_score_to_metadata.py", "--score", args.score, "--out", metadata], env=env, log=log)

        prompt_wav = args.soulx_root / "example" / "audio" / "en_prompt.mp3"
        prompt_meta = args.soulx_root / "example" / "audio" / "en_prompt.json"
        phoneset = args.soulx_root / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"
        config = args.soulx_root / "soulxsinger" / "config" / "soulxsinger.yaml"
        for required in (prompt_wav, prompt_meta, phoneset, config):
            if not required.exists():
                raise FileNotFoundError(required)

        guide_copy = args.out.parent / "soulx_score_guide.wav"
        if guide_copy.exists():
            guide_copy.unlink()
        print("\n🎤 SoulX-Singer: phrase chunks with exact words + phonemes + MIDI score")
        print("   Each phrase is generated separately, kept inside its own time window and overlap-added with short edge fades.")
        run([
            py, args.repo / "soulx_chunked_inference.py",
            "--model-path", model,
            "--config", config,
            "--prompt-wav", prompt_wav,
            "--prompt-metadata", prompt_meta,
            "--target-metadata", metadata,
            "--phoneset", phoneset,
            "--out", guide_copy,
            "--device", "cuda",
            "--fp16",
        ], cwd=args.soulx_root, env=env, log=log)

        if not guide_copy.exists() or guide_copy.stat().st_size < 4096:
            raise FileNotFoundError(f"SoulX chunked inference did not create a usable {guide_copy}")

        final_source = guide_copy
        method = "soulx-score-chunked"
        if args.voice_reference is not None:
            svc_model = ensure_model(py, models_dir, "model-svc.pt", env, log)
            cloned = args.out.parent / "soulx_voice_cloned.wav"
            print("\n🧑‍🎤 SoulX-Singer-SVC: converting clip-safe guide to authorized reference timbre")
            run([
                py, args.repo / "soulx_svc_runner.py",
                "--soulx-root", args.soulx_root,
                "--model", svc_model,
                "--prompt", args.voice_reference,
                "--target", guide_copy,
                "--out", cloned,
                "--steps", str(args.svc_steps),
            ], cwd=args.soulx_root, env=env, log=log)
            if not cloned.exists() or cloned.stat().st_size < 4096:
                raise FileNotFoundError(cloned)
            final_source = cloned
            method = "soulx-score-chunked+svc"

        shutil.copy2(final_source, args.out)
        status.update({
            "success": True,
            "method": method,
            "model": str(model),
            "target_metadata": str(metadata),
            "chunk_report": str(guide_copy.with_suffix('.chunks.json')),
            "guide": str(guide_copy),
            "output": str(args.out),
        })
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print("\n✅ SCORE-CONTROLLED CHUNKED SINGING CREATED")
        print("Method:", method)
        print("Vocal:", args.out)
    except Exception as exc:
        status["errors"].append(str(exc))
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        tail = "\n".join(log.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]) if log.exists() else ""
        print("\n================ SOULX VOCAL FAILURE ================")
        print(tail)
        print("=====================================================")
        raise


if __name__ == "__main__":
    main()
