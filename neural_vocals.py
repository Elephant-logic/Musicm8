from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ACE_REPO = "https://github.com/ace-step/ACE-Step-1.5.git"


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def ensure_uv() -> str:
    exe = shutil.which("uv")
    if exe:
        return exe
    print("Installing uv for ACE-Step's isolated Python 3.12 environment...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv>=0.8"], check=True)
    exe = shutil.which("uv")
    if not exe:
        raise RuntimeError("uv installed but executable was not found")
    return exe


def ensure_ace_source(ace_root: Path) -> None:
    if (ace_root / ".git").exists():
        print("♻️ ACE-Step source already present:", ace_root)
        return
    shutil.rmtree(ace_root, ignore_errors=True)
    ace_root.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", ACE_REPO, ace_root])


def ensure_environment(uv: str, ace_root: Path, env: dict[str, str]) -> None:
    marker = ace_root / ".venv" / "pyvenv.cfg"
    if marker.exists():
        print("♻️ ACE-Step Python environment already prepared")
        return
    print("Preparing isolated Python 3.12 environment for ACE-Step...")
    run([uv, "python", "install", "3.12"], env=env)
    run([uv, "sync", "--python", "3.12"], cwd=ace_root, env=env)


def ensure_models(uv: str, ace_root: Path, models_dir: Path, env: dict[str, str]) -> None:
    # ACE-Step's downloader automatically obtains its shared VAE/text encoder from
    # the main model repo before downloading the base DiT used by Lego vocals.
    base = models_dir / "acestep-v15-base"
    vae = models_dir / "vae"
    embed = models_dir / "Qwen3-Embedding-0.6B"
    if base.exists() and vae.exists() and embed.exists():
        print("♻️ ACE-Step vocal models already cached in Drive")
        return
    models_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading ACE-Step vocal models. This is the slow/large first-time step...")
    run(
        [uv, "run", "acestep-download", "--model", "acestep-v15-base", "--dir", models_dir],
        cwd=ace_root,
        env=env,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap ACE-Step 1.5 in Python 3.12 and generate a Musicm8 neural vocal stem.")
    p.add_argument("--repo", type=Path, required=True, help="Musicm8 source repo")
    p.add_argument("--root", type=Path, required=True, help="Musicm8 Drive root")
    p.add_argument("--backing", type=Path, required=True)
    p.add_argument("--lyrics", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--style", default="expressive contemporary lead vocal, intimate verses, stronger hook, clear words")
    p.add_argument("--language", default="en")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=36)
    p.add_argument("--ace-root", type=Path, default=Path("/content/ACE-Step-1.5"))
    args = p.parse_args()

    uv = ensure_uv()
    ensure_ace_source(args.ace_root)

    work = args.root / "work"
    models_dir = work / "ace_step_models"
    uv_cache = work / "uv_cache"
    hf_cache = work / "hf_cache"
    for path in (models_dir, uv_cache, hf_cache):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ACESTEP_CHECKPOINTS_DIR"] = str(models_dir)
    env["UV_CACHE_DIR"] = str(uv_cache)
    env["HF_HOME"] = str(hf_cache)
    env["TOKENIZERS_PARALLELISM"] = "false"

    ensure_environment(uv, args.ace_root, env)
    ensure_models(uv, args.ace_root, models_dir, env)

    runner = args.repo / "ace_step_vocal_runner.py"
    if not runner.exists():
        raise FileNotFoundError(runner)
    run(
        [
            uv,
            "run",
            "--project",
            args.ace_root,
            "python",
            runner,
            "--ace-root",
            args.ace_root,
            "--backing",
            args.backing,
            "--lyrics",
            args.lyrics,
            "--plan",
            args.plan,
            "--out",
            args.out,
            "--style",
            args.style,
            "--language",
            args.language,
            "--seed",
            str(args.seed),
            "--steps",
            str(args.steps),
        ],
        cwd=args.ace_root,
        env=env,
    )
    print("✅ Musicm8 neural vocal backend complete:", args.out)


if __name__ == "__main__":
    main()
