from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf

ACE_REPO = "https://github.com/ace-step/ACE-Step-1.5.git"


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, log: Path | None = None) -> subprocess.CompletedProcess:
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
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        text = proc.stdout or ""
        with log.open("a", encoding="utf-8") as f:
            f.write("\n$ " + " ".join(cmd) + "\n")
            f.write(text)
            if text:
                print(text[-5000:], flush=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


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


def ensure_ace_source(ace_root: Path, log: Path) -> None:
    if (ace_root / ".git").exists():
        print("♻️ Refreshing ACE-Step source...")
        run(["git", "-C", ace_root, "fetch", "--depth", "1", "origin", "main"], log=log)
        run(["git", "-C", ace_root, "reset", "--hard", "origin/main"], log=log)
        return
    shutil.rmtree(ace_root, ignore_errors=True)
    ace_root.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", ACE_REPO, ace_root], log=log)


def ensure_environment(uv: str, ace_root: Path, env: dict[str, str], log: Path) -> None:
    # ACE-Step requires Python <3.13. Keep uv's cache + managed Python on Colab's
    # local filesystem: Google Drive/FUSE does not support all atomic lock/rename
    # operations uv uses and can raise "Operation not permitted".
    print("Preparing/updating isolated ACE-Step Python 3.12 environment...")
    run([uv, "python", "install", "3.12"], env=env, log=log)
    run([uv, "sync", "--python", "3.12"], cwd=ace_root, env=env, log=log)


def ensure_models(uv: str, ace_root: Path, models_dir: Path, env: dict[str, str], log: Path) -> None:
    base = models_dir / "acestep-v15-base"
    vae = models_dir / "vae"
    embed = models_dir / "Qwen3-Embedding-0.6B"
    if base.exists() and vae.exists() and embed.exists():
        print("♻️ ACE-Step vocal models already cached in Drive")
        return
    models_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading ACE-Step base vocal models. This is the large first-time step...")
    run(
        [uv, "run", "acestep-download", "--model", "acestep-v15-base", "--dir", models_dir],
        cwd=ace_root,
        env=env,
        log=log,
    )


def run_ace(
    uv: str,
    ace_root: Path,
    runner: Path,
    mode: str,
    backing: Path,
    lyrics: Path,
    plan: Path,
    out: Path,
    style: str,
    language: str,
    seed: int,
    steps: int,
    env: dict[str, str],
    log: Path,
) -> None:
    if out.exists():
        out.unlink()
    run(
        [
            uv,
            "run",
            "--project",
            ace_root,
            "python",
            runner,
            "--ace-root",
            ace_root,
            "--backing",
            backing,
            "--lyrics",
            lyrics,
            "--plan",
            plan,
            "--out",
            out,
            "--style",
            style,
            "--language",
            language,
            "--seed",
            str(seed),
            "--steps",
            str(steps),
            "--mode",
            mode,
        ],
        cwd=ace_root,
        env=env,
        log=log,
    )
    if not out.exists():
        raise RuntimeError(f"ACE-Step {mode} finished without creating {out}")


def audio_rms_db(path: Path) -> float:
    audio, _ = sf.read(path, always_2d=True, dtype="float32")
    if audio.size == 0:
        return -120.0
    import math
    rms = math.sqrt(float((audio.astype("float64") ** 2).mean()) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)


def extract_vocals_with_demucs(full_song: Path, out: Path, work_dir: Path, log: Path) -> None:
    sep = work_dir / "ace_cover_separation"
    shutil.rmtree(sep, ignore_errors=True)
    sep.mkdir(parents=True, exist_ok=True)
    run(
        [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "--out", sep, full_song],
        log=log,
    )
    candidates = sorted(sep.rglob("vocals.wav"))
    if not candidates:
        raise RuntimeError("Demucs fallback did not create a vocals.wav stem")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], out)
    rms = audio_rms_db(out)
    if rms < -60.0:
        raise RuntimeError(f"Fallback vocal stem is effectively silent ({rms:.1f} dB RMS)")
    print(f"✅ Extracted fallback singing vocal with Demucs ({rms:.1f} dB RMS)")


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap ACE-Step and generate a Musicm8 neural vocal stem with automatic fallback.")
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

    if not args.backing.exists():
        raise FileNotFoundError(args.backing)
    if not args.lyrics.exists():
        raise FileNotFoundError(args.lyrics)

    work = args.root / "work"
    models_dir = work / "ace_step_models"
    hf_cache = work / "hf_cache"

    # IMPORTANT: uv itself must use /content, not Google Drive. Drive is a FUSE
    # mount and rejects some lock/temporary-file operations used by uv.
    uv_cache = Path("/content/musicm8_uv_cache")
    uv_python = Path("/content/musicm8_uv_python")

    log = args.out.parent / "vocal_backend.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("Musicm8 vocal backend log\n", encoding="utf-8")

    uv = ensure_uv()
    ensure_ace_source(args.ace_root, log)

    for path in (models_dir, hf_cache, uv_cache, uv_python):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ACESTEP_CHECKPOINTS_DIR"] = str(models_dir)
    env["UV_CACHE_DIR"] = str(uv_cache)
    env["UV_PYTHON_INSTALL_DIR"] = str(uv_python)
    env["HF_HOME"] = str(hf_cache)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("uv cache (local):", uv_cache)
    print("uv Python (local):", uv_python)
    print("ACE-Step models (Drive):", models_dir)

    ensure_environment(uv, args.ace_root, env, log)
    ensure_models(uv, args.ace_root, models_dir, env, log)

    runner = args.repo / "ace_step_vocal_runner.py"
    if not runner.exists():
        raise FileNotFoundError(runner)

    status = {
        "format": "musicm8-vocal-backend-status-v3",
        "backend": "ACE-Step-1.5",
        "primary": "lego-vocals",
        "fallback": "cover-then-demucs-vocals",
        "success": False,
        "method": None,
        "errors": [],
        "log": str(log),
        "uv_cache": str(uv_cache),
        "uv_python": str(uv_python),
    }

    # Primary path: ACE-Step Lego generates the vocals track directly in context.
    try:
        print("\n🎤 PRIMARY: ACE-Step Lego vocal track")
        run_ace(
            uv, args.ace_root, runner, "lego", args.backing, args.lyrics, args.plan,
            args.out, args.style, args.language, args.seed, args.steps, env, log,
        )
        status["success"] = True
        status["method"] = "lego"
    except Exception as exc:
        status["errors"].append(f"lego: {exc}")
        print(f"⚠️ Lego vocal generation failed: {exc}")

    # Automatic fallback: ask the base model for a close cover with the supplied
    # lyrics, then isolate the resulting singer. This is slower but means a Lego
    # failure does not quietly turn a requested vocal song into an instrumental.
    if not status["success"]:
        cover_song = args.out.parent / "ace_cover_with_vocals.wav"
        try:
            print("\n🎤 FALLBACK: ACE-Step cover + Demucs vocal extraction")
            run_ace(
                uv, args.ace_root, runner, "cover", args.backing, args.lyrics, args.plan,
                cover_song, args.style, args.language, args.seed + 97, max(args.steps, 32), env, log,
            )
            extract_vocals_with_demucs(cover_song, args.out, args.out.parent, log)
            status["success"] = True
            status["method"] = "cover+demucs"
            status["cover_song"] = str(cover_song)
        except Exception as exc:
            status["errors"].append(f"cover+demucs: {exc}")
            print(f"❌ Vocal fallback failed: {exc}")

    status["output"] = str(args.out) if args.out.exists() else None
    status["rms_db"] = round(audio_rms_db(args.out), 3) if args.out.exists() else None
    status_path = args.out.parent / "vocal_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    if not status["success"] or not args.out.exists():
        raise RuntimeError(
            "Musicm8 could not create a singing vocal after both ACE-Step methods. "
            f"See {log} and {status_path} for the exact errors."
        )

    print("✅ Musicm8 neural vocal backend complete:", args.out)
    print("Method:", status["method"])
    print("Status:", status_path)


if __name__ == "__main__":
    main()
