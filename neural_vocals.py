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
    text = proc.stdout or ""
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write("\n$ " + " ".join(cmd) + "\n")
            f.write(text)
        if text:
            print(text[-7000:], flush=True)
    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-30:])
        raise RuntimeError(f"Command failed with exit {proc.returncode}: {' '.join(cmd)}\n--- backend tail ---\n{tail}")
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
    else:
        shutil.rmtree(ace_root, ignore_errors=True)
        ace_root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", ACE_REPO, ace_root], log=log)


def ensure_environment(uv: str, ace_root: Path, env: dict[str, str], log: Path) -> None:
    print("Preparing/updating isolated ACE-Step Python 3.12 environment...")
    run([uv, "python", "install", "3.12"], env=env, log=log)
    run([uv, "sync", "--python", "3.12"], cwd=ace_root, env=env, log=log)


def ensure_models(uv: str, ace_root: Path, models_dir: Path, env: dict[str, str], log: Path) -> None:
    # Ask ACE-Step itself to validate the checkpoint cache. The previous code only
    # checked whether directories existed; a partial/failed download could therefore
    # be mistaken for a usable model cache.
    code = r'''
from pathlib import Path
from acestep.model_downloader import check_main_model_exists, check_model_exists, ensure_main_model, ensure_dit_model
root = Path(r"%s")
root.mkdir(parents=True, exist_ok=True)
print("main model valid:", check_main_model_exists(root))
if not check_main_model_exists(root):
    ok, msg = ensure_main_model(root, prefer_source="huggingface")
    print(msg)
    if not ok:
        raise RuntimeError(msg)
print("base model valid:", check_model_exists("acestep-v15-base", root))
if not check_model_exists("acestep-v15-base", root):
    ok, msg = ensure_dit_model("acestep-v15-base", root, prefer_source="huggingface")
    print(msg)
    if not ok:
        raise RuntimeError(msg)
print("turbo model valid:", check_model_exists("acestep-v15-turbo", root))
print("ACE-Step checkpoint validation complete")
''' % str(models_dir)
    run([uv, "run", "--project", ace_root, "python", "-c", code], cwd=ace_root, env=env, log=log)


def run_ace(uv: str, ace_root: Path, runner: Path, mode: str, backing: Path, lyrics: Path, plan: Path,
            out: Path, style: str, language: str, seed: int, steps: int, env: dict[str, str], log: Path) -> None:
    if out.exists():
        out.unlink()
    run([
        uv, "run", "--project", ace_root, "python", runner,
        "--ace-root", ace_root, "--backing", backing, "--lyrics", lyrics,
        "--plan", plan, "--out", out, "--style", style, "--language", language,
        "--seed", str(seed), "--steps", str(steps), "--mode", mode,
    ], cwd=ace_root, env=env, log=log)
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
    sep = work_dir / ("separate_" + full_song.stem)
    shutil.rmtree(sep, ignore_errors=True)
    sep.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "--out", sep, full_song], log=log)
    candidates = sorted(sep.rglob("vocals.wav"))
    if not candidates:
        raise RuntimeError("Demucs did not create vocals.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], out)
    rms = audio_rms_db(out)
    if rms < -60.0:
        raise RuntimeError(f"Extracted vocal stem is effectively silent ({rms:.1f} dB RMS)")
    print(f"✅ Extracted singing vocal with Demucs ({rms:.1f} dB RMS)")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Musicm8 singing with ACE-Step and multiple automatic fallbacks.")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--backing", type=Path, required=True)
    p.add_argument("--lyrics", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--style", default="expressive contemporary lead vocal, intimate verses, stronger hook, clear words")
    p.add_argument("--language", default="en")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--ace-root", type=Path, default=Path("/content/ACE-Step-1.5"))
    args = p.parse_args()

    for path in (args.backing, args.lyrics, args.plan):
        if not path.exists():
            raise FileNotFoundError(path)

    work = args.root / "work"
    models_dir = work / "ace_step_models"
    # Keep all lock-heavy caches on Colab's local filesystem. Only finalized model
    # checkpoints live in Drive.
    hf_cache = Path("/content/musicm8_hf_cache")
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
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("uv cache (local):", uv_cache)
    print("uv Python (local):", uv_python)
    print("HF cache (local):", hf_cache)
    print("ACE-Step models (Drive):", models_dir)

    ensure_environment(uv, args.ace_root, env, log)
    ensure_models(uv, args.ace_root, models_dir, env, log)

    runner = args.repo / "ace_step_vocal_runner.py"
    status = {
        "format": "musicm8-vocal-backend-status-v4",
        "backend": "ACE-Step-1.5",
        "primary": "base-lego-vocals",
        "fallbacks": ["base-cover+demucs", "turbo-text2music+demucs"],
        "success": False,
        "method": None,
        "errors": [],
        "log": str(log),
        "uv_cache": str(uv_cache),
        "uv_python": str(uv_python),
        "hf_cache": str(hf_cache),
    }

    try:
        print("\n🎤 METHOD 1/3: ACE-Step BASE Lego vocals")
        run_ace(uv, args.ace_root, runner, "lego", args.backing, args.lyrics, args.plan,
                args.out, args.style, args.language, args.seed, args.steps, env, log)
        status["success"], status["method"] = True, "lego"
    except Exception as exc:
        status["errors"].append(f"lego: {exc}")
        print(f"⚠️ Lego failed:\n{exc}")

    if not status["success"]:
        cover_song = args.out.parent / "ace_cover_with_vocals.wav"
        try:
            print("\n🎤 METHOD 2/3: ACE-Step BASE cover + Demucs")
            run_ace(uv, args.ace_root, runner, "cover", args.backing, args.lyrics, args.plan,
                    cover_song, args.style, args.language, args.seed + 97, max(args.steps, 28), env, log)
            extract_vocals_with_demucs(cover_song, args.out, args.out.parent, log)
            status["success"], status["method"] = True, "cover+demucs"
            status["cover_song"] = str(cover_song)
        except Exception as exc:
            status["errors"].append(f"cover+demucs: {exc}")
            print(f"⚠️ Cover fallback failed:\n{exc}")

    # If the base model path still fails on a T4, fall back to the smaller/faster
    # turbo text-to-music route. It writes a vocalized song from the same lyrics,
    # BPM and key; Demucs then isolates only the singer for Musicm8's mix.
    if not status["success"]:
        turbo_song = args.out.parent / "ace_turbo_vocal_song.wav"
        try:
            print("\n🎤 METHOD 3/3: ACE-Step TURBO lyric song + Demucs")
            run_ace(uv, args.ace_root, runner, "turbo", args.backing, args.lyrics, args.plan,
                    turbo_song, args.style, args.language, args.seed + 211, 8, env, log)
            extract_vocals_with_demucs(turbo_song, args.out, args.out.parent, log)
            status["success"], status["method"] = True, "turbo+demucs"
            status["turbo_song"] = str(turbo_song)
        except Exception as exc:
            status["errors"].append(f"turbo+demucs: {exc}")
            print(f"❌ Turbo vocal fallback failed:\n{exc}")

    status["output"] = str(args.out) if args.out.exists() else None
    status["rms_db"] = round(audio_rms_db(args.out), 3) if args.out.exists() else None
    status_path = args.out.parent / "vocal_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    if not status["success"] or not args.out.exists():
        tail = "\n".join(log.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]) if log.exists() else ""
        print("\n================ VOCAL BACKEND FAILURE ================")
        print(tail)
        print("=======================================================")
        raise RuntimeError(f"All 3 singing methods failed. Exact log: {log}")

    print("\n✅ Musicm8 singing created")
    print("Method:", status["method"])
    print("Vocal:", args.out)
    print("Status:", status_path)


if __name__ == "__main__":
    main()
