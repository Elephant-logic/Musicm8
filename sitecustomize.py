from __future__ import annotations

"""Musicm8 runtime compatibility hooks for Colab.

Python imports ``sitecustomize`` automatically during interpreter startup when
this repository is on ``sys.path``.  ACE-Step is launched with the Musicm8
vocal runner as the script, so this hook can repair a couple of Colab/T4
runtime incompatibilities before ACE-Step itself is imported.
"""

import os
import subprocess
from pathlib import Path


def _gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
    except Exception:
        return ""


def _patch_ace_cuda_dtype() -> bool:
    target = Path(
        "/content/ACE-Step-1.5/acestep/core/generation/handler/"
        "init_service_orchestrator.py"
    )
    if not target.exists():
        return False

    text = target.read_text(encoding="utf-8")
    marker = "Musicm8 CUDA dtype override"
    if marker in text:
        return True

    old = '''            elif resolved_device == "cuda":\n                if gpu_config.cuda_supports_bfloat16():\n                    self.dtype = torch.bfloat16\n                else:\n                    self.dtype = torch.float16\n                    logger.info(\n                        "[initialize_service] Pre-Ampere CUDA detected: "\n                        "using float16 instead of bfloat16."\n                    )\n'''
    new = '''            elif resolved_device == "cuda":\n                # Musicm8 CUDA dtype override: ACE-Step float16 diffusion can\n                # overflow on pre-Ampere GPUs such as Tesla T4.\n                forced_dtype = os.environ.get("ACESTEP_DTYPE", "").strip().lower()\n                forced_map = {\n                    "float32": torch.float32,\n                    "float16": torch.float16,\n                    "bfloat16": torch.bfloat16,\n                }\n                if forced_dtype in forced_map:\n                    self.dtype = forced_map[forced_dtype]\n                    logger.info(\n                        f"[initialize_service] Forced CUDA dtype from ACESTEP_DTYPE: {self.dtype}"\n                    )\n                elif gpu_config.cuda_supports_bfloat16():\n                    self.dtype = torch.bfloat16\n                else:\n                    self.dtype = torch.float16\n                    logger.info(\n                        "[initialize_service] Pre-Ampere CUDA detected: "\n                        "using float16 instead of bfloat16."\n                    )\n'''
    if old not in text:
        return False

    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def _patch_musicm8_runner_offload() -> bool:
    target = Path("/content/Musicm8/ace_step_vocal_runner.py")
    if not target.exists():
        return False
    text = target.read_text(encoding="utf-8")
    old = 'offload_to_cpu=(low_vram and args.mode != "turbo"),'
    new = 'offload_to_cpu=low_vram,'
    if old in text:
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def _apply() -> None:
    gpu = _gpu_name()
    if "T4" not in gpu.upper():
        return

    # ACE-Step's own NaN diagnostic recommends float32 on pre-Ampere GPUs.
    os.environ["ACESTEP_DTYPE"] = "float32"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/content/musicm8_mpl_config")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    ace_ok = _patch_ace_cuda_dtype()
    runner_ok = _patch_musicm8_runner_offload()
    print(
        "[Musicm8 T4-safe] ACE-Step float32 enabled; "
        f"ACE patch={ace_ok}, CPU-offload patch={runner_ok}",
        flush=True,
    )


_apply()
