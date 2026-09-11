from __future__ import annotations

"""Minimal Musicm8 Colab startup settings.

V10 no longer uses ACE-Step for the lead vocal, so the old automatic ACE source
patching is deliberately gone.  ``sitecustomize`` is imported by Python before
normal script imports; keeping this file tiny prevents a startup hook from
breaking every Musicm8 subprocess.
"""

import os

# Safe, process-wide defaults only.  No file patching, GPU probing or third-party
# imports are allowed here because a failure in sitecustomize prevents Python
# itself from starting the requested Musicm8 stage.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/content/musicm8_mpl_config")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
