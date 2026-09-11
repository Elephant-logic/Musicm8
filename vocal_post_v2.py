from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Repair ACE phrase placement, score-lock the vocal, then reject it if pitch is still poor.")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--score", type=Path, required=True)
    p.add_argument("--synced", type=Path, required=True)
    p.add_argument("--quality-report", type=Path, required=True)
    args = p.parse_args()

    source = args.raw
    manifest = args.raw.with_suffix(".json")
    timed = args.raw.parent / "neural_lead_timed.wav"
    if manifest.exists():
        try:
            run([sys.executable, "-u", args.repo / "vocal_phrase_fit.py", "--vocal", args.raw, "--manifest", manifest, "--out", timed])
            source = timed
            print("✅ Using phrase-timing-rebuilt vocal for score lock")
        except Exception as exc:
            print(f"⚠️ Phrase timing rebuild unavailable ({exc}); using raw vocal for score lock")
    else:
        print("ℹ️ No ACE phrase manifest for this fallback vocal; using raw stem")

    run([sys.executable, "-u", args.repo / "vocal_sync.py", "--vocal", source, "--score", args.score, "--out", args.synced])
    run([sys.executable, "-u", args.repo / "vocal_quality_gate.py", "--vocal", args.synced, "--score", args.score, "--report", args.quality_report])
    print("✅ Vocal timing/pitch QA passed")


if __name__ == "__main__":
    main()
