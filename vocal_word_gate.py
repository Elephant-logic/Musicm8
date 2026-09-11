from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import torch


def normalize(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]+\]", " ", text)
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def lcs_recall(expected: list[str], observed: list[str]) -> float:
    if not expected:
        return 1.0
    matcher = difflib.SequenceMatcher(a=expected, b=observed, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / max(1, len(expected))


def main() -> None:
    p = argparse.ArgumentParser(description="Reject a Musicm8 vocal when a lightweight ASR cannot recover enough of the requested lyric words.")
    p.add_argument("--vocal", type=Path, required=True)
    p.add_argument("--lyrics", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--model", default="openai/whisper-tiny.en")
    p.add_argument("--min-recall", type=float, default=0.34)
    args = p.parse_args()

    if not args.vocal.exists():
        raise FileNotFoundError(args.vocal)
    if not args.lyrics.exists():
        raise FileNotFoundError(args.lyrics)

    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if device == 0 else torch.float32
    print("📝 Lyric intelligibility QA using", args.model)
    asr = pipeline("automatic-speech-recognition", model=args.model, device=device, torch_dtype=dtype)
    result = asr(str(args.vocal), chunk_length_s=25, batch_size=4)
    transcript = str(result.get("text", "")).strip()
    expected_text = args.lyrics.read_text(encoding="utf-8", errors="ignore")
    expected = normalize(expected_text)
    observed = normalize(transcript)
    recall = lcs_recall(expected, observed)
    ratio = difflib.SequenceMatcher(a=" ".join(expected), b=" ".join(observed), autojunk=False).ratio() if expected else 1.0
    passed = bool(recall >= float(args.min_recall) and len(observed) >= max(1, int(0.25 * len(expected))))

    payload = {
        "format": "musicm8-vocal-word-qa-v1",
        "pass": passed,
        "expected_words": len(expected),
        "recognized_words": len(observed),
        "ordered_word_recall": round(recall, 4),
        "text_similarity": round(ratio, 4),
        "transcript": transcript,
        "reason": "enough requested lyric words were recoverable" if passed else "too few requested lyric words were intelligible to ASR",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
