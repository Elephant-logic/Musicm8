from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


def _clean_lines(lines: Any) -> list[str]:
    if not isinstance(lines, list):
        return []
    out: list[str] = []
    for line in lines:
        text = re.sub(r"\s+", " ", str(line)).strip()
        if text:
            out.append(text[:140])
    return out[:12]


def fallback_lyrics(idea: str, plan: dict[str, Any], language: str) -> dict[str, Any]:
    # A deterministic backup so the full Musicm8 workflow never depends on the LLM succeeding.
    lower = idea.lower()
    if any(x in lower for x in ("love", "relationship", "heart", "missing", "miss ")):
        hook = "I still hear you in the quiet"
        verse = [
            "Late lights fading through the glass",
            "I keep moving but the feeling lasts",
            "Every road turns back somehow",
            "I'm here without you now",
        ]
    elif any(x in lower for x in ("dark", "night", "moody", "garage")):
        hook = "We move where the city can't see us"
        verse = [
            "Streetlights slide across the rain",
            "Low end running through my veins",
            "No one needs to know our names",
            "We disappear and start again",
        ]
    else:
        hook = "Turn the moment into something new"
        verse = [
            "Another night is opening wide",
            "A little noise and a little light",
            "We leave the old words at the door",
            "And make a reason for one more",
        ]

    sections = []
    for sec in plan.get("sections", []):
        name = str(sec.get("name", "section")).lower()
        bars = int(sec.get("bars", 4))
        if "intro" in name or "outro" in name:
            lines: list[str] = []
        elif name in {"drop", "chorus", "final"} or "chorus" in name:
            lines = [hook, hook]
        elif "build" in name or "pre" in name:
            lines = ["Hold it close, don't let it go", "One more breath before we know"]
        elif "break" in name:
            lines = [hook]
        else:
            lines = verse[: max(2, min(4, bars // 2))]
        sections.append({"name": name, "bars": bars, "lines": lines})

    return {
        "format": "musicm8-lyrics-v1",
        "title": "Musicm8 Song",
        "language": language,
        "idea": idea,
        "hook": hook,
        "sections": sections,
    }


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b + 1])
        raise


def generate_lyrics(idea: str, plan: dict[str, Any], model_name: str, device: str, language: str) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    section_spec = [
        {
            "name": str(s.get("name", "section")),
            "bars": int(s.get("bars", 4)),
            "energy": float(s.get("energy", 0.6)),
        }
        for s in plan.get("sections", [])
    ]
    system = """You are Musicm8's lyric writer. Write ORIGINAL song lyrics for a new composition.
Do not imitate or quote any existing song or named artist. Fit the supplied section plan and mood.
Use short singable lines, concrete imagery, natural stress, a memorable repeated hook, and enough repetition for music.
Intro/outro may be instrumental. Choruses/drops should reuse the hook. Avoid stuffing too many words into a line.
Return JSON only with schema:
{"title":string,"language":string,"hook":string,"sections":[{"name":string,"bars":int,"lines":[string,...]}]}"""
    user = json.dumps(
        {
            "idea": idea,
            "style": plan.get("style"),
            "bpm": plan.get("bpm"),
            "key_root": plan.get("key_root"),
            "mode": plan.get("mode"),
            "sections": section_spec,
            "language": language,
        },
        ensure_ascii=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = system + "\nUSER:\n" + user + "\nJSON:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1400,
            do_sample=True,
            temperature=0.62,
            top_p=0.92,
            repetition_penalty=1.08,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _extract_json(text)


def normalize(raw: dict[str, Any], fallback: dict[str, Any], plan: dict[str, Any], language: str) -> dict[str, Any]:
    raw_sections = raw.get("sections", []) if isinstance(raw, dict) else []
    by_name: dict[str, list[dict[str, Any]]] = {}
    for sec in raw_sections if isinstance(raw_sections, list) else []:
        if isinstance(sec, dict):
            by_name.setdefault(str(sec.get("name", "section")).lower(), []).append(sec)

    sections = []
    occurrence: dict[str, int] = {}
    fallback_sections = fallback["sections"]
    for i, planned in enumerate(plan.get("sections", [])):
        name = str(planned.get("name", "section")).lower()
        idx = occurrence.get(name, 0)
        occurrence[name] = idx + 1
        candidates = by_name.get(name, [])
        chosen = candidates[idx] if idx < len(candidates) else None
        lines = _clean_lines(chosen.get("lines", [])) if chosen else []
        if not lines and i < len(fallback_sections):
            lines = list(fallback_sections[i].get("lines", []))
        sections.append({"name": name, "bars": int(planned.get("bars", 4)), "lines": lines})

    hook = re.sub(r"\s+", " ", str(raw.get("hook", fallback["hook"]))).strip()[:160]
    title = re.sub(r"\s+", " ", str(raw.get("title", fallback["title"]))).strip()[:100]
    return {
        "format": "musicm8-lyrics-v1",
        "title": title or fallback["title"],
        "language": language,
        "idea": fallback["idea"],
        "hook": hook or fallback["hook"],
        "sections": sections,
    }


def as_ace_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for sec in payload.get("sections", []):
        lines = [str(x).strip() for x in sec.get("lines", []) if str(x).strip()]
        if not lines:
            parts.append(f"[{str(sec.get('name', 'section')).title()}]\n[Instrumental]")
        else:
            parts.append(f"[{str(sec.get('name', 'section')).title()}]\n" + "\n".join(lines))
    return "\n\n".join(parts).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Write section-aware original lyrics for a Musicm8 producer plan.")
    p.add_argument("--idea", required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="lyrics.json")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--language", default="en")
    p.add_argument("--no-ai", action="store_true")
    args = p.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    fallback = fallback_lyrics(args.idea, plan, args.language)
    raw = None
    if not args.no_ai:
        try:
            print(f"✍️ Loading lyric writer: {args.model}")
            raw = generate_lyrics(args.idea, plan, args.model, args.device, args.language)
            print("✅ AI lyrics received")
        except Exception as exc:
            print(f"⚠️ AI lyric writer failed ({exc}); using original deterministic fallback lyrics.")
    payload = normalize(raw or {}, fallback, plan, args.language)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    txt = as_ace_text(payload)
    args.out.with_suffix(".txt").write_text(txt + "\n", encoding="utf-8")
    print("✅ Lyrics JSON:", args.out)
    print("✅ Lyrics text:", args.out.with_suffix(".txt"))
    print("Hook:", payload["hook"])


if __name__ == "__main__":
    main()
