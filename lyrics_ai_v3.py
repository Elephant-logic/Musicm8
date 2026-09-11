from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


def clean_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def wrap_line(line: str, max_words: int = 6) -> list[str]:
    tokens = re.findall(r"\S+", clean_line(line))
    if not tokens:
        return []
    if len(tokens) <= max_words:
        return [" ".join(tokens)]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        end = min(len(tokens), i + max_words)
        for j in range(end - 1, max(i + 2, end - 3) - 1, -1):
            if re.search(r"[,;:.!?]$", tokens[j]):
                end = j + 1
                break
        out.append(" ".join(tokens[i:end]))
        i = end
    return out


def fallback(plan: dict[str, Any], idea: str, language: str) -> dict[str, Any]:
    hook = "I know we're over / I still can't leave"
    verse = [
        "Your coat is by the doorway",
        "I hear you in the hall",
        "I know I should be leaving",
        "But I don't move at all",
    ]
    build = ["Say it once and mean it", "Tell me this is really done"]
    breakdown = ["I breathe your name in silence", "Then let the silence come"]
    sections = []
    for sec in plan.get("sections", []):
        name = str(sec.get("name", "section")).lower()
        bars = int(sec.get("bars", 4))
        if name in {"intro", "outro", "instrumental"}:
            lines: list[str] = []
        elif name in {"drop", "chorus", "final"} or "chorus" in name:
            lines = ["I know we're over", "I still can't leave", "I let go slowly", "Then pull you back to me"]
        elif "build" in name or "pre" in name:
            lines = build
        elif "break" in name:
            lines = breakdown
        else:
            lines = verse[: max(2, min(4, bars // 2))]
        sections.append({"name": name, "bars": bars, "lines": lines})
    return {"format": "musicm8-lyrics-v3", "title": "Still Can't Leave", "language": language, "idea": idea, "hook": hook, "source": "fallback", "sections": sections}


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b + 1])
        raise


def generate(plan: dict[str, Any], idea: str, model_name: str, device: str, language: str) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sections = [{"name": str(s.get("name", "section")), "bars": int(s.get("bars", 4)), "energy": float(s.get("energy", 0.6))} for s in plan.get("sections", [])]
    system = """You write original contemporary song lyrics for Musicm8.
Write lyrics about the HUMAN STORY and emotion, not the production prompt.
Never put genre/production terminology into the sung lyrics: do not sing words such as garage, UK garage, chords, bass, BPM, synth, drums, production, mix, build, drop, breakdown, key or tempo unless a user explicitly wrote those words as lyrics.
Use natural conversational English, concrete images and emotionally believable lines.
Prioritise intelligibility: 3-6 words per sung line, simple word order, no tongue-twisters, no filler descriptions, no meta commentary.
Use a memorable hook that can repeat. Intro/outro may be instrumental.
Return JSON only with this schema:
{"title":string,"language":string,"hook":string,"sections":[{"name":string,"bars":int,"lines":[string,...]}]}"""
    user = json.dumps({
        "story_idea": idea,
        "style_context_do_not_quote_in_lyrics": plan.get("style"),
        "bpm_context_do_not_quote_in_lyrics": plan.get("bpm"),
        "sections": sections,
        "language": language,
    }, ensure_ascii=False)
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if hasattr(tok, "apply_chat_template") else system + "\n" + user
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1000, do_sample=True, temperature=0.70, top_p=0.90, repetition_penalty=1.10)
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return extract_json(text)


def normalize(raw: dict[str, Any], fb: dict[str, Any], plan: dict[str, Any], language: str) -> dict[str, Any]:
    raw_sections = raw.get("sections", []) if isinstance(raw, dict) else []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for sec in raw_sections if isinstance(raw_sections, list) else []:
        if isinstance(sec, dict):
            buckets.setdefault(str(sec.get("name", "section")).lower(), []).append(sec)
    occurrence: dict[str, int] = {}
    sections = []
    for i, planned in enumerate(plan.get("sections", [])):
        name = str(planned.get("name", "section")).lower()
        idx = occurrence.get(name, 0)
        occurrence[name] = idx + 1
        choices = buckets.get(name, [])
        chosen = choices[idx] if idx < len(choices) else None
        lines = []
        if chosen:
            for line in chosen.get("lines", []) if isinstance(chosen.get("lines", []), list) else []:
                lines.extend(wrap_line(str(line)))
        if not lines and i < len(fb["sections"]):
            lines = [p for line in fb["sections"][i].get("lines", []) for p in wrap_line(line)]
        # Production words are a sign the LLM copied the prompt instead of writing a lyric.
        banned = re.compile(r"\b(uk garage|garage|chords?|bass|bpm|synths?|drums?|production|mix|tempo|key)\b", re.I)
        if any(banned.search(line) for line in lines):
            lines = [p for line in fb["sections"][i].get("lines", []) for p in wrap_line(line)] if i < len(fb["sections"]) else []
        sections.append({"name": name, "bars": int(planned.get("bars", 4)), "lines": lines})
    hook = clean_line(raw.get("hook", fb["hook"]))[:160] if isinstance(raw, dict) else fb["hook"]
    title = clean_line(raw.get("title", fb["title"]))[:100] if isinstance(raw, dict) else fb["title"]
    return {"format": "musicm8-lyrics-v3", "title": title or fb["title"], "language": language, "idea": fb["idea"], "hook": hook or fb["hook"], "source": "ai", "sections": sections}


def parse_user(text: str, plan: dict[str, Any], language: str, idea: str) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    plain: list[str] = []
    title = "Musicm8 Song"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip() or title
        elif line.startswith("[") and line.endswith("]"):
            if current is not None:
                blocks.append(current)
            current = {"name": line[1:-1].strip().lower(), "lines": []}
        elif current is None:
            plain.append(line)
        else:
            current["lines"].append(line)
    if current is not None:
        blocks.append(current)
    by_name: dict[str, list[list[str]]] = {}
    for b in blocks:
        by_name.setdefault(str(b["name"]), []).append(list(b["lines"]))
    vocal_sections = [s for s in plan.get("sections", []) if str(s.get("name", "")).lower() not in {"intro", "outro", "instrumental"}]
    plain_phrases = [p for line in plain for p in wrap_line(line)]
    chunks = []
    if plain_phrases and not blocks:
        n = max(1, len(vocal_sections))
        chunks = [plain_phrases[round(i * len(plain_phrases) / n):round((i + 1) * len(plain_phrases) / n)] for i in range(n)]
    occurrences: dict[str, int] = {}
    vocal_i = 0
    sections = []
    for sec in plan.get("sections", []):
        name = str(sec.get("name", "section")).lower()
        bars = int(sec.get("bars", 4))
        lines: list[str] = []
        if name not in {"intro", "outro", "instrumental"}:
            aliases = [name] + (["chorus", "drop"] if name == "final" else [])
            for alias in aliases:
                idx = occurrences.get(alias, 0)
                q = by_name.get(alias, [])
                if idx < len(q):
                    lines = q[idx]
                    occurrences[alias] = idx + 1
                    break
            if not lines and chunks:
                lines = chunks[min(vocal_i, len(chunks) - 1)]
            vocal_i += 1
        wrapped = [p for line in lines for p in wrap_line(line)]
        sections.append({"name": name, "bars": bars, "lines": wrapped})
    all_lines = [x for s in sections for x in s["lines"]]
    hook = next((s["lines"][0] for s in sections if ("chorus" in s["name"] or s["name"] in {"drop", "final"}) and s["lines"]), all_lines[0] if all_lines else "")
    return {"format": "musicm8-lyrics-v3", "title": title, "language": language, "idea": idea, "hook": hook, "source": "user", "sections": sections}


def as_text(payload: dict[str, Any]) -> str:
    chunks = []
    for sec in payload.get("sections", []):
        lines = [clean_line(x) for x in sec.get("lines", []) if clean_line(x)]
        chunks.append(f"[{str(sec.get('name', 'section')).title()}]\n" + ("\n".join(lines) if lines else "[Instrumental]"))
    return "\n\n".join(chunks).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Write or import singable section-aware lyrics for Musicm8.")
    p.add_argument("--idea", required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--language", default="en")
    p.add_argument("--lyrics-file", type=Path, default=None)
    p.add_argument("--no-ai", action="store_true")
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.lyrics_file is not None and args.lyrics_file.exists() and args.lyrics_file.read_text(encoding="utf-8").strip():
        payload = parse_user(args.lyrics_file.read_text(encoding="utf-8"), plan, args.language, args.idea)
        print("✅ USER LYRICS MODE — preserving your words")
    else:
        fb = fallback(plan, args.idea, args.language)
        raw = None
        if not args.no_ai:
            try:
                print("✍️ Writing natural, singable lyrics:", args.model)
                raw = generate(plan, args.idea, args.model, args.device, args.language)
            except Exception as exc:
                print("⚠️ AI lyric writer failed; using singable fallback:", exc)
        payload = normalize(raw or {}, fb, plan, args.language)
        if raw is None:
            payload["source"] = "fallback"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    text = as_text(payload)
    args.out.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    print("✅ Lyrics:", args.out.with_suffix(".txt"))
    print("\n================ LYRICS USED ================\n" + text + "\n=============================================\n")


if __name__ == "__main__":
    main()
