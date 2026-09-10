from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

STYLES = {"uk_garage", "house", "techno", "dnb", "trap", "hiphop", "ambient", "electronic"}
MODES = {"major", "minor"}


def clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return max(lo, min(hi, x))
    except Exception:
        pass
    return default


def infer_style(idea: str) -> str:
    s = idea.lower()
    if "drum and bass" in s or "dnb" in s or "jungle" in s:
        return "dnb"
    if "garage" in s or "2-step" in s or "2 step" in s:
        return "uk_garage"
    if "techno" in s:
        return "techno"
    if "house" in s:
        return "house"
    if "trap" in s:
        return "trap"
    if "hip hop" in s or "hip-hop" in s or "boom bap" in s:
        return "hiphop"
    if "ambient" in s or "drone" in s:
        return "ambient"
    return "electronic"


def default_bpm(style: str, refs: list[dict[str, Any]]) -> float:
    targets = {"uk_garage":132,"house":126,"techno":136,"dnb":174,"trap":142,"hiphop":92,"ambient":90,"electronic":120}
    target = targets.get(style, 120)
    bpms = []
    for r in refs:
        try:
            x = float(r.get("bpm", target))
            if 50 <= x <= 220:
                bpms.append(x)
        except Exception:
            pass
    if not bpms:
        return float(target)
    nearest = min(bpms, key=lambda x: abs(x - target))
    return round(0.6 * target + 0.4 * nearest, 1)


def parse_key(text: str | None) -> tuple[int, str]:
    names = {"C":0,"C#":1,"DB":1,"D":2,"D#":3,"EB":3,"E":4,"F":5,"F#":6,"GB":6,"G":7,"G#":8,"AB":8,"A":9,"A#":10,"BB":10,"B":11}
    if not text:
        return 0, "minor"
    m = re.match(r"\s*([A-Ga-g])([#b]?)(?:\s+|:)?(major|minor|maj|min|m)?", text)
    if not m:
        return 0, "minor"
    root = (m.group(1).upper() + m.group(2)).upper()
    mode = (m.group(3) or "minor").lower()
    return names.get(root, 0), ("minor" if mode in {"minor","min","m"} else "major")


def choose_reference(refs: list[dict[str, Any]], role: str, bpm: float) -> str | None:
    candidates = [r for r in refs if role in r.get("sonic", {}) or role in r.get("musical", {})] or refs
    if not candidates:
        return None
    def score(r: dict[str, Any]) -> float:
        tempo = abs(float(r.get("bpm", bpm)) - bpm) / 80.0
        sonic = r.get("sonic", {}).get(role, {})
        musical = r.get("musical", {}).get(role, {})
        richness = float(sonic.get("dynamic_range_db", 0)) / 30.0 + min(1.0, float(musical.get("notes_per_second", 0)) / 6.0)
        return richness - tempo
    return max(candidates, key=score).get("id")


def fallback_plan(idea: str, library: dict[str, Any], bars: int = 32) -> dict[str, Any]:
    refs = library.get("references", [])
    style = infer_style(idea)
    bpm = default_bpm(style, refs)
    keys = [r.get("key") for r in refs if r.get("key")]
    root, mode = parse_key(keys[0] if keys else "F minor")
    if "major" in idea.lower(): mode = "major"
    if "minor" in idea.lower() or "dark" in idea.lower(): mode = "minor"
    bars = int(max(8, min(128, bars)))
    q = max(4, bars // 8)
    sections = [
        {"name":"intro","bars":q,"energy":0.30},
        {"name":"verse","bars":q,"energy":0.58},
        {"name":"build","bars":q,"energy":0.74},
        {"name":"drop","bars":q*2,"energy":0.96},
        {"name":"breakdown","bars":q,"energy":0.42},
        {"name":"final","bars":max(1,bars-q*6),"energy":1.0},
    ]
    total = sum(s["bars"] for s in sections)
    sections[-1]["bars"] += bars - total
    progression = [1,6,3,7] if mode == "minor" else [1,5,6,4]
    if style in {"house","techno"}: progression = [1,6,4,5]
    refs_by_role = {role: choose_reference(refs, role, bpm) for role in ("drums","bass","other","vocals")}
    return {
        "format":"musicm8-producer-plan-v1","title":"Musicm8 idea","idea":idea,"style":style,
        "bpm":bpm,"key_root":root,"mode":mode,"bars":bars,"sections":sections,
        "progression_degrees":progression,
        "groove":{"swing":0.16 if style=="uk_garage" else (0.08 if style in {"hiphop","trap"} else 0.0),"drum_density":0.78,"bass_density":0.62,"chord_density":0.52,"melody_density":0.42},
        "references":{"drums":refs_by_role["drums"],"bass":refs_by_role["bass"],"chords":refs_by_role["other"],"melody":refs_by_role["vocals"]},
        "sound_design":{
            "drums":{"brightness":0.55,"drive":0.15,"room":0.10},
            "bass":{"sub":0.72,"drive":0.35,"filter_motion":0.35,"width":0.05},
            "chords":{"brightness":0.48,"detune":0.18,"reverb":0.35,"width":0.65},
            "melody":{"brightness":0.62,"detune":0.08,"delay":0.22,"reverb":0.28}},
        "mix":{"master_drive":0.08,"reverb_send":0.16,"target_peak_db":-1.0},
    }


def compact_references(library: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in library.get("references", []):
        sonic = {}
        for role, f in r.get("sonic", {}).items():
            sonic[role] = {k:f.get(k) for k in ("spectral_centroid_hz","rolloff_hz","flatness","sub_ratio","bass_ratio","high_mid_ratio","air_ratio","onset_rate","dynamic_range_db")}
        musical = {}
        for role, f in r.get("musical", {}).items():
            musical[role] = {k:f.get(k) for k in ("notes_per_second","median_pitch","pitch_range")}
        out.append({"id":r.get("id"),"name":r.get("name"),"bpm":r.get("bpm"),"key":r.get("key"),"sonic":sonic,"musical":musical,"codec_token_count":r.get("codec_token_count",0)})
    return out


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b+1])
        raise


def normalize_plan(plan: dict[str, Any], fallback: dict[str, Any], valid_refs: set[str]) -> dict[str, Any]:
    out = dict(fallback)
    for key in ("title","style","bpm","key_root","mode","bars","sections","progression_degrees","groove","references","sound_design","mix"):
        if key in plan: out[key] = plan[key]
    out["format"] = "musicm8-producer-plan-v1"; out["idea"] = fallback["idea"]
    out["style"] = out.get("style") if out.get("style") in STYLES else fallback["style"]
    out["bpm"] = round(clamp(out.get("bpm"),55,200,fallback["bpm"]),2)
    out["key_root"] = int(clamp(out.get("key_root"),0,11,fallback["key_root"]))
    out["mode"] = out.get("mode") if out.get("mode") in MODES else fallback["mode"]
    out["bars"] = int(clamp(out.get("bars"),8,128,fallback["bars"]))
    prog = []
    for x in out.get("progression_degrees", []):
        try:
            d=int(x)
            if 1<=d<=7: prog.append(d)
        except Exception: pass
    out["progression_degrees"] = prog[:8] or fallback["progression_degrees"]
    sections=[]; remaining=out["bars"]
    for item in out.get("sections", []):
        if remaining<=0 or not isinstance(item,dict): break
        n=int(clamp(item.get("bars"),1,remaining,min(8,remaining)))
        sections.append({"name":str(item.get("name","section"))[:32],"bars":n,"energy":round(clamp(item.get("energy"),0.05,1.0,0.6),3)})
        remaining-=n
    if not sections:
        sections=fallback["sections"]; remaining=out["bars"]-sum(s["bars"] for s in sections)
    if remaining!=0: sections[-1]["bars"]+=remaining
    out["sections"]=sections
    refs=dict(fallback["references"])
    if isinstance(out.get("references"),dict):
        for role,rid in out["references"].items():
            if rid in valid_refs: refs[role]=rid
    out["references"]=refs
    return out


def run_local_ai(idea: str, library: dict[str, Any], fallback: dict[str, Any], model_name: str, device: str) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    refs = compact_references(library)
    system = '''You are Musicm8 Producer Brain, an expert composer, arranger, sound designer and mix engineer.
Create a coherent NEW song plan from the user's idea and the supplied reference analyses.
References are inspiration for groove/timbre/production traits; do not copy melodies. Return JSON only.
Use a stable key, deliberate chord progression, repeated motifs, section-level energy and genre-appropriate groove.
Schema: {"title":string,"style":"uk_garage|house|techno|dnb|trap|hiphop|ambient|electronic","bpm":55..200,"key_root":0..11,"mode":"major|minor","bars":8..128,"sections":[{"name":string,"bars":int,"energy":0..1}],"progression_degrees":[1..7],"groove":{"swing":0..0.35,"drum_density":0..1,"bass_density":0..1,"chord_density":0..1,"melody_density":0..1},"references":{"drums":"ref_NNN","bass":"ref_NNN","chords":"ref_NNN","melody":"ref_NNN"},"sound_design":{"drums":{"brightness":0..1,"drive":0..1,"room":0..1},"bass":{"sub":0..1,"drive":0..1,"filter_motion":0..1,"width":0..1},"chords":{"brightness":0..1,"detune":0..1,"reverb":0..1,"width":0..1},"melody":{"brightness":0..1,"detune":0..1,"delay":0..1,"reverb":0..1}},"mix":{"master_drive":0..1,"reverb_send":0..1,"target_peak_db":-6..-0.1}}.
Section bars must add to bars. Choose only supplied reference IDs.'''
    user = json.dumps({"idea":idea,"references":refs,"fallback_hint":fallback}, ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    messages=[{"role":"system","content":system},{"role":"user","content":user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if hasattr(tokenizer,"apply_chat_template") else system+"\nUSER:\n"+user+"\nJSON:\n"
    inputs=tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output=model.generate(**inputs,max_new_tokens=1200,do_sample=True,temperature=0.45,top_p=0.9,repetition_penalty=1.05)
    new_tokens=output[0, inputs["input_ids"].shape[1]:]
    text=tokenizer.decode(new_tokens, skip_special_tokens=True)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return extract_json(text)


def main() -> None:
    p=argparse.ArgumentParser(description="Musicm8 local AI producer brain.")
    p.add_argument("--idea",required=True); p.add_argument("--references",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--model",default="Qwen/Qwen2.5-1.5B-Instruct"); p.add_argument("--device",default="cuda"); p.add_argument("--bars",type=int,default=32); p.add_argument("--no-ai",action="store_true")
    args=p.parse_args()
    library=json.loads(args.references.read_text(encoding="utf-8")); fallback=fallback_plan(args.idea,library,args.bars); valid_refs={str(r.get("id")) for r in library.get("references",[])}
    raw=None
    if not args.no_ai:
        try:
            print(f"🧠 Loading producer brain: {args.model}"); raw=run_local_ai(args.idea,library,fallback,args.model,args.device); print("✅ AI plan received")
        except Exception as exc:
            print(f"⚠️ Local AI planner failed ({exc}); using deterministic music-theory fallback.")
    plan=normalize_plan(raw or fallback,fallback,valid_refs); args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(plan,indent=2),encoding="utf-8")
    print(f"✅ Producer plan: {args.out}"); print(f"{plan['style']} | {plan['bpm']} BPM | root={plan['key_root']} {plan['mode']} | {plan['bars']} bars"); print("References:",plan["references"])


if __name__ == "__main__": main()
