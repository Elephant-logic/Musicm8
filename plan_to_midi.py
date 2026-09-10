from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi

SCALE_INTERVALS = {"major":[0,2,4,5,7,9,11],"minor":[0,2,3,5,7,8,10]}

def clamp(v: float, lo: float, hi: float) -> float: return max(lo,min(hi,float(v)))

def key_note(root:int, degree:int, octave:int, mode:str)->int:
    scale=SCALE_INTERVALS.get(mode,SCALE_INTERVALS["minor"]); degree=max(1,min(7,int(degree)))
    return 12*(octave+1)+int(root)+scale[degree-1]

def diatonic_triad(root:int, degree:int, octave:int, mode:str)->list[int]:
    scale=SCALE_INTERVALS.get(mode,SCALE_INTERVALS["minor"]); i=max(0,min(6,int(degree)-1)); notes=[]
    for add in (0,2,4):
        j=i+add; notes.append(12*(octave+1)+int(root)+scale[j%7]+12*(j//7))
    return notes

def add_note(inst,pitch,start,end,velocity):
    if end<=start:return
    inst.notes.append(pretty_midi.Note(velocity=int(np.clip(velocity,1,127)),pitch=int(np.clip(pitch,0,127)),start=max(0.0,float(start)),end=max(float(start)+0.01,float(end))))

def section_energy(plan:dict[str,Any])->list[float]:
    bars=int(plan["bars"]); result=[]
    for sec in plan.get("sections",[]): result.extend([float(sec.get("energy",0.6))]*int(sec.get("bars",1)))
    if not result: result=[0.65]*bars
    if len(result)<bars: result.extend([result[-1]]*(bars-len(result)))
    return result[:bars]

def style_patterns(style:str)->dict[str,list[int]]:
    patterns={
      "house":{"kick":[0,4,8,12],"snare":[4,12],"hat":[2,6,10,14],"ghost":[]},
      "techno":{"kick":[0,4,8,12],"snare":[4,12],"hat":[2,3,6,7,10,11,14,15],"ghost":[]},
      "uk_garage":{"kick":[0,6,10],"snare":[4,12],"hat":[2,5,7,10,14],"ghost":[15]},
      "dnb":{"kick":[0,10],"snare":[4,12],"hat":[0,2,4,6,8,10,12,14],"ghost":[7,15]},
      "trap":{"kick":[0,7,11],"snare":[8],"hat":list(range(0,16,2)),"ghost":[13,14,15]},
      "hiphop":{"kick":[0,6,10],"snare":[4,12],"hat":[0,2,4,6,8,10,12,14],"ghost":[]},
      "ambient":{"kick":[0],"snare":[12],"hat":[6,14],"ghost":[]},
      "electronic":{"kick":[0,8],"snare":[4,12],"hat":[2,6,10,14],"ghost":[]}}
    return patterns.get(style,patterns["electronic"])

def timing(step:int,step_len:float,swing:float)->float:return step*step_len+(step_len*swing if step%2 else 0.0)

def make_drums(plan,bpm,rng):
    inst=pretty_midi.Instrument(program=0,is_drum=True,name="drums"); beat=60.0/bpm; step_len=beat/4; bar_len=beat*4
    pat=style_patterns(plan["style"]); groove=plan.get("groove",{}); swing=clamp(groove.get("swing",0),0,0.35); density=clamp(groove.get("drum_density",0.75),0.1,1); energies=section_energy(plan)
    for bar,energy in enumerate(energies):
        base=bar*bar_len; local=clamp(density*(0.45+0.7*energy),0.1,1)
        for s in pat["kick"]:
            if rng.random()<=min(1,local+0.18):
                t=base+timing(s,step_len,swing); add_note(inst,36,t,t+0.10,78+int(38*energy))
        for s in pat["snare"]:
            if rng.random()<=min(1,local+0.25):
                t=base+timing(s,step_len,swing); add_note(inst,38,t,t+0.09,72+int(40*energy))
        for s in pat["hat"]:
            if rng.random()<=local:
                t=base+timing(s,step_len,swing); add_note(inst,46 if s in {6,14} and energy>0.75 else 42,t,t+0.05,42+int(42*energy))
        for s in pat["ghost"]:
            if energy>0.65 and rng.random()<0.35*density:
                t=base+timing(s,step_len,swing); add_note(inst,38 if plan["style"]!="trap" else 42,t,t+0.05,34+int(28*energy))
        if energy>0.82 and (bar+1)%8==0:
            for s in (13,14,15):
                t=base+timing(s,step_len,swing); add_note(inst,42 if s<15 else 38,t,t+0.05,58+8*(s-13))
    return inst

def bass_positions(style:str)->list[int]:
    return {"uk_garage":[0,3,6,10,13],"house":[0,4,8,12],"techno":[0,3,6,8,11,14],"dnb":[0,3,7,10,13],"trap":[0,5,9,12],"hiphop":[0,4,7,11],"ambient":[0,8],"electronic":[0,4,10,13]}.get(style,[0,4,10,13])

def make_bass(plan,bpm,rng):
    inst=pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Synth Bass 1"),name="bass"); beat=60/bpm; step_len=beat/4; bar_len=beat*4
    root=int(plan["key_root"]); mode=plan["mode"]; progression=plan["progression_degrees"]; density=clamp(plan.get("groove",{}).get("bass_density",0.6),0.1,1); swing=clamp(plan.get("groove",{}).get("swing",0),0,0.35); positions=bass_positions(plan["style"])
    for bar,energy in enumerate(section_energy(plan)):
        degree=int(progression[bar%len(progression)]); root_pitch=key_note(root,degree,1,mode); base=bar*bar_len
        for j,pos in enumerate(positions):
            chance=clamp(density*(0.45+0.8*energy),0.15,1)
            if j==0 or rng.random()<chance:
                pitch=root_pitch
                if j>0 and rng.random()<0.22:pitch+=7 if rng.random()<0.65 else 12
                start=base+timing(pos,step_len,swing); dur=2 if plan["style"] in {"uk_garage","dnb","trap"} else 3
                add_note(inst,pitch,start,min(base+bar_len,start+dur*step_len*0.92),70+int(38*energy))
    return inst

def make_chords(plan,bpm):
    inst=pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Electric Piano 1"),name="chords"); beat=60/bpm; bar_len=beat*4
    root=int(plan["key_root"]); mode=plan["mode"]; prog=plan["progression_degrees"]; density=clamp(plan.get("groove",{}).get("chord_density",0.55),0.1,1)
    for bar,energy in enumerate(section_energy(plan)):
        degree=int(prog[bar%len(prog)]); chord=diatonic_triad(root,degree,3,mode); start=bar*bar_len
        slots=[(start,start+bar_len*0.46),(start+bar_len*0.5,start+bar_len*0.96)] if density>0.65 and energy>0.6 else [(start,start+bar_len*0.92)]
        for a,b in slots:
            for pitch in chord:add_note(inst,pitch,a,b,45+int(36*energy))
    return inst

def build_motif(root,mode,rng):
    scale=SCALE_INTERVALS[mode]; degrees=[0,2,4,1,3,2,0,4]; rng.shuffle(degrees); return [12*5+root+scale[d%7] for d in degrees[:8]]

def make_melody(plan,bpm,rng):
    inst=pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Lead 2 (sawtooth)"),name="melody"); beat=60/bpm; step_len=beat/4; bar_len=beat*4
    root=int(plan["key_root"]); mode=plan["mode"]; density=clamp(plan.get("groove",{}).get("melody_density",0.4),0.05,0.95); swing=clamp(plan.get("groove",{}).get("swing",0),0,0.35); motif=build_motif(root,mode,rng); positions=[0,2,4,6,8,10,12,14]
    for bar,energy in enumerate(section_energy(plan)):
        chance=density*(0.25+0.8*energy)
        if energy<0.38 and bar%2:continue
        base=bar*bar_len
        for j,pos in enumerate(positions):
            if rng.random()>chance:continue
            pitch=motif[(j+bar)%len(motif)]
            start=base+timing(pos,step_len,swing); add_note(inst,pitch,start,min(base+bar_len,start+step_len*(1.7 if j%2==0 else 0.9)),46+int(42*energy))
    return inst

def write_project(plan,out,seed):
    bpm=float(plan["bpm"]); rng=random.Random(seed); tracks={"drums":make_drums(plan,bpm,rng),"bass":make_bass(plan,bpm,rng),"chords":make_chords(plan,bpm),"melody":make_melody(plan,bpm,rng)}
    out.mkdir(parents=True,exist_ok=True); stems=out/"midi_stems"; stems.mkdir(parents=True,exist_ok=True); full=pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for name,inst in tracks.items():
        if inst.notes:
            full.instruments.append(inst); one=pretty_midi.PrettyMIDI(initial_tempo=bpm); one.instruments.append(inst); one.write(str(stems/f"{name}.mid"))
    full.write(str(out/"arrangement.mid"))

def main():
    p=argparse.ArgumentParser(description="Render a Musicm8 producer plan into coherent editable MIDI."); p.add_argument("--plan",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--seed",type=int,default=42); args=p.parse_args()
    plan=json.loads(args.plan.read_text(encoding="utf-8")); write_project(plan,args.out,args.seed); print(f"✅ Arrangement MIDI: {args.out/'arrangement.mid'}"); print(f"✅ MIDI stems: {args.out/'midi_stems'}")

if __name__=="__main__":main()
