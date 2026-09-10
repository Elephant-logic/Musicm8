from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import butter, sosfilt

SR=44100

def clamp(x:Any,lo:float,hi:float,default:float=0.0)->float:
    try:
        v=float(x)
        if math.isfinite(v): return max(lo,min(hi,v))
    except Exception: pass
    return default

def midi_hz(note:int)->float:return 440.0*(2.0**((int(note)-69)/12.0))

def poly_blep(t:np.ndarray,dt:float)->np.ndarray:
    out=np.zeros_like(t); a=t<dt
    if np.any(a):
        x=t[a]/dt; out[a]=x+x-x*x-1.0
    b=t>1.0-dt
    if np.any(b):
        x=(t[b]-1.0)/dt; out[b]=x*x+x+x+1.0
    return out

def oscillator(kind:str,freq:float,n:int,phase0:float=0.0)->np.ndarray:
    if n<=0:return np.zeros(0,dtype=np.float32)
    phase=(phase0+np.arange(n,dtype=np.float64)*freq/SR)%1.0
    if kind=="sine": y=np.sin(2*np.pi*phase)
    elif kind=="triangle": y=2.0*np.abs(2.0*phase-1.0)-1.0
    elif kind=="square":
        y=np.where(phase<0.5,1.0,-1.0); dt=min(0.5,freq/SR); y+=poly_blep(phase,dt); y-=poly_blep((phase+0.5)%1.0,dt)
    else:
        y=2.0*phase-1.0; y-=poly_blep(phase,min(0.5,freq/SR))
    return y.astype(np.float32)

def adsr(n:int,attack:float,decay:float,sustain:float,release:float)->np.ndarray:
    if n<=0:return np.zeros(0,dtype=np.float32)
    a=min(n,max(1,int(attack*SR))); d=min(max(0,n-a),max(1,int(decay*SR))); r=min(max(1,int(release*SR)),max(1,n-a-d)); s=max(0,n-a-d-r)
    env=np.empty(n,dtype=np.float32); pos=0; env[pos:pos+a]=np.linspace(0,1,a,endpoint=False); pos+=a
    if d: env[pos:pos+d]=np.linspace(1,sustain,d,endpoint=False); pos+=d
    if s: env[pos:pos+s]=sustain; pos+=s
    if pos<n: env[pos:]=np.linspace(env[pos-1] if pos else sustain,0,n-pos,endpoint=True)
    return env

def filt(x:np.ndarray,cutoff:float,kind:str)->np.ndarray:
    cutoff=clamp(cutoff,20,SR*0.45,8000); sos=butter(2,cutoff/(SR*0.5),btype=kind,output="sos")
    if x.ndim==1:return sosfilt(sos,x).astype(np.float32)
    return np.vstack([sosfilt(sos,ch) for ch in x]).astype(np.float32)

def saturate(x:np.ndarray,drive:float)->np.ndarray:
    amount=1.0+8.0*clamp(drive,0,1); return (np.tanh(x*amount)/max(np.tanh(amount),1e-6)).astype(np.float32)

def stereo_delay(x:np.ndarray,amount:float)->np.ndarray:
    amount=clamp(amount,0,1)
    if amount<=0:return x
    d=max(1,int((0.012+0.018*amount)*SR)); wet=np.zeros_like(x); wet[0,d:]=x[1,:-d]; wet[1,d:]=x[0,:-d]
    return (x*(1-0.28*amount)+wet*(0.28*amount)).astype(np.float32)

def echo(x:np.ndarray,amount:float,bpm:float)->np.ndarray:
    amount=clamp(amount,0,1)
    if amount<=0:return x
    d=max(1,int((60.0/bpm)*0.75*SR)); y=x.copy(); gain=0.35*amount
    for k in range(1,4):
        dk=d*k
        if dk>=x.shape[-1]:break
        y[:,dk:]+=x[:,:-dk]*(gain**k)
    return y.astype(np.float32)

def reverb(x:np.ndarray,amount:float)->np.ndarray:
    amount=clamp(amount,0,1)
    if amount<=0:return x
    y=x.copy()
    for seconds,gain in ((0.031,0.34),(0.047,0.27),(0.071,0.19),(0.109,0.13)):
        d=int(seconds*SR)
        if d<x.shape[-1]:y[:,d:]+=x[:,:-d]*gain*amount
    y=filt(y,11000-4500*amount,"low"); return (x*(1-0.12*amount)+y*(0.24*amount)).astype(np.float32)

def sonic_for_reference(library:dict[str,Any],ref_id:str|None,role:str)->dict[str,Any]:
    if not ref_id:return {}
    source_role={"chords":"other","melody":"vocals"}.get(role,role)
    for ref in library.get("references",[]):
        if ref.get("id")==ref_id:return ref.get("sonic",{}).get(source_role,{})
    return {}

def patch_from_fingerprint(role:str,f:dict[str,Any],overrides:dict[str,Any])->dict[str,Any]:
    centroid=clamp(f.get("spectral_centroid_hz",1800),100,9000,1800); rolloff=clamp(f.get("rolloff_hz",6000),300,15000,6000); flat=clamp(f.get("flatness",0.04),0,0.5,0.04); sub_ratio=clamp(f.get("sub_ratio",0.15),0,1,0.15); air=clamp(f.get("air_ratio",0.08),0,1,0.08); dynamic=clamp(f.get("dynamic_range_db",10),0,40,10)
    base={
      "drums":{"brightness":clamp(centroid/6500,0.2,1),"drive":clamp(flat*2,0.05,0.65),"room":clamp(dynamic/35,0.05,0.5),"gain":0.85},
      "bass":{"wave":"saw" if centroid>850 else "square","sub":clamp(0.35+sub_ratio*1.2,0.25,0.95),"cutoff":clamp(rolloff*0.42,250,4200),"drive":clamp(0.12+flat*3,0.08,0.7),"attack":0.004,"decay":0.14,"sustain":0.72,"release":0.08,"width":0.03,"gain":0.62},
      "chords":{"wave":"saw","detune":clamp(0.08+air*0.8,0.04,0.35),"cutoff":clamp(rolloff*0.78,1000,11000),"attack":0.025,"decay":0.28,"sustain":0.62,"release":0.32,"reverb":clamp(0.18+dynamic/70,0.15,0.65),"width":0.55,"gain":0.34},
      "melody":{"wave":"saw" if centroid>1500 else "triangle","detune":clamp(0.04+air*0.5,0.02,0.20),"cutoff":clamp(rolloff*0.9,1800,13500),"attack":0.008,"decay":0.18,"sustain":0.58,"release":0.18,"delay":0.18,"reverb":0.22,"width":0.34,"gain":0.28}}[role].copy()
    if role=="bass":
        for k in ("sub","drive","width"):
            if k in overrides:base[k]=clamp(overrides[k],0,1,base[k])
        if "filter_motion" in overrides:base["cutoff"]*=0.7+0.8*clamp(overrides["filter_motion"],0,1,0.3)
    elif role=="drums":
        for k in ("brightness","drive","room"):
            if k in overrides:base[k]=clamp(overrides[k],0,1,base[k])
    else:
        if "brightness" in overrides:base["cutoff"]*=0.55+0.9*clamp(overrides["brightness"],0,1,0.5)
        for k in ("detune","reverb","delay","width"):
            if k in overrides:base[k]=clamp(overrides[k],0,1,base.get(k,0))
    return base

def synth_note(note:int,duration:float,velocity:int,patch:dict[str,Any],role:str)->np.ndarray:
    release=float(patch.get("release",0.1)); n=max(1,int((duration+release)*SR)); freq=midi_hz(note); kind=str(patch.get("wave","saw")); base=oscillator(kind,freq,n); detune=clamp(patch.get("detune",0),0,0.5)
    if detune>0:
        cents=5+17*detune; ratio=2**(cents/1200); base=0.58*base+0.21*oscillator(kind,freq*ratio,n,0.17)+0.21*oscillator(kind,freq/ratio,n,0.41)
    if role=="bass":
        sub=clamp(patch.get("sub",0.5),0,1); base=base*(1-0.45*sub)+oscillator("sine",freq,n,0.25)*(0.7*sub)
    env=adsr(n,float(patch.get("attack",0.01)),float(patch.get("decay",0.18)),float(patch.get("sustain",0.65)),release)
    return (base*env*(velocity/127.0)).astype(np.float32)

def kick(n:int,velocity:int,brightness:float)->np.ndarray:
    t=np.arange(n,dtype=np.float32)/SR; f0,f1=145+25*brightness,43.0; phase=2*np.pi*(f1*t+(f0-f1)*(1-np.exp(-t/0.028))*0.028); body=np.sin(phase)*np.exp(-t/0.19); click=np.random.default_rng(0).normal(0,1,n).astype(np.float32)*np.exp(-t/0.008)
    return ((body*0.95+click*0.05*brightness)*velocity/127.0).astype(np.float32)

def snare(n:int,velocity:int,brightness:float)->np.ndarray:
    t=np.arange(n,dtype=np.float32)/SR; noise=np.random.default_rng(1).normal(0,1,n).astype(np.float32); tone=np.sin(2*np.pi*185*t)+0.45*np.sin(2*np.pi*330*t); env=np.exp(-t/(0.11+0.07*(1-brightness)))
    return ((0.72*noise+0.28*tone)*env*0.55*velocity/127.0).astype(np.float32)

def hat(n:int,velocity:int,brightness:float,open_hat:bool)->np.ndarray:
    t=np.arange(n,dtype=np.float32)/SR; noise=np.random.default_rng(2 if open_hat else 3).normal(0,1,n).astype(np.float32); y=noise*np.exp(-t/(0.18 if open_hat else 0.045)); y=filt(y,4500+2500*brightness,"high"); return (y*0.22*velocity/127.0).astype(np.float32)

def render_instrument(inst,role,patch,total_n):
    mono=np.zeros(total_n,dtype=np.float32)
    if role=="drums":
        bright=clamp(patch.get("brightness",0.5),0,1)
        for note in inst.notes:
            start=int(note.start*SR)
            voice=kick(int(0.55*SR),note.velocity,bright) if note.pitch in (35,36) else snare(int(0.42*SR),note.velocity,bright) if note.pitch in (38,40) else hat(int((0.38 if note.pitch==46 else 0.18)*SR),note.velocity,bright,note.pitch==46)
            end=min(total_n,start+len(voice))
            if start<total_n:mono[start:end]+=voice[:end-start]
    else:
        for note in inst.notes:
            start=int(note.start*SR); voice=synth_note(note.pitch,max(0.03,note.end-note.start),note.velocity,patch,role); end=min(total_n,start+len(voice))
            if start<total_n:mono[start:end]+=voice[:end-start]
        mono=filt(mono,float(patch.get("cutoff",8000)),"low")
    mono=saturate(mono,float(patch.get("drive",0))); mono*=float(patch.get("gain",0.5)); stereo=np.vstack([mono.copy(),mono.copy()]); width=clamp(patch.get("width",0),0,1)
    if width>0:stereo=stereo_delay(stereo,width)
    return stereo.astype(np.float32)

def render_project(plan_path:Path,midi_path:Path,library_path:Path,out_dir:Path)->Path:
    plan=json.loads(plan_path.read_text(encoding="utf-8")); library=json.loads(library_path.read_text(encoding="utf-8")); pm=pretty_midi.PrettyMIDI(str(midi_path)); bpm=float(plan["bpm"]); total_s=max(pm.get_end_time()+1.5,float(plan["bars"])*4*60/bpm+1); total_n=int(total_s*SR); out_dir.mkdir(parents=True,exist_ok=True); stems_dir=out_dir/"audio_stems"; stems_dir.mkdir(parents=True,exist_ok=True)
    by_name={(inst.name or "").lower():inst for inst in pm.instruments}; mix=np.zeros((2,total_n),dtype=np.float32); patches={}
    for role in ("drums","bass","chords","melody"):
        inst=by_name.get(role)
        if inst is None:continue
        ref_id=plan.get("references",{}).get(role); features=sonic_for_reference(library,ref_id,role); patch=patch_from_fingerprint(role,features,plan.get("sound_design",{}).get(role,{})); patches[role]={"reference_id":ref_id,"reference_features":features,"patch":patch}; audio=render_instrument(inst,role,patch,total_n)
        if role=="chords":audio=reverb(audio,float(patch.get("reverb",0)))
        elif role=="melody":audio=reverb(echo(audio,float(patch.get("delay",0)),bpm),float(patch.get("reverb",0)))
        elif role=="drums":audio=reverb(audio,float(patch.get("room",0))*0.45)
        sf.write(stems_dir/f"{role}.wav",audio.T,SR,subtype="PCM_24"); mix+=audio
    mix=saturate(mix,float(plan.get("mix",{}).get("master_drive",0.05))); peak=float(np.max(np.abs(mix)))+1e-9; target=10**(clamp(plan.get("mix",{}).get("target_peak_db",-1),-6,-0.1,-1)/20); mix*=target/peak
    master=out_dir/"master.wav"; sf.write(master,mix.T,SR,subtype="PCM_24"); (out_dir/"synth_patches.json").write_text(json.dumps(patches,indent=2),encoding="utf-8"); return master

def main():
    p=argparse.ArgumentParser(description="Render Musicm8 MIDI with its own synth + FX engine."); p.add_argument("--plan",type=Path,required=True); p.add_argument("--midi",type=Path,required=True); p.add_argument("--references",type=Path,required=True); p.add_argument("--out",type=Path,required=True); args=p.parse_args(); master=render_project(args.plan,args.midi,args.references,args.out); print(f"✅ Musicm8 synth master: {master}"); print(f"✅ Audio stems: {args.out/'audio_stems'}"); print(f"✅ Synth patches: {args.out/'synth_patches.json'}")

if __name__=="__main__":main()
