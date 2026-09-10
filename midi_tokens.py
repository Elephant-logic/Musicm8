from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi

PAD = 0
BOS = 1
EOS = 2
BAR = 3
POS_BASE = 4
N_POS = 16
TRACK_BASE = POS_BASE + N_POS
TRACKS = {"drums": 0, "bass": 1, "chords": 2, "melody": 3}
INV_TRACKS = {v: k for k, v in TRACKS.items()}
PITCH_BASE = TRACK_BASE + len(TRACKS)
N_PITCH = 128
DUR_BASE = PITCH_BASE + N_PITCH
N_DUR = 32
VEL_BASE = DUR_BASE + N_DUR
N_VEL = 16
VOCAB_SIZE = VEL_BASE + N_VEL


@dataclass
class MidiEvent:
    bar: int
    pos: int
    track: int
    pitch: int
    dur: int
    vel: int


def token_ranges() -> dict[str, range]:
    return {
        "pos": range(POS_BASE, POS_BASE + N_POS),
        "track": range(TRACK_BASE, TRACK_BASE + len(TRACKS)),
        "pitch": range(PITCH_BASE, PITCH_BASE + N_PITCH),
        "dur": range(DUR_BASE, DUR_BASE + N_DUR),
        "vel": range(VEL_BASE, VEL_BASE + N_VEL),
    }


def event_to_tokens(e: MidiEvent) -> list[int]:
    return [
        POS_BASE + int(np.clip(e.pos, 0, N_POS - 1)),
        TRACK_BASE + int(np.clip(e.track, 0, len(TRACKS) - 1)),
        PITCH_BASE + int(np.clip(e.pitch, 0, 127)),
        DUR_BASE + int(np.clip(e.dur - 1, 0, N_DUR - 1)),
        VEL_BASE + int(np.clip(e.vel, 0, N_VEL - 1)),
    ]


def _track_id(inst: pretty_midi.Instrument) -> int | None:
    name = (inst.name or "").lower()
    if inst.is_drum or "drum" in name:
        return TRACKS["drums"]
    for key in ("bass", "chords", "melody"):
        if key in name:
            return TRACKS[key]
    # Infer from program when track names were stripped by another DAW.
    if 32 <= inst.program <= 39:
        return TRACKS["bass"]
    return TRACKS["melody"]


def midi_to_events(path: str | Path, bars: int | None = None) -> tuple[list[MidiEvent], float]:
    pm = pretty_midi.PrettyMIDI(str(path))
    tempo_times, tempi = pm.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) else 120.0
    beat = 60.0 / max(bpm, 1e-6)
    bar_len = beat * 4.0
    step = beat / 4.0
    events: list[MidiEvent] = []
    for inst in pm.instruments:
        track = _track_id(inst)
        if track is None:
            continue
        for note in inst.notes:
            bar = int(max(0.0, note.start) // bar_len)
            if bars is not None and bar >= bars:
                continue
            local = max(0.0, note.start - bar * bar_len)
            pos = int(np.clip(round(local / step), 0, N_POS - 1))
            dur = int(np.clip(round(max(note.end - note.start, step) / step), 1, N_DUR))
            vel = int(np.clip(note.velocity // 8, 0, N_VEL - 1))
            events.append(MidiEvent(bar, pos, track, int(note.pitch), dur, vel))
    events.sort(key=lambda e: (e.bar, e.pos, e.track, e.pitch))
    return events, bpm


def encode_midi(path: str | Path, max_bars: int | None = None) -> tuple[list[int], float]:
    events, bpm = midi_to_events(path, max_bars)
    tokens = [BOS]
    current_bar = -1
    for e in events:
        while current_bar < e.bar:
            tokens.append(BAR)
            current_bar += 1
        tokens.extend(event_to_tokens(e))
    tokens.append(EOS)
    return tokens, bpm


def valid_next_tokens(phase: str, allow_eos: bool = True) -> list[int]:
    if phase == "start":
        return [BAR]
    if phase == "event_or_bar":
        out = list(token_ranges()["pos"]) + [BAR]
        if allow_eos:
            out.append(EOS)
        return out
    return list(token_ranges()[phase])


def advance_phase(token: int, phase: str) -> str:
    if token == EOS:
        return "done"
    if token == BAR:
        return "event_or_bar"
    if POS_BASE <= token < POS_BASE + N_POS:
        return "track"
    if TRACK_BASE <= token < TRACK_BASE + len(TRACKS):
        return "pitch"
    if PITCH_BASE <= token < PITCH_BASE + N_PITCH:
        return "dur"
    if DUR_BASE <= token < DUR_BASE + N_DUR:
        return "vel"
    if VEL_BASE <= token < VEL_BASE + N_VEL:
        return "event_or_bar"
    return phase


def decode_tokens(tokens: Iterable[int], bpm: float = 120.0, max_bars: int | None = None) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    instruments = {
        TRACKS["drums"]: pretty_midi.Instrument(program=0, is_drum=True, name="drums"),
        TRACKS["bass"]: pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Electric Bass (finger)"), name="bass"),
        TRACKS["chords"]: pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Electric Piano 1"), name="chords"),
        TRACKS["melody"]: pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Lead 1 (square)"), name="melody"),
    }
    beat = 60.0 / max(float(bpm), 1e-6)
    bar_len = beat * 4.0
    step = beat / 4.0
    bar = -1
    buf: list[int] = []
    for token in tokens:
        token = int(token)
        if token in (PAD, BOS):
            continue
        if token == EOS:
            break
        if token == BAR:
            bar += 1
            buf.clear()
            if max_bars is not None and bar >= max_bars:
                break
            continue
        buf.append(token)
        if len(buf) < 5 or bar < 0:
            continue
        pos_t, track_t, pitch_t, dur_t, vel_t = buf[:5]
        buf.clear()
        if not (POS_BASE <= pos_t < POS_BASE + N_POS):
            continue
        if not (TRACK_BASE <= track_t < TRACK_BASE + len(TRACKS)):
            continue
        if not (PITCH_BASE <= pitch_t < PITCH_BASE + N_PITCH):
            continue
        if not (DUR_BASE <= dur_t < DUR_BASE + N_DUR):
            continue
        if not (VEL_BASE <= vel_t < VEL_BASE + N_VEL):
            continue
        pos = pos_t - POS_BASE
        track = track_t - TRACK_BASE
        pitch = pitch_t - PITCH_BASE
        dur = (dur_t - DUR_BASE) + 1
        vel_bucket = vel_t - VEL_BASE
        start = bar * bar_len + pos * step
        end = start + dur * step
        velocity = int(np.clip(vel_bucket * 8 + 7, 1, 127))
        instruments[track].notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    for inst in instruments.values():
        if inst.notes:
            pm.instruments.append(inst)
    return pm


def save_tokens_as_midi(tokens: Iterable[int], out: str | Path, bpm: float, max_bars: int | None = None) -> None:
    pm = decode_tokens(tokens, bpm=bpm, max_bars=max_bars)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out))
