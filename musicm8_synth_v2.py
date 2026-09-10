from __future__ import annotations

import math
from typing import Any

import numpy as np
import pretty_midi
from scipy.signal import butter, sosfilt

SR = 44100


def clamp(x: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return max(lo, min(hi, v))
    except Exception:
        pass
    return default


def midi_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((int(note) - 69) / 12.0))


def poly_blep(t: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(t)
    if dt <= 0:
        return out
    a = t < dt
    if np.any(a):
        x = t[a] / dt
        out[a] = x + x - x * x - 1.0
    b = t > 1.0 - dt
    if np.any(b):
        x = (t[b] - 1.0) / dt
        out[b] = x * x + x + x + 1.0
    return out


def oscillator(kind: str, freq: float, n: int, phase0: float = 0.0) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    phase = (phase0 + np.arange(n, dtype=np.float64) * freq / SR) % 1.0
    kind = str(kind).lower()
    if kind == "sine":
        y = np.sin(2 * np.pi * phase)
    elif kind == "triangle":
        y = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    elif kind == "square":
        y = np.where(phase < 0.5, 1.0, -1.0)
        dt = min(0.5, freq / SR)
        y += poly_blep(phase, dt)
        y -= poly_blep((phase + 0.5) % 1.0, dt)
    else:
        y = 2.0 * phase - 1.0
        y -= poly_blep(phase, min(0.5, freq / SR))
    return y.astype(np.float32)


def harmonic_oscillator(freq: float, n: int, harmonics: list[float] | None) -> np.ndarray:
    amps = np.asarray(harmonics or [1.0], dtype=np.float64).reshape(-1)
    if amps.size == 0:
        amps = np.array([1.0], dtype=np.float64)
    t = np.arange(n, dtype=np.float64) / SR
    y = np.zeros(n, dtype=np.float64)
    used = 0.0
    for i, amp in enumerate(amps, start=1):
        if i * freq >= SR * 0.47:
            break
        a = float(max(0.0, amp))
        if a <= 1e-8:
            continue
        y += a * np.sin(2 * np.pi * (i * freq) * t + 0.17 * i)
        used += a
    if used > 1e-8:
        y /= used
    return y.astype(np.float32)


def fm_oscillator(freq: float, n: int, ratio: float, index: float) -> np.ndarray:
    ratio = clamp(ratio, 0.25, 8.0, 2.0)
    index = clamp(index, 0.0, 10.0, 0.0)
    t = np.arange(n, dtype=np.float64) / SR
    mod = np.sin(2 * np.pi * freq * ratio * t)
    y = np.sin(2 * np.pi * freq * t + index * mod)
    return y.astype(np.float32)


def adsr(n: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    attack = clamp(attack, 0.0005, 4.0, 0.01)
    decay = clamp(decay, 0.001, 6.0, 0.2)
    sustain = clamp(sustain, 0.0, 1.0, 0.7)
    release = clamp(release, 0.001, 8.0, 0.15)
    a = min(n, max(1, int(attack * SR)))
    d = min(max(0, n - a), max(1, int(decay * SR)))
    r = min(max(1, int(release * SR)), max(1, n - a - d))
    s = max(0, n - a - d - r)
    env = np.empty(n, dtype=np.float32)
    pos = 0
    env[pos:pos+a] = np.linspace(0, 1, a, endpoint=False)
    pos += a
    if d:
        env[pos:pos+d] = np.linspace(1, sustain, d, endpoint=False)
        pos += d
    if s:
        env[pos:pos+s] = sustain
        pos += s
    if pos < n:
        start = float(env[pos-1]) if pos else sustain
        env[pos:] = np.linspace(start, 0, n-pos, endpoint=True)
    return env


def filt(x: np.ndarray, cutoff: float, kind: str = "low", order: int = 2) -> np.ndarray:
    cutoff = clamp(cutoff, 20, SR * 0.45, 8000)
    sos = butter(order, cutoff / (SR * 0.5), btype=kind, output="sos")
    if x.ndim == 1:
        return sosfilt(sos, x).astype(np.float32)
    return np.vstack([sosfilt(sos, ch) for ch in x]).astype(np.float32)


def saturate(x: np.ndarray, drive: float) -> np.ndarray:
    drive = clamp(drive, 0, 1, 0)
    amount = 1.0 + 10.0 * drive
    return (np.tanh(x * amount) / max(np.tanh(amount), 1e-6)).astype(np.float32)


def three_band_eq(x: np.ndarray, low_db: float = 0.0, mid_db: float = 0.0, high_db: float = 0.0) -> np.ndarray:
    low = filt(x, 180.0, "low")
    high = filt(x, 4200.0, "high")
    mid = x - low - high
    gl = 10 ** (clamp(low_db, -18, 18, 0) / 20)
    gm = 10 ** (clamp(mid_db, -18, 18, 0) / 20)
    gh = 10 ** (clamp(high_db, -18, 18, 0) / 20)
    return (low * gl + mid * gm + high * gh).astype(np.float32)


def block_compressor(x: np.ndarray, threshold_db: float = -16.0, ratio: float = 3.0, makeup_db: float = 0.0) -> np.ndarray:
    threshold_db = clamp(threshold_db, -40, 0, -16)
    ratio = clamp(ratio, 1, 20, 3)
    block = 512
    mono = np.mean(np.abs(x), axis=0) if x.ndim == 2 else np.abs(x)
    n_blocks = max(1, math.ceil(mono.size / block))
    rms = np.zeros(n_blocks, dtype=np.float32)
    for i in range(n_blocks):
        part = mono[i*block:(i+1)*block]
        rms[i] = math.sqrt(float(np.mean(part * part)) + 1e-12) if part.size else 0.0
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    over = np.maximum(0.0, db - threshold_db)
    reduction_db = over * (1.0 - 1.0 / ratio)
    gain_points = 10 ** ((-reduction_db + clamp(makeup_db, -12, 12, 0)) / 20)
    xp = np.linspace(0, max(0, mono.size - 1), n_blocks)
    gain = np.interp(np.arange(mono.size), xp, gain_points).astype(np.float32)
    if x.ndim == 2:
        return (x * gain[None]).astype(np.float32)
    return (x * gain).astype(np.float32)


def stereo_delay(x: np.ndarray, amount: float) -> np.ndarray:
    amount = clamp(amount, 0, 1, 0)
    if amount <= 0:
        return x
    d = max(1, int((0.007 + 0.021 * amount) * SR))
    wet = np.zeros_like(x)
    wet[0, d:] = x[1, :-d]
    wet[1, d:] = x[0, :-d]
    return (x * (1 - 0.24 * amount) + wet * (0.24 * amount)).astype(np.float32)


def echo(x: np.ndarray, amount: float, bpm: float) -> np.ndarray:
    amount = clamp(amount, 0, 1, 0)
    if amount <= 0:
        return x
    d = max(1, int((60.0 / max(bpm, 1.0)) * 0.75 * SR))
    y = x.copy()
    gain = 0.34 * amount
    for k in range(1, 4):
        dk = d * k
        if dk >= x.shape[-1]:
            break
        y[:, dk:] += x[:, :-dk] * (gain ** k)
    return y.astype(np.float32)


def reverb(x: np.ndarray, amount: float) -> np.ndarray:
    amount = clamp(amount, 0, 1, 0)
    if amount <= 0:
        return x
    y = x.copy()
    taps = ((0.029, 0.35), (0.043, 0.29), (0.067, 0.23), (0.097, 0.17), (0.149, 0.11))
    for seconds, gain in taps:
        d = int(seconds * SR)
        if d < x.shape[-1]:
            y[:, d:] += x[:, :-d] * gain * amount
    y = filt(y, 12000 - 5000 * amount, "low")
    return (x * (1 - 0.10 * amount) + y * (0.22 * amount)).astype(np.float32)


def role_fx(audio: np.ndarray, role: str, patch: dict[str, Any], bpm: float) -> np.ndarray:
    y = audio
    y = three_band_eq(
        y,
        float(patch.get("eq_low_db", 0.0)),
        float(patch.get("eq_mid_db", 0.0)),
        float(patch.get("eq_high_db", 0.0)),
    )
    y = block_compressor(
        y,
        float(patch.get("comp_threshold_db", -18.0)),
        float(patch.get("comp_ratio", 2.0)),
        float(patch.get("comp_makeup_db", 0.0)),
    )
    if role == "chords":
        y = reverb(y, float(patch.get("reverb", 0.0)))
    elif role == "melody":
        y = reverb(echo(y, float(patch.get("delay", 0.0)), bpm), float(patch.get("reverb", 0.0)))
    elif role == "drums":
        y = reverb(y, float(patch.get("room", 0.0)) * 0.45)
    return y.astype(np.float32)


def _noise(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, n).astype(np.float32)


def synth_note(note: int, duration: float, velocity: int, patch: dict[str, Any], role: str) -> np.ndarray:
    release = clamp(patch.get("release", 0.12), 0.005, 4.0, 0.12)
    n = max(1, int((duration + release) * SR))
    freq = midi_hz(note)
    kind = str(patch.get("wave", "saw")).lower()

    if kind == "wavetable":
        base = harmonic_oscillator(freq, n, patch.get("harmonics"))
    elif kind == "fm":
        base = fm_oscillator(freq, n, float(patch.get("fm_ratio", 2.0)), float(patch.get("fm_index", 1.0)))
    else:
        base = oscillator(kind, freq, n)

    fm_mix = clamp(patch.get("fm_mix", 0.0), 0, 1, 0)
    if fm_mix > 0 and kind != "fm":
        fm = fm_oscillator(freq, n, float(patch.get("fm_ratio", 2.0)), float(patch.get("fm_index", 1.0)))
        base = base * (1 - fm_mix) + fm * fm_mix

    detune = clamp(patch.get("detune", 0.0), 0, 0.6, 0)
    if detune > 0:
        cents = 4 + 22 * detune
        ratio = 2 ** (cents / 1200)
        if kind == "wavetable":
            up = harmonic_oscillator(freq * ratio, n, patch.get("harmonics"))
            down = harmonic_oscillator(freq / ratio, n, patch.get("harmonics"))
        else:
            up = oscillator("saw" if kind == "fm" else kind, freq * ratio, n, 0.17)
            down = oscillator("saw" if kind == "fm" else kind, freq / ratio, n, 0.41)
        base = 0.58 * base + 0.21 * up + 0.21 * down

    if role == "bass":
        sub = clamp(patch.get("sub", 0.5), 0, 1, 0.5)
        base = base * (1 - 0.42 * sub) + oscillator("sine", freq, n, 0.25) * (0.72 * sub)

    noise_mix = clamp(patch.get("noise_mix", 0.0), 0, 0.5, 0)
    if noise_mix > 0:
        texture = filt(_noise(n, note + 17), clamp(patch.get("noise_cutoff", 6000), 400, 16000, 6000), "low")
        base = base * (1 - noise_mix) + texture * (0.28 * noise_mix)

    lfo_depth = clamp(patch.get("lfo_depth", 0.0), 0, 1, 0)
    lfo_rate = clamp(patch.get("lfo_rate", 2.0), 0.05, 16.0, 2.0)
    if lfo_depth > 0:
        t = np.arange(n, dtype=np.float32) / SR
        trem = 1.0 - 0.22 * lfo_depth + 0.22 * lfo_depth * np.sin(2 * np.pi * lfo_rate * t)
        base *= trem.astype(np.float32)

    env = adsr(
        n,
        float(patch.get("attack", 0.01)),
        float(patch.get("decay", 0.18)),
        float(patch.get("sustain", 0.65)),
        release,
    )

    cutoff = clamp(patch.get("cutoff", 8000), 80, 18000, 8000)
    motion = clamp(patch.get("filter_env", 0.0), 0, 1, 0)
    low = filt(base, max(80, cutoff * (0.35 + 0.35 * motion)), "low")
    if motion > 0:
        high = filt(base, min(18000, cutoff * (1.0 + 1.8 * motion)), "low")
        sweep = np.clip(env * (0.4 + 0.8 * motion), 0, 1)
        base = low * (1 - sweep) + high * sweep
    else:
        base = low

    y = base * env * (velocity / 127.0)
    return y.astype(np.float32)


def kick_voice(n: int, velocity: int, cfg: dict[str, Any]) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    start_hz = clamp(cfg.get("pitch_start", 150), 70, 260, 150)
    end_hz = clamp(cfg.get("pitch_end", 45), 28, 80, 45)
    sweep = clamp(cfg.get("sweep", 0.028), 0.008, 0.12, 0.028)
    decay = clamp(cfg.get("decay", 0.20), 0.05, 0.8, 0.20)
    phase = 2 * np.pi * (end_hz * t + (start_hz - end_hz) * (1 - np.exp(-t / sweep)) * sweep)
    body = np.sin(phase) * np.exp(-t / decay)
    click = _noise(n, 101) * np.exp(-t / 0.006)
    y = body + click * (0.02 + 0.16 * clamp(cfg.get("click", 0.25), 0, 1, 0.25))
    return (y * velocity / 127.0).astype(np.float32)


def snare_voice(n: int, velocity: int, cfg: dict[str, Any]) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    tone = clamp(cfg.get("tone_hz", 190), 110, 360, 190)
    decay = clamp(cfg.get("decay", 0.14), 0.04, 0.7, 0.14)
    noise_mix = clamp(cfg.get("noise", 0.72), 0.2, 1, 0.72)
    brightness = clamp(cfg.get("brightness", 0.55), 0, 1, 0.55)
    noise = _noise(n, 202)
    noise = filt(noise, 1000 + 5000 * brightness, "high")
    tonal = np.sin(2*np.pi*tone*t) + 0.45*np.sin(2*np.pi*tone*1.78*t)
    env = np.exp(-t / decay)
    y = (noise * noise_mix + tonal * (1 - noise_mix)) * env
    return (0.62 * y * velocity / 127.0).astype(np.float32)


def hat_voice(n: int, velocity: int, cfg: dict[str, Any], open_hat: bool = False) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    decay = clamp(cfg.get("open_decay" if open_hat else "decay", 0.18 if open_hat else 0.05), 0.015, 0.8, 0.08)
    hp = clamp(cfg.get("highpass", 5200), 2500, 12000, 5200)
    metallic = clamp(cfg.get("metallic", 0.5), 0, 1, 0.5)
    noise = filt(_noise(n, 303 if open_hat else 304), hp, "high")
    freqs = (4000, 5170, 6320, 8030)
    metal = sum(np.sign(np.sin(2*np.pi*f*t)) for f in freqs) / len(freqs)
    y = (noise * (1 - 0.45 * metallic) + metal.astype(np.float32) * 0.45 * metallic) * np.exp(-t / decay)
    return (0.24 * y * velocity / 127.0).astype(np.float32)


def render_instrument(inst: pretty_midi.Instrument, role: str, patch: dict[str, Any], total_n: int) -> np.ndarray:
    mono = np.zeros(total_n, dtype=np.float32)
    if role == "drums":
        kick_cfg = patch.get("kick", {})
        snare_cfg = patch.get("snare", {})
        hat_cfg = patch.get("hat", {})
        for note in inst.notes:
            start = int(note.start * SR)
            if note.pitch in (35, 36):
                voice = kick_voice(int(0.85 * SR), note.velocity, kick_cfg)
            elif note.pitch in (38, 40):
                voice = snare_voice(int(0.75 * SR), note.velocity, snare_cfg)
            else:
                voice = hat_voice(int((0.75 if note.pitch == 46 else 0.35) * SR), note.velocity, hat_cfg, note.pitch == 46)
            end = min(total_n, start + len(voice))
            if 0 <= start < total_n:
                mono[start:end] += voice[: end - start]
    else:
        for note in inst.notes:
            start = int(note.start * SR)
            voice = synth_note(note.pitch, max(0.03, note.end - note.start), note.velocity, patch, role)
            end = min(total_n, start + len(voice))
            if 0 <= start < total_n:
                mono[start:end] += voice[: end - start]

    mono = saturate(mono, float(patch.get("drive", 0.0)))
    mono *= float(patch.get("gain", 0.5))
    stereo = np.vstack([mono.copy(), mono.copy()])
    width = clamp(patch.get("width", 0.0), 0, 1, 0)
    if width > 0:
        stereo = stereo_delay(stereo, width)
    return stereo.astype(np.float32)


def sidechain_gain(inst: pretty_midi.Instrument | None, total_n: int, depth: float = 0.55, release_s: float = 0.22) -> np.ndarray:
    gain = np.ones(total_n, dtype=np.float32)
    if inst is None:
        return gain
    depth = clamp(depth, 0, 0.95, 0.55)
    release_s = clamp(release_s, 0.04, 1.0, 0.22)
    length = max(1, int(release_s * SR * 5))
    env = np.exp(-np.arange(length, dtype=np.float32) / max(1.0, release_s * SR))
    for note in inst.notes:
        if note.pitch not in (35, 36):
            continue
        start = int(note.start * SR)
        if start >= total_n:
            continue
        end = min(total_n, start + length)
        gain[start:end] = np.minimum(gain[start:end], 1.0 - depth * env[: end - start])
    return gain


def rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(math.sqrt(float(np.mean(np.square(x, dtype=np.float64))) + 1e-12) + 1e-12)


def set_rms(x: np.ndarray, target_db: float, max_gain_db: float = 12.0) -> np.ndarray:
    cur = rms_db(x)
    gain_db = max(-18.0, min(max_gain_db, target_db - cur))
    return (x * (10 ** (gain_db / 20))).astype(np.float32)
