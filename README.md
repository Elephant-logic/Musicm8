# Musicm8

Musicm8 is a hybrid **AI producer + reference library + native Synth v2/FX engine**.

```text
reference songs
    ↓
Demucs stems + extracted MIDI + BPM/key
    ↓
spectral fingerprints + retained EnCodec token links
    ↓
local pretrained producer AI
    ↓
structured song plan: sections / chords / groove / reference choices / sound design
    ↓
coherent editable MIDI
    ↓
Synth v2 inverse sound designer
    ├─ learn harmonic/wavetable fingerprints from pitched stems
    ├─ analyse kick / snare / hats separately
    ├─ try wavetable / FM / subtractive / noise layers
    ├─ render the aligned reference MIDI
    ├─ compare several spectral resolutions + mel/envelope/transients
    └─ search and cache the best reusable patch
    ↓
EQ + compression + FX + kick sidechain + stem balance
    ↓
drums.wav / bass.wav / chords.wav / melody.wav / master.wav
```

The songs are used as **references**, not as a tiny waveform model's memorisation target. MIDI represents what was played; stems and spectral measurements describe how it sounded; the existing EnCodec token cache remains linked for a future neural-residual/resynthesis layer.

## One-click Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elephant-logic/Musicm8/blob/main/notebooks/Musicm8_Colab.ipynb)

1. Choose **Runtime → Change runtime type → GPU**.
2. Put authorised reference audio in `MyDrive/Musicm8/audio/`.
3. Change the `IDEA` line.
4. Run the large cell.

```python
IDEA = "dark UK garage, emotional chords, deep moving bass, spacious pads"
BARS = 32
MATCH_ITERS = 48
```

The notebook refreshes the repo automatically. Reference analysis, producer-model files and Synth v2 patches are cached in Drive.

## Synth v2

`musicm8_synth_v2.py` adds a substantially larger controllable sound space:

- learned additive/harmonic wavetable oscillator
- FM synthesis and FM blending
- sine / triangle / saw / square oscillators
- sub oscillator, detune/unison, noise texture and LFO amplitude modulation
- ADSR plus envelope-driven filter motion
- separate parametric kick, snare and hat synthesis
- 3-band EQ, block compressor, saturation, stereo width, delay and reverb
- kick-triggered sidechain for bass/chords
- DAW-style per-stem level balancing before the master bus

`sound_matcher_v2.py` learns the harmonic fingerprint directly from the selected reference window, estimates separate drum-voice properties, and searches the larger synth/FX parameter space. Its distance metric combines log-spectrum profiles at FFT sizes **512, 2048 and 8192**, a mel spectrum, amplitude envelope, onset envelope, spectral-centroid motion and a small loudness term.

For pitched parts whose extracted MIDI is sparse, the matcher can fall back to direct pitch estimation from the reference window rather than dropping the sound entirely.

`render_matched_v2.py` takes the reusable matched reference patch, nudges it toward the producer AI's requested character, renders the new MIDI, balances stems, sidechains the low end and produces the final master.

## Persistent Drive layout

```text
MyDrive/Musicm8/work/
  daw_dataset/                 # Demucs stems + extracted MIDI
  tokens-encodec24/            # retained acoustic-token cache
  reference_library/library.json
  hf_cache/                    # local producer AI cache
  sound_patch_cache_v2/        # reusable Synth v2 matches
  ai_projects/latest/
    plan.json
    arrangement.mid
    midi_stems/
    matched_patches.json
    sound_matches/
      bass_reference.wav
      bass_matched.wav
      drums_reference.wav
      drums_matched.wav
      ...
    synth_patches.json
    audio_stems/
    project.json
    master.wav
```

## Where the AI is

`producer_ai.py` is the high-level producer brain. It reads compact summaries of the reference library plus the text idea and writes a validated `plan.json` containing BPM, key, sections, chord degrees, groove, selected references, synth controls and mix controls.

The downstream composition engine remains constrained by music rules so the LLM makes creative decisions without blindly emitting arbitrary MIDI.

## Local command

```bash
pip install -r requirements-ai.txt

python ai_producer_workflow.py \
  --root /path/to/Musicm8 \
  --idea "dark garage with a huge moving bass" \
  --bars 32 \
  --match-iters 48
```

Use `--force-sound-match` to ignore the v2 cache and search again. Use `--skip-sound-match` for a faster fallback render.

## What comes after v2

Synth v2 still produces a controllable approximation, not perfect resynthesis. The retained EnCodec tokens are intentionally preserved for a later neural-residual layer that can add acoustic details conventional synthesis cannot represent while leaving the MIDI, synth patch and mix editable.

## Important

Only analyse/train on audio you are authorised to use. Pretrained models, Demucs weights, codecs and third-party plugins have their own licences. GPU availability and Colab quotas are controlled by Google.
