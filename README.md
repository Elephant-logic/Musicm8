# Musicm8

Musicm8 is a hybrid **AI producer + reference library + native synth/FX engine**.

The main workflow is now:

```text
reference songs
    ↓
Demucs stems + extracted MIDI + BPM/key
    ↓
spectral fingerprints + existing EnCodec token links
    ↓
local pretrained producer AI
    ↓
song plan: sections / key / chords / groove / reference choices / sound design
    ↓
coherent editable MIDI
    ↓
inverse sound designer
    ├─ take selected reference stem + aligned MIDI
    ├─ render Musicm8's own synth
    ├─ compare frequency shape / envelope / transients / loudness
    ├─ change synth + FX parameters
    └─ repeat and cache the best patch
    ↓
Musicm8 native synth + FX
    ↓
drums.wav / bass.wav / chords.wav / melody.wav
    ↓
master.wav + editable MIDI + synth patch JSON
```

The reference songs are **references**, not one tiny waveform model's memorization target. MIDI represents what was played; stems and spectral measurements describe how it sounded; the existing EnCodec token cache stays linked for future neural residual/resynthesis work.

## One-click Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elephant-logic/Musicm8/blob/main/notebooks/Musicm8_Colab.ipynb)

1. Choose **Runtime → Change runtime type → GPU**.
2. Put audio you are authorized to use in `MyDrive/Musicm8/audio/`.
3. Change the `IDEA` line in the large cell.
4. Run it.

Example:

```python
IDEA = "dark UK garage, emotional chords, deep moving bass, spacious pads"
BARS = 32
MATCH_ITERS = 40
```

The first run can take longer because stems and the local producer model have to be prepared. Expensive reference work is cached in Drive.

## Persistent Drive layout

```text
MyDrive/Musicm8/work/
  daw_dataset/                 # Demucs stems + extracted MIDI
  tokens-encodec24/            # older acoustic-token cache, retained
  reference_library/
    library.json               # multimodal reference index
  hf_cache/                    # local producer AI cache
  sound_patch_cache/           # reusable inverse-synthesis patches
  ai_projects/latest/
    plan.json
    arrangement.mid
    midi_stems/
      drums.mid
      bass.mid
      chords.mid
      melody.mid
    matched_patches.json
    sound_matches/
      bass_reference.wav
      bass_matched.wav
      ...
    synth_patches.json
    audio_stems/
      drums.wav
      bass.wav
      chords.wav
      melody.wav
    project.json
    master.wav
```

## Where the AI is

`producer_ai.py` is the high-level producer brain. It reads a compact form of the reference library plus the text idea and writes a constrained `plan.json` containing BPM, key, section energy, chord degrees, groove, selected references, synth controls and mix controls.

The planner is deliberately separated from deterministic music rules and rendering. AI can make creative decisions, while the downstream engine enforces valid key/chord/groove structure instead of blindly emitting random MIDI.

## Inverse sound design

`sound_matcher.py` is the first inverse-synthesis loop.

For each role selected by the producer:

1. Find the chosen reference song and its matching stem/MIDI.
2. Select a short active aligned window.
3. Build an initial native-synth patch from the measured spectral fingerprint.
4. Render the same MIDI through Musicm8's synth.
5. Compare the synth render with the real reference using:
   - log-mel spectral/frequency shape
   - amplitude envelope
   - onset/transient envelope
   - a small loudness term
6. Mutate oscillator/filter/envelope/drive/width/reverb/delay controls.
7. Keep improvements and repeat.
8. Cache the best reference patch.

`render_matched.py` then takes that matched reference patch and nudges it with the producer AI's idea-specific sound-design controls before rendering the new composition.

The reported similarity value is an internal optimization diagnostic, **not** a claim of human perceptual identity. The native synth has a limited parameter space, so it will approximate sounds it can represent and fall back to spectral-fingerprint patches when it cannot.

## Current native synth

`musicm8_synth.py` currently provides:

- sine / triangle / saw / square oscillators
- detuning and sub layers
- ADSR envelopes
- low/high filtering
- saturation
- stereo widening
- delay
- simple reverb
- synthesized kick/snare/hats
- stem rendering and mastering

This is intentionally a controllable starting point. More synthesis methods (wavetable, FM, granular/sample layers, convolution and eventually a neural residual layer using the retained acoustic tokens) can be added without changing the producer/reference architecture.

## Local command

```bash
pip install -r requirements-ai.txt

python ai_producer_workflow.py \
  --root /path/to/Musicm8 \
  --idea "dark garage with a huge moving bass" \
  --bars 32 \
  --match-iters 40
```

Use `--force-sound-match` to discard cached patches and search again, or `--skip-sound-match` for a faster fingerprint-only render.

## Important

Only analyse/train on audio you are authorized to use. Pretrained models, Demucs weights, codecs, SoundFonts and third-party plugins have their own licences. GPU availability and Colab quotas are controlled by Google.
