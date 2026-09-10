# Musicm8

Musicm8 is now built as an **AI producer + reference-library + synth/FX system** rather than a tiny end-to-end waveform generator.

The main path is:

```text
your reference songs
    ↓
Demucs stems + extracted MIDI + spectral fingerprints
    ↓
existing EnCodec-token links are kept in the reference library
    ↓
local pretrained AI producer brain
    ↓
BPM / key / sections / chord degrees / groove / reference choices / sound-design controls
    ↓
coherent deterministic music engine
    ↓
editable drums / bass / chords / melody MIDI
    ↓
Musicm8 native synth + FX engine
    ↓
audio stems + master.wav
```

The idea is that the songs in `MyDrive/Musicm8/audio/` act as **references**. Musicm8 learns from their arrangement and sonic measurements instead of trying to memorize the final waveform. MIDI describes what was played; stem spectra describe what it sounded like; old EnCodec tokens remain linked for a later neural-resynthesis layer.

## One-click Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elephant-logic/Musicm8/blob/main/notebooks/Musicm8_Colab.ipynb)

1. Choose **Runtime → Change runtime type → GPU**.
2. Put reference audio you are authorized to use in `MyDrive/Musicm8/audio/`.
3. Edit the `IDEA = "..."` line in the big notebook cell.
4. Run the cell.
5. First run: Demucs prepares/caches the reference stems and the local producer model downloads into Drive.
6. Later runs reuse the cached reference analysis and AI-model files.
7. Listen to `master.wav`, or edit the MIDI/audio stems and synth patches.

The default producer brain is a small local pretrained instruct model (`Qwen/Qwen2.5-1.5B-Instruct`) loaded with Transformers. It does **not** synthesize the sound directly. Its job is to reason about the idea and the reference library and write a validated `plan.json`. That separation means the AI model can later be swapped for a stronger local or hosted model without rewriting the music/synth engine.

## Persistent files

```text
MyDrive/Musicm8/work/
  daw_dataset/                 # cached Demucs stems + extracted MIDI/analysis
  tokens-encodec24/            # legacy acoustic tokens, retained as references
  reference_library/
    library.json               # multimodal reference catalogue
  hf_cache/                    # local producer-brain model cache
  ai_projects/latest/
    plan.json                  # AI producer decisions
    arrangement.mid            # complete editable arrangement
    midi_stems/
      drums.mid
      bass.mid
      chords.mid
      melody.mid
    audio_stems/
      drums.wav
      bass.wav
      chords.wav
      melody.wav
    synth_patches.json         # measured-reference-derived synth settings
    master.wav
    project.json
```

## What the reference library stores

`reference_library.py` combines the existing analysis into one catalogue. For every reference it keeps:

- BPM, approximate key and duration
- source stem paths
- extracted drums/bass/chords/melody MIDI paths
- note density, median pitch and pitch range
- RMS/dynamics
- spectral centroid and rolloff
- spectral flatness/noise character
- sub, bass, low-mid, high-mid and air energy ratios
- transient/onset density
- paths to existing EnCodec token clips when available

The LLM only sees compact summaries of these measurements; the raw audio and token files stay on disk for the rendering/resynthesis layers.

## AI producer brain

`producer_ai.py` turns a text idea plus reference summaries into a structured plan. It chooses:

- style
- BPM and key
- section layout and section energy
- a diatonic chord-degree progression
- swing and instrument densities
- which reference to use for drums, bass, chords and melody character
- sound-design controls such as sub level, brightness, drive, detune, filter movement, delay, reverb and width
- basic master controls

The JSON is validated/clamped before the rest of the system uses it. If the local model cannot load, Musicm8 falls back to a deterministic theory-based planner so the workflow still runs.

## Composition engine

`plan_to_midi.py` is deliberately constrained. It converts the producer plan into repeated, key-aware musical patterns instead of sampling arbitrary MIDI events. It currently supports genre-specific drum grids, bass patterns tied to the chord roots, diatonic chord voicings, repeated melody motifs, section energy and swing.

This is the first step toward a more capable hierarchical composer. The earlier learned symbolic Transformer is still in the repo for experiments, but it is no longer the only source of musical structure.

## Musicm8 synth / FX

`musicm8_synth.py` is the first native sound engine. It currently provides:

- sine / triangle / band-limited-ish saw and square oscillators
- detuned/unison voices
- bass sub layer
- ADSR envelopes
- low/high-pass filtering
- kick/snare/hat synthesis
- saturation
- stereo widening
- tempo delay
- simple reverb
- per-track rendering and final master normalization

The starting patch for each role is derived from the selected reference stem's measured spectral fingerprint. The AI then steers those parameters. This is the first implementation of the **"analyse sonic frequencies, then rebuild a similar sound with editable synthesis parameters"** idea.

It is not yet a full Serum/Vital-class synth. The next sound-design milestones are wavetable import/learning, better filters, LFO/modulation routing, convolution/algorithmic reverb, compressors/EQ, and an optimizer that repeatedly renders a patch and minimizes multi-resolution spectral/perceptual distance to a chosen reference stem.

## Local command

```bash
pip install -r requirements-ai.txt

python ai_producer_workflow.py \
  --root /path/to/Musicm8 \
  --idea "dark UK garage, emotional chords, deep moving bass" \
  --bars 32 \
  --device cuda
```

(Colab supplies the GPU and Drive paths automatically through the notebook.)

## Legacy experiments

The earlier EnCodec autoregressive model and the first symbolic-arranger experiment remain in the repository. They are useful research components and their cached data is retained, but the AI-producer/reference/synth workflow is now the recommended path.

## Important

Only use reference audio you are authorized to use. Demucs and any pretrained models have their own licenses. The system is designed to learn reusable musical/sonic characteristics and create new arrangements rather than reproduce reference melodies or recordings verbatim.
