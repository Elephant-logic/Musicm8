# Musicm8

Musicm8 is now built around a **hybrid AI-producer / DAW workflow** rather than asking a tiny model to learn finished waveform audio from scratch.

The main path is:

```text
training songs
    ↓
Demucs stem separation
    ↓
drums / bass / vocals / other
    ↓
BPM + key + beat/chord/pitch analysis
    ↓
editable MIDI: drums / bass / chords / melody
    ↓
small symbolic arrangement Transformer
    ↓
new multitrack MIDI arrangement
    ↓
SoundFont preview in Colab
    ↓
DAW + your VST instruments / effects / mix
```

This keeps the part that a small dataset can realistically teach — **rhythm, notes, arrangement and style patterns** — separate from instrument synthesis and mastering. The generated MIDI remains editable.

## One-click Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elephant-logic/Musicm8/blob/main/notebooks/Musicm8_Colab.ipynb)

1. Choose **Runtime → Change runtime type → GPU**.
2. Put audio you are authorized to use in `MyDrive/Musicm8/audio/`.
3. Run the large notebook cell.
4. The first run separates each song with Demucs, analyzes it and builds MIDI. This is cached in Drive.
5. A compact symbolic Transformer trains/resumes from the cached MIDI dataset.
6. Musicm8 generates `arrangement.mid`, individual MIDI stems and a quick SoundFont preview.
7. Import the MIDI into Ableton, FL Studio, Logic, Reaper or another DAW and assign your preferred VSTs.

Persistent files are stored under `MyDrive/Musicm8/work/`:

```text
work/
  daw_dataset/          # separated stems, analyses and extracted MIDI
  daw_symbolic/         # symbolic arranger checkpoints
  daw_projects/latest/  # generated arrangement.mid, MIDI stems, preview.wav, project.json
```

The workflow is restart-friendly: stem analysis is cached, and symbolic training resumes from `latest.pt`.

## What is extracted

`daw_extract.py` creates four Demucs stems and derives:

- **drums MIDI** from onset/transient analysis, mapped to kick/snare/hat GM notes
- **bass MIDI** from monophonic pitch tracking
- **chord MIDI** from harmonic/chroma analysis
- **melody MIDI** primarily from the vocal/melodic stem
- BPM and an approximate musical key

The extractor is deliberately conservative and editable: it gives the arranger symbolic material without pretending that automatic transcription is perfect.

## VST / DAW rendering

Colab does not host normal desktop VST plugins reliably. Musicm8 therefore uses FluidSynth/SoundFont only for a quick preview. The real output is the MIDI project bundle:

```text
daw_projects/latest/
  arrangement.mid
  midi_stems/
    drums.mid
    bass.mid
    chords.mid
    melody.mid
  project.json
  preview.wav
```

Open those MIDI stems in your DAW and route them to the drum machines, synths, samplers and effects you actually want to use. This is intentionally closer to a normal production workflow than end-to-end raw-audio generation.

## Local commands

```bash
pip install -r requirements-daw.txt

python daw_extract.py \
  --audio-dir data/audio \
  --out data/daw_dataset \
  --device cuda

python train_symbolic.py \
  --index data/daw_dataset/index.jsonl \
  --out runs/daw_symbolic \
  --steps 4000 \
  --device cuda

python generate_symbolic.py \
  --checkpoint runs/daw_symbolic/latest.pt \
  --out project/arrangement.mid \
  --bars 8 \
  --device cuda

python render_daw.py \
  --midi project/arrangement.mid \
  --out project/preview.wav
```

Or run the orchestration command:

```bash
python daw_workflow.py --root /path/to/Musicm8 --steps 4000 --bars 8
```

## Symbolic model

The DAW arranger uses a compact causal Transformer over a constrained MIDI-event vocabulary. Events encode bar position, instrument role, MIDI pitch, duration and velocity. During generation, the grammar is constrained so the output remains valid MIDI. Training also uses pitch-transposition augmentation, which is much more data-efficient than learning EnCodec acoustic codebooks from a handful of songs.

## Legacy neural-audio experiment

The earlier EnCodec-token autoregressive model remains in the repository (`model.py`, `training_model.py`, `train.py`, `generate.py`) as a research path. It successfully proves the audio-token pipeline, but the tiny from-scratch model is no longer the recommended route for the main Musicm8 workflow.

## Important

Only train on audio you are authorized to use. Demucs, pretrained separation weights, SoundFonts and any VSTs you use have their own licenses. GPU availability and Colab session limits are controlled by Google and can change.
