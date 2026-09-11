# Musicm8

Musicm8 is a hybrid **AI producer + reference library + native Synth v2/FX engine + neural vocal system**.

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
original section-aware AI lyrics
    ↓
hierarchical chord-aware MIDI composition
    ↓
Synth v2 inverse sound designer
    ├─ learn harmonic/wavetable fingerprints from pitched stems
    ├─ analyse kick / snare / hats separately
    ├─ try wavetable / FM / subtractive / noise layers
    ├─ render the aligned reference MIDI
    ├─ compare several spectral resolutions + mel/envelope/transients
    └─ search and cache the best reusable patch
    ↓
EQ + compression + FX + kick sidechain + stereo mix polish
    ↓
lyric-to-melody vocal score
    ↓
ACE-Step 1.5 neural vocals generated in context of Musicm8's backing
    ↓
vocal EQ / de-essing / compression / doubles / reverb / delay / backing ducking
    ↓
complete master.wav
```

The songs are used as **references**, not as a tiny waveform model's memorisation target. MIDI represents what was played; stems and spectral measurements describe how it sounded; the existing EnCodec token cache remains linked for later neural-residual/resynthesis work.

## One-click Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elephant-logic/Musicm8/blob/main/notebooks/Musicm8_Colab.ipynb)

1. Choose **Runtime → Change runtime type → GPU**.
2. Put authorised reference audio in `MyDrive/Musicm8/audio/`.
3. Change `IDEA` and optionally `VOCAL_STYLE` in the large cell.
4. Run the cell.

```python
IDEA = "dark UK garage song about a relationship ending, emotional chords, deep moving bass"
BARS = 32
VOCALS = True
VOCAL_STYLE = "expressive contemporary lead vocal, intimate verses, emotional hook, clear lyrics"
VOCAL_LANGUAGE = "en"
VOCAL_STEPS = 40
MATCH_ITERS = 48
```

The notebook refreshes Musicm8 automatically. Reference analysis, producer-model files, Synth v2 patches, uv downloads and ACE-Step model weights are cached in Google Drive.

## Lyrics and vocal score

`lyrics_ai.py` uses the same local producer-class language model to write **original** lyrics that follow the planned sections and energy curve. It saves both `lyrics.json` and a bracketed `lyrics.txt` suitable for the singing backend. It has a deterministic original fallback, so a producer-model failure does not stop the project.

`vocal_score.py` aligns lyric syllables to Musicm8's generated melody MIDI. It saves note start/end, MIDI pitch, lyric syllable/word and an IPA phoneme guide when `espeak-ng` is available. The melody is also copied to `vocal_melody.mid`, so the vocal line remains editable even when the neural singer is changed later.

## Neural singing backend

`neural_vocals.py` automatically bootstraps **ACE-Step 1.5** in an isolated Python 3.12 environment using `uv`. This is deliberate because the main Colab runtime may be on a newer Python version. ACE-Step model weights are stored under `MyDrive/Musicm8/work/ace_step_models/` so they survive Colab resets.

Musicm8 uses ACE-Step's **base-model `lego` vocals task**: the polished Musicm8 instrumental is supplied as audio context, along with the generated lyrics, BPM, key, language and vocal-style description. The result is requested as a vocal track rather than asking ACE-Step to replace Musicm8's composition and synth engine.

The first neural-vocal run is the largest/slowest setup step because ACE-Step and its base model must be downloaded. If the neural backend fails, Musicm8 keeps the instrumental, lyrics and vocal score instead of discarding the project.

## Vocal production

`mix_vocals.py` keeps the raw neural vocal and produces a DAW-style vocal stack:

- centered lead vocal
- high-pass / low-pass cleanup
- presence EQ
- de-essing
- compression and light saturation
- quiet left/right micro-delay doubles
- short room + tempo delay
- light backing-track ducking while the lead is active
- final bus limiting/peak protection

It preserves the instrumental as `master_instrumental.wav` and writes the completed song to `master.wav`.

## Synth v2

`musicm8_synth_v2.py` provides:

- learned additive/harmonic wavetable oscillator
- FM synthesis and FM blending
- sine / triangle / saw / square oscillators
- sub oscillator, detune/unison, noise texture and LFO amplitude modulation
- ADSR plus envelope-driven filter motion
- separate parametric kick, snare and hat synthesis
- 3-band EQ, compressor, saturation, stereo width, delay and reverb
- kick-triggered sidechain for bass/chords
- DAW-style per-stem level balancing

`sound_matcher_v2.py` compares reference and synth sound at FFT sizes **512, 2048 and 8192**, plus mel spectrum, amplitude envelope, onset envelope, spectral-centroid motion and loudness, then searches the controllable patch space and caches the best reusable result.

## Persistent Drive layout

```text
MyDrive/Musicm8/work/
  daw_dataset/
  tokens-encodec24/
  reference_library/library.json
  hf_cache/
  uv_cache/
  ace_step_models/
  sound_patch_cache_v2/
  ai_projects/latest/
    plan.json
    lyrics.json
    lyrics.txt
    vocal_score.json
    vocal_melody.mid
    arrangement.mid
    midi_stems/
    matched_patches.json
    sound_matches/
    synth_patches.json
    audio_stems/
    audio_stems_polished/
    master_prepolish.wav
    master_instrumental.wav
    vocals/
      neural_lead_raw.wav
      lead.wav
      double_L.wav
      double_R.wav
      vocal_mix.wav
    vocal_mix_report.json
    project.json
    master.wav
```

## Local command

```bash
pip install -r requirements-ai.txt

python ai_producer_workflow.py \
  --root /path/to/Musicm8 \
  --idea "dark garage song about not being able to leave" \
  --bars 32 \
  --match-iters 48 \
  --vocal-style "intimate lead vocal, emotional chorus"
```

Use `--no-vocals` for an instrumental-only run, `--force-sound-match` to ignore the v2 sound cache, or `--skip-sound-match` for a faster fallback render.

## Important

Only analyse/train on audio you are authorised to use. Pretrained models, Demucs weights, codecs and third-party tools have their own licences. Voice-reference use should also respect the singer's rights and permission. GPU availability and Colab quotas are controlled by Google.
