# Musicm8

A small research stack for training and serving a Suno-style autoregressive music model.

## Free GPU training

You do **not** need to pay for Render or rent a GPU just to start experimenting.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elephant-logic/Musicm8/blob/main/notebooks/Musicm8_Colab.ipynb)

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/Elephant-logic/Musicm8/blob/main/notebooks/Musicm8_Kaggle.ipynb)

Both notebooks:

- clone the latest `Musicm8` code from this repo
- verify that a CUDA GPU is available
- install the model dependencies
- build a training manifest from your audio when needed
- tokenize audio with EnCodec
- train the tiny model
- save/resume `latest.pt`
- generate a short WAV so you can hear whether training is working

Start with roughly 20–50 tracks you are authorized to use. The first milestone is simply to overfit a small dataset and produce recognizable short music.

### Colab

1. Open the Colab badge above.
2. Choose **Runtime → Change runtime type → GPU**.
3. Put your audio in Google Drive at `MyDrive/Musicm8/audio/`.
4. Run the notebook from top to bottom.
5. Checkpoints and tokenized audio are stored in `MyDrive/Musicm8/work/`, so later sessions can resume.

### Kaggle

1. Open the Kaggle badge above.
2. Enable a GPU accelerator in notebook settings.
3. Attach a Kaggle Dataset containing your authorized training audio.
4. Enable Internet for the notebook so it can clone this repo and download pretrained codec/text weights.
5. Run all cells. Use **Save Version** / notebook outputs to keep `latest.pt` between sessions.

GPU availability, accelerator type, quotas and session limits are controlled by Colab/Kaggle and can change.

## Local training commands

The notebooks automate these same steps:

```bash
pip install -r requirements.txt

python tokenize_dataset.py \
  --manifest data/manifest.jsonl \
  --out data/tokens \
  --codec encodec24 \
  --channels 1 \
  --clip-seconds 8 \
  --stride-seconds 8 \
  --keep-tail \
  --device cuda

python train.py \
  --data data/tokens/index.jsonl \
  --config configs/v2-tiny.json \
  --out runs/tiny-overfit \
  --batch-size 2 \
  --grad-accum 2 \
  --steps 2000 \
  --device cuda

python generate.py \
  --checkpoint runs/tiny-overfit/latest.pt \
  --prompt "dark atmospheric electronic music with deep bass" \
  --seconds 8 \
  --out sample.wav \
  --device cuda
```

## Architecture

The core path is:

```text
prompt / structured controls
          │
          ▼
      text encoder
          │
          ▼
previous codec tokens → causal Transformer
          │
          ▼
delayed RVQ codebook predictions
          │
          ▼
       audio codec
          │
          ▼
          WAV
```

The model code also contains conditioning support for tempo, key, chords, sections, melody, energy, phonemes and semantic IDs, plus reference-audio context for continuation/infill experiments.

## Optional web service

The repository still contains `render_app.py` and `render.yaml` for serving a trained checkpoint through FastAPI. That path is optional. If you want to stay at £0, focus on the Colab/Kaggle notebooks first and keep the checkpoint in Drive/Kaggle outputs rather than deploying an always-on service.

## Checkpoint format

A Musicm8 checkpoint contains:

- `model_config`
- `model`
- optimizer/scheduler state for resume
- `data_meta.codec`

`data_meta.codec` tells generation which neural codec produced the training tokens.

## Important

Only train on audio you are authorized to use. Third-party pretrained components and model weights can have licenses separate from this repository.
