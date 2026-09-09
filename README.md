# Musicm8

A Render-ready inference service for the mini-Suno V2 music model architecture.

## What this repo does

- serves a trained music checkpoint with FastAPI
- provides a browser UI at `/`
- exposes queued generation jobs under `/api/jobs`
- downloads `latest.pt` lazily from `MODEL_CHECKPOINT_URL`
- caches model files on a Render persistent disk
- supports prompt, BPM, key, lyrics, seed, CFG, top-k/top-p and temperature controls
- protects generation with a generated `API_KEY`

> This deployment does **inference**, not training. You still need to train a compatible `latest.pt` checkpoint elsewhere.

## Deploy on Render

1. In Render choose **New → Blueprint**.
2. Connect this GitHub repository: `Elephant-logic/Musicm8`.
3. Render will read `render.yaml` and create the service.
4. After you have trained `latest.pt`, set `MODEL_CHECKPOINT_URL` in the Render service environment to a direct HTTPS download URL.
5. Optionally set `MODEL_CHECKPOINT_SHA256` and `MODEL_DOWNLOAD_TOKEN`.
6. Copy the generated `API_KEY` from Render and paste it into the Musicm8 web UI.

The Blueprint uses a CPU service and a persistent `/var/data` disk. The first generation can take longer while the checkpoint and pretrained codec/text models are downloaded.

## Health and API

- `GET /health` — lightweight Render health check
- `GET /api/status` — model/checkpoint status
- `POST /api/warmup` — load the model ahead of the first generation
- `POST /api/jobs` — queue generation
- `GET /api/jobs/{id}` — poll a generation
- `GET /api/jobs/{id}/audio` — fetch the WAV result

Example request:

```bash
curl -X POST https://YOUR-SERVICE.onrender.com/api/jobs \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_API_KEY' \
  -d '{
    "prompt":"dark atmospheric R&B with deep bass and wide synth pads",
    "seconds":8,
    "bpm":92,
    "key":"F# minor",
    "seed":42
  }'
```

## Required checkpoint format

The checkpoint is the output format used by mini-Suno V2 and must contain:

- `model_config`
- `model`
- `data_meta.codec`

`data_meta.codec` determines whether the service loads EnCodec 24 kHz, MusicGen EnCodec 32 kHz, DAC, or the optional custom music codec.

## Important

Only train on audio you are authorized to use. Model weights and third-party pretrained components can have separate licenses from this code.
