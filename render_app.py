from __future__ import annotations

import hashlib
import os
import queue
import secrets
import shutil
import threading
import time
import traceback
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from generate import load_model
from planner import SongPlan, heuristic_plan, plan_to_conditions

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", APP_DIR / ".render-data/models")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", APP_DIR / ".render-data/outputs")).resolve()
CHECKPOINT = Path(os.getenv("MODEL_CHECKPOINT_PATH", MODEL_DIR / "latest.pt")).resolve()
CHECKPOINT_URL = os.getenv("MODEL_CHECKPOINT_URL", "").strip()
CHECKPOINT_SHA256 = os.getenv("MODEL_CHECKPOINT_SHA256", "").strip().lower()
DOWNLOAD_TOKEN = os.getenv("MODEL_DOWNLOAD_TOKEN", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "12"))
TORCH_THREADS = max(1, int(os.getenv("TORCH_NUM_THREADS", "2")))
RETENTION_HOURS = float(os.getenv("JOB_RETENTION_HOURS", "24"))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(TORCH_THREADS)


def choose_device() -> torch.device:
    wanted = os.getenv("MODEL_DEVICE", "auto").strip().lower()
    if wanted not in {"", "auto"}:
        return torch.device(wanted)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_checkpoint() -> Path:
    if CHECKPOINT.exists():
        if not CHECKPOINT_SHA256 or sha256(CHECKPOINT) == CHECKPOINT_SHA256:
            return CHECKPOINT
        CHECKPOINT.unlink(missing_ok=True)
    if not CHECKPOINT_URL:
        raise RuntimeError("No checkpoint found. Set MODEL_CHECKPOINT_URL in Render after training latest.pt.")
    part = CHECKPOINT.with_suffix(CHECKPOINT.suffix + ".part")
    headers = {"User-Agent": "musicm8-render/1.0"}
    if DOWNLOAD_TOKEN:
        headers["Authorization"] = f"Bearer {DOWNLOAD_TOKEN}"
    req = urllib.request.Request(CHECKPOINT_URL, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as response, part.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if CHECKPOINT_SHA256 and sha256(part) != CHECKPOINT_SHA256:
            raise RuntimeError("Downloaded checkpoint SHA-256 does not match MODEL_CHECKPOINT_SHA256")
        os.replace(part, CHECKPOINT)
        return CHECKPOINT
    finally:
        part.unlink(missing_ok=True)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    seconds: float = Field(default=8.0, gt=0)
    bpm: float | None = Field(default=None, ge=30, le=300)
    key: str | None = Field(default=None, max_length=40)
    lyrics: str = Field(default="", max_length=12000)
    plan: dict[str, Any] | None = None
    temperature: float = Field(default=1.0, ge=0.05, le=2.5)
    top_k: int = Field(default=250, ge=0, le=2048)
    top_p: float = Field(default=0.95, gt=0, le=1.0)
    cfg_scale: float = Field(default=3.0, ge=0, le=12)
    seed: int = Field(default=0, ge=0, le=2_147_483_647)


class Job:
    def __init__(self, request: GenerateRequest):
        self.id = uuid.uuid4().hex
        self.request = request
        self.state = "queued"
        self.progress = "Waiting for generation worker"
        self.error: str | None = None
        self.output: Path | None = None
        self.created_at = self.updated_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "progress": self.progress,
            "error": self.error,
            "audio_url": f"/api/jobs/{self.id}/audio" if self.state == "done" else None,
        }


class Runtime:
    def __init__(self):
        self.device = choose_device()
        self.model = None
        self.codec = None
        self.loaded = False
        self.loading = False
        self.last_error: str | None = None
        self.lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "loading": self.loading,
            "device": str(self.device),
            "checkpoint_present": CHECKPOINT.exists(),
            "checkpoint_url_configured": bool(CHECKPOINT_URL),
            "last_error": self.last_error,
            "codec": getattr(getattr(self.codec, "info", None), "name", None),
        }

    def ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self.lock:
            if self.loaded:
                return
            self.loading = True
            self.last_error = None
            try:
                checkpoint = ensure_checkpoint()
                _, self.model, self.codec = load_model(checkpoint, self.device)
                self.loaded = True
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.loading = False

    @torch.inference_mode()
    def generate(self, req: GenerateRequest, output: Path) -> None:
        self.ensure_loaded()
        assert self.model is not None and self.codec is not None
        delays = self.model.cfg.resolved_delays()
        context_max = max(1, self.model.cfg.max_seq_len - max(delays, default=0)) / float(self.codec.info.frame_rate)
        hard_max = min(MAX_SECONDS, context_max)
        if req.seconds > hard_max + 1e-6:
            raise ValueError(f"This checkpoint supports at most {hard_max:.2f}s in this Render endpoint")
        frames = max(1, round(req.seconds * self.codec.info.frame_rate))
        if req.plan:
            data = dict(req.plan)
            data["prompt"] = req.prompt
            data["duration"] = req.seconds
            plan = SongPlan.from_dict(data)
        else:
            plan = heuristic_plan(req.prompt, req.seconds, req.bpm, req.key, lyrics=req.lyrics)
        plan.duration = req.seconds
        cond = plan_to_conditions(plan, frames, self.codec.info.frame_rate)
        codes = self.model.generate(
            req.prompt, frames, cond,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
            cfg_scale=req.cfg_scale, seed=req.seed,
        )
        wav = self.codec.decode(codes)
        wav = wav[..., : round(req.seconds * self.codec.info.sample_rate)].detach().cpu().float().clamp(-1, 1)
        audio = wav.numpy()
        if audio.ndim == 2:
            audio = audio.T
            if audio.shape[1] == 1:
                audio = audio[:, 0]
        sf.write(output, audio, int(self.codec.info.sample_rate), subtype="PCM_16")


runtime = Runtime()
jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()
job_queue: queue.Queue[str] = queue.Queue()


def require_key(value: str | None) -> None:
    if API_KEY and (not value or not secrets.compare_digest(value, API_KEY)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def cleanup() -> None:
    cutoff = time.time() - RETENTION_HOURS * 3600
    with jobs_lock:
        old = [job_id for job_id, job in jobs.items() if job.created_at < cutoff]
        for job_id in old:
            job = jobs.pop(job_id)
            if job.output:
                job.output.unlink(missing_ok=True)


def worker() -> None:
    while True:
        job_id = job_queue.get()
        job = None
        try:
            with jobs_lock:
                job = jobs.get(job_id)
            if job is None:
                continue
            job.state = "running"
            job.progress = "Loading model and generating audio"
            out = OUTPUT_DIR / f"{job.id}.wav"
            runtime.generate(job.request, out)
            job.output = out
            job.state = "done"
            job.progress = "Generation complete"
            job.updated_at = time.time()
        except Exception as exc:
            if job is not None:
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.progress = "Generation failed"
                job.updated_at = time.time()
            traceback.print_exc()
        finally:
            cleanup()
            job_queue.task_done()


threading.Thread(target=worker, name="music-generation-worker", daemon=True).start()
app = FastAPI(title="Musicm8 API", version="1.0")


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((APP_DIR / "static/index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "Musicm8"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {"service": "Musicm8", "runtime": runtime.status(), "queue_size": job_queue.qsize(), "max_seconds": MAX_SECONDS, "api_key_required": bool(API_KEY)}


@app.post("/api/warmup")
def warmup(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    require_key(x_api_key)
    runtime.ensure_loaded()
    return {"ok": True, "runtime": runtime.status()}


@app.post("/api/jobs")
def create_job(req: GenerateRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    require_key(x_api_key)
    if req.seconds > MAX_SECONDS:
        raise HTTPException(status_code=422, detail=f"seconds must be <= {MAX_SECONDS}")
    job = Job(req)
    with jobs_lock:
        jobs[job.id] = job
    job_queue.put(job.id)
    return job.snapshot()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    require_key(x_api_key)
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.snapshot()


@app.get("/api/jobs/{job_id}/audio")
def get_audio(job_id: str, x_api_key: str | None = Header(default=None)) -> FileResponse:
    require_key(x_api_key)
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.state != "done" or not job.output or not job.output.exists():
        raise HTTPException(status_code=409, detail="Audio is not ready")
    return FileResponse(job.output, media_type="audio/wav", filename=f"musicm8-{job.id[:8]}.wav")
