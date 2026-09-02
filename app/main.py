"""DepthWizard web API + static frontend server.

Endpoints:
  POST /api/process                  upload image (+bbox, +reference, +gcp) -> {job_id}
  GET  /api/jobs/{jid}               poll status / progress / metrics / asset urls
  POST /api/jobs/{jid}/refit         {points:[{x,y,h},...]} -> recalibrated metrics
  GET  /api/jobs/{jid}/asset/{name}  web asset (heights.bin, tex.jpg, header.json, ...)
  GET  /api/jobs/{jid}/download      original full-resolution DSM GeoTIFF
  DELETE /api/jobs/{jid}             cleanup
  GET  /                             landing page
  GET  /app                          Three.js terrain viewer frontend

Runs standalone: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import threading
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from depthwizard import calibrate

from . import jobs
from .jobs import JobStore, WEB_ROOT

app = FastAPI(title="DepthWizard", version="0.1.0")
store = JobStore()

_SAMPLES = {
    "hilly": {
        "label": "Hilly (Asheville)",
        "dir": "data/external/hilly_asheville",
        "image": "input.png",
        "reference": "reference.tif",
        "meta": "meta.json",
    },
    "forest": {
        "label": "Forest (GSMNP)",
        "dir": "data/external/forest_gsmnp",
        "image": "input.png",
        "reference": "reference.tif",
        "meta": "meta.json",
    },
    "urban": {
        "label": "Urban (GAMUS DC)",
        "dir": "data/heldout",
        "image": "DC_11_16.png",
        "reference": "DC_11_16_ndsm.tif",
        "meta": None,
    },
}

# Static frontend + vendored Three.js.
# NOTE: mounted AFTER the /api routes (below), because Starlette resolves routes
# in declaration order and a "/" mount must not shadow the API.
_STATIC = Path(__file__).resolve().parent.parent / "static"


_worker_lock = threading.Lock()


def _run_worker(job) -> None:
    with _worker_lock:
        store.process(job)


@app.post("/api/process")
async def process(
    image: UploadFile = File(...),
    bbox: str | None = Form(None),
    gcp: str | None = Form(None),
    reference: UploadFile | None = File(None),
):
    """Accept an upload, launch the pipeline in a background worker, return job id."""
    suffix = _suffix(image.filename)
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise HTTPException(400, f"unsupported image type: {suffix} (use PNG/JPG/GeoTIFF)")
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(400, "empty upload")

    reference_bytes = None
    ref_suffix = ".tif"
    if reference is not None:
        reference_bytes = await reference.read()

    job = store.create(
        image_bytes,
        suffix,
        bbox,
        reference_bytes=reference_bytes,
        reference_suffix=ref_suffix,
        gcp=gcp,
    )
    threading.Thread(target=_run_worker, args=(job,), daemon=True).start()
    return {"job_id": job.id, "status": job.status}


@app.get("/api/jobs/{jid}")
async def job_status(jid: str):
    job = store.get(jid)
    if job is None:
        raise HTTPException(404, "job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "metrics": job.metrics,
        "warnings": job.warnings,
        "error": job.error,
        "header": job.header,
        "created": job.created,
    }


@app.post("/api/jobs/{jid}/refit")
async def refit(jid: str, body: dict):
    job = store.get(jid)
    if job is None:
        raise HTTPException(404, "job not found")
    points, heights = [], []
    for pt in body.get("points", []):
        points.append((float(pt["x"]), float(pt["y"])))
        heights.append(float(pt["h"]))
    if not points:
        raise HTTPException(400, "need at least one GCP point {x,y,h}")
    threading.Thread(target=lambda: _with_lock_refit(job, points, heights), daemon=True).start()
    return {"job_id": job.id, "status": "running"}


def _with_lock_refit(job, points, heights) -> None:
    with _worker_lock:
        store.refit(job, points, heights)


@app.get("/api/jobs/{jid}/asset/{name:path}")
async def asset(jid: str, name: str):
    """Serve an exported web asset from this job's directory (e.g. web/heights.bin)."""
    job = store.get(jid)
    if job is None:
        raise HTTPException(404, "job not found")
    p = (job.out_dir / name).resolve()
    if not p.is_relative_to(job.out_dir.resolve()) or not p.is_file():
        raise HTTPException(404, "asset not found")
    return FileResponse(p)


@app.get("/api/jobs/{jid}/download")
async def download(jid: str):
    job = store.get(jid)
    if job is None:
        raise HTTPException(404, "job not found")
    p = job.out_dir / "input_dsm.tif"
    if not p.exists():
        raise HTTPException(404, "DSM not ready yet")
    return FileResponse(p, media_type="image/tiff", filename="depthwizard_dsm.tif")


@app.delete("/api/jobs/{jid}")
async def delete_job(jid: str):
    job = store.get(jid)
    if job is None:
        raise HTTPException(404, "job not found")
    store.delete(jid)
    return {"ok": True}


def _suffix(name: str) -> str:
    return Path(name or "x").suffix.lower()


@app.get("/api/samples")
async def samples():
    return {name: spec["label"] for name, spec in _SAMPLES.items()}


@app.post("/api/samples/{name}/process")
async def samples_process(name: str):
    """Kick off a job from a bundled sample (no browser upload needed)."""
    spec = _SAMPLES.get(name)
    if spec is None:
        raise HTTPException(404, f"unknown sample '{name}'")
    root = Path(__file__).resolve().parent.parent
    sdir = root / spec["dir"]

    bbox = None
    meta_path = (sdir / spec["meta"]) if spec.get("meta") else None
    if meta_path and meta_path.exists():
        bbox = " ".join(str(v) for v in json.loads(meta_path.read_text())["bbox"])

    image_path = sdir / spec["image"]
    reference_path = sdir / spec["reference"] if spec.get("reference") else None

    image_bytes = image_path.read_bytes()
    reference_bytes = reference_path.read_bytes() if reference_path else None

    job = store.create(
        image_bytes,
        image_path.suffix.lower(),
        bbox,
        reference_bytes=reference_bytes,
        reference_suffix=reference_path.suffix.lower() if reference_path else ".tif",
    )
    threading.Thread(target=_run_worker, args=(job,), daemon=True).start()
    return {"job_id": job.id, "status": job.status, "label": spec["label"]}


# Explicit page routes — declared BEFORE the static mount so they are not shadowed.
@app.get("/", response_class=None)
async def landing_page():
    return FileResponse(_STATIC / "landing.html")


@app.get("/app", response_class=None)
async def app_page():
    return FileResponse(_STATIC / "app.html")


# Static asset mount — serves /css/*, /js/*, /vendor/*, etc.
# Must come AFTER explicit page routes and /api/* so it does not shadow them.
app.mount("/", StaticFiles(directory=_STATIC), name="static")