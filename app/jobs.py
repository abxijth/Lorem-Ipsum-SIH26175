"""Background job store + serialized worker for the web API.

Jobs are processed one at a time (shared GPU), with coarse progress stages
surfaced through `progress_cb`. Each job keeps its own working directory under
results/web/<id> holding the uploaded inputs, pipeline outputs, persisted
calibration/terrain, and exported browser assets.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from depthwizard import calibrate, export_web, georef, pipeline, validate
from depthwizard.calibrate import load_calibration_any, apply_calibration_any
from depthwizard.depth import DepthEstimator
from depthwizard.dsm import assemble, write_geotiff
from depthwizard.config import RESULTS_DIR

WEB_ROOT = RESULTS_DIR / "web"


@dataclass
class Job:
    id: str
    input_path: Path
    out_dir: Path
    stage_dir: Path
    bbox: str | None = None
    gcp: str | None = None
    reference_path: Path | None = None
    status: str = "queued"          # queued | running | done | error
    progress: str = ""
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    header: dict = field(default_factory=dict)
    error: str = ""
    created: float = field(default_factory=time.time)


class JobStore:
    """Thread-safe job registry + single-worker queue."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._estimator: DepthEstimator | None = None

    def create(
        self,
        image_bytes: bytes,
        image_suffix: str,
        bbox: str | None,
        reference_bytes: bytes | None = None,
        reference_suffix: str = ".tif",
        gcp: str | None = None,
    ) -> Job:
        jid = uuid.uuid4().hex[:12]
        out_dir = WEB_ROOT / jid
        stage_dir = out_dir / "stage"
        out_dir.mkdir(parents=True, exist_ok=True)
        stage_dir.mkdir(parents=True, exist_ok=True)

        input_path = out_dir / f"input{image_suffix}"
        input_path.write_bytes(image_bytes)
        reference_path = None
        if reference_bytes:
            reference_path = out_dir / f"reference{reference_suffix}"
            reference_path.write_bytes(reference_bytes)

        job = Job(
            id=jid,
            input_path=input_path,
            out_dir=out_dir,
            stage_dir=stage_dir,
            bbox=bbox,
            gcp=gcp,
            reference_path=reference_path,
        )
        with self._lock:
            self._jobs[jid] = job
        return job

    def get(self, jid: str) -> Job | None:
        with self._lock:
            return self._jobs.get(jid)

    def _update(self, jid: str, **kw) -> None:
        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                return
            for k, v in kw.items():
                setattr(job, k, v)

    def _estimator_once(self) -> DepthEstimator:
        if self._estimator is None:
            with self._lock:
                if self._estimator is None:
                    self._estimator = DepthEstimator()
        return self._estimator

    def process(self, job: Job) -> None:
        """Run the full pipeline + web export in the caller thread."""
        self._update(job.id, status="running", progress="georef")
        try:
            est = self._estimator_once()

            def _cb(stage: str) -> None:
                self._update(job.id, progress=stage)

            result = pipeline.run(
                job.input_path,
                bbox=job.bbox,
                reference_path=job.reference_path,
                out_dir=job.out_dir,
                progress_cb=_cb,
                stage_dir=job.stage_dir,
                estimator=est,
                gcp=job.gcp,
            )
            self._export(job, result.metrics)
            self._update(job.id, status="done", progress="done")
        except Exception as e:  # noqa: BLE001 - surface any failure to the UI
            self._update(job.id, status="error", error=str(e))

    def refit(self, job: Job, points: list, heights: list) -> None:
        """Re-calibrate from user GCPs using the persisted depth + terrain."""
        assert points and heights and len(points) == len(heights)
        self._update(job.id, status="running", progress="calibrate")
        try:
            depth = np.load(job.out_dir / "input_depth.npy")
            base = load_calibration_any(job.stage_dir / "base_calib.json")
            calib = calibrate.gcp_refit(depth, points, heights, base_calib=base)
            calib.to_json(job.stage_dir / "effective_calib.json")

            terrain = None
            tpath = job.stage_dir / "terrain.npy"
            if tpath.exists():
                terrain = np.load(tpath)

            dsm = assemble(depth, calib, terrain)
            gr = georef.load(job.input_path, job.bbox)
            write_geotiff(dsm, gr, job.out_dir / "input_dsm.tif")

            metrics: dict = {}
            warnings: list[str] = []
            if len(points) < 2:
                warnings.append("1 GCP -> intercept-only shift (slope kept from base calibration)")
            if job.reference_path and job.reference_path.exists():
                ref, _ = validate.load_raster_flat(job.reference_path)
                if ref.shape != dsm.shape:
                    from PIL import Image
                    ref = np.asarray(
                        Image.fromarray(ref).resize((dsm.shape[1], dsm.shape[0]), Image.BILINEAR)
                    )
                metrics = validate.compare(dsm, ref)
                validate.error_heatmap(dsm, ref, gr, job.out_dir / "input_error.tif")

            self._export(job, metrics)
            self._update(job.id, status="done", progress="done", metrics=metrics,
                         warnings=warnings, error=f"GCP calibration updated")
        except Exception as e:  # noqa: BLE001
            self._update(job.id, status="error", error=str(e))

    def _export(self, job: Job, metrics: dict) -> None:
        self._update(job.id, progress="export")
        gr = georef.load(job.input_path, job.bbox)
        dsm, _ = validate.load_raster_flat(job.out_dir / "input_dsm.tif")
        depth = np.load(job.out_dir / "input_depth.npy")
        calib = load_calibration_any(job.stage_dir / "effective_calib.json")
        struct = apply_calibration_any(depth, calib)

        err = None
        err_path = job.out_dir / "input_error.tif"
        if err_path.exists():
            err, _ = validate.load_raster_flat(err_path)
        header = export_web.export_web_assets(
            gr, dsm, struct, job.out_dir, err=err, metrics=metrics
        )
        self._update(job.id, header=header, metrics=metrics)

    def delete(self, jid: str) -> None:
        with self._lock:
            self._jobs.pop(jid, None)
        shutil.rmtree(WEB_ROOT / jid, ignore_errors=True)

    def cleanup(self, max_age_s: float = 7200.0) -> None:
        """Remove jobs (and their dirs) idle for > max_age_s."""
        now = time.time()
        stale = []
        with self._lock:
            for jid, job in self._jobs.items():
                if job.status in ("done", "error") and now - job.created > max_age_s:
                    stale.append(jid)
            for jid in stale:
                del self._jobs[jid]
        for jid in stale:
            shutil.rmtree(WEB_ROOT / jid, ignore_errors=True)