"""Roboflow Batch Processing — GPU offload for stage 1.

Instead of running the workflow locally (CPU-bound, ~20x slower than
realtime with the OCR branch), the video is uploaded to Roboflow, a cloud
GPU runs the saved `reelcut-tracking` workflow over it, and the exported
JSONL results feed the exact same pipeline stages 2-4 locally (seconds).

Everything here is programmable — a web backend calls these same functions
from an upload handler; the CLI (`scripts/run_batch.py`) is just the
developer surface. Flow:

    stage_video() -> submit_job() -> wait_for_job() -> export_results()
    -> BatchResultsClient(results_dir) as the pipeline's stage-1 client

Torso color histograms (identity evidence) are computed locally by decoding
only the sampled frames from the source video — decode-only is cheap; it is
the model inference that needed the GPU.

NOTE: the exported record schema is validated against the first real job;
`_observation_from_record` parses tolerantly and raises a clear error listing
the keys it saw if Roboflow's export shape differs from expectations.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from .config import ReelcutConfig
from .ffmpeg import probe
from .types import BallObs, BBox, FrameObservation, OcrRead, PlayerObs
from .workflow_client import torso_histogram


def _rf_cloud(args: list[str]) -> str:
    """Run `inference rf-cloud <args>` (venv console script), return stdout."""
    proc = subprocess.run(
        ["inference", "rf-cloud", *args], capture_output=True, text=True
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        raise RuntimeError(f"rf-cloud {' '.join(args[:2])} failed:\n{tail}")
    return proc.stdout


def stage_video(video: Path, batch_id: str) -> None:
    """Upload the video into a Data Staging batch."""
    _rf_cloud([
        "data-staging", "create-batch-of-videos",
        "--videos-dir", str(video.parent),
        "--batch-id", batch_id,
    ])


def submit_job(
    batch_id: str,
    workflow_id: str,
    sample_fps: float,
    machine_type: str = "gpu",
    job_id: str | None = None,
    notifications_url: str | None = None,
) -> str:
    """Submit the processing job; returns the job id."""
    args = [
        "batch-processing", "process-videos-with-workflow",
        "--batch-id", batch_id,
        "--workflow-id", workflow_id,
        "--machine-type", machine_type,
        "--max-video-fps", str(int(round(sample_fps))),
        "--aggregation-format", "jsonl",
    ]
    if job_id:
        args += ["--job-id", job_id]
    if notifications_url:
        args += ["--notifications-url", notifications_url]
    out = _rf_cloud(args)
    for token in out.replace("=", " ").replace(":", " ").split():
        if token.startswith("job"):  # CLI prints the created job id
            continue
    # The CLI prints a line containing the job id; fall back to provided one.
    for line in out.splitlines():
        low = line.lower()
        if "job" in low and ("id" in low or "created" in low):
            candidates = [t.strip(" .,'\"") for t in line.split() if len(t) > 8]
            if candidates:
                return candidates[-1]
    if job_id:
        return job_id
    raise RuntimeError(f"could not parse job id from rf-cloud output:\n{out}")


def job_status(job_id: str) -> str:
    return _rf_cloud([
        "batch-processing", "show-job-details", "--job-id", job_id
    ])


def wait_for_job(job_id: str, poll_s: float = 30.0, timeout_s: float = 4 * 3600) -> None:
    """Poll until the job reports success; raise on failure/timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        details = job_status(job_id).lower()
        if "fail" in details or "error" in details and "0 error" not in details:
            raise RuntimeError(f"batch job {job_id} failed:\n{details[-1500:]}")
        if "success" in details or "completed" in details or "finished" in details:
            return
        time.sleep(poll_s)
    raise TimeoutError(f"batch job {job_id} still running after {timeout_s}s")


def export_results(batch_id: str, target_dir: Path) -> Path:
    """Download the job's output batch (JSONL aggregation) to target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    _rf_cloud([
        "data-staging", "export-batch",
        "--batch-id", batch_id,
        "--target-dir", str(target_dir),
    ])
    return target_dir


# --------------------------------------------------------------------------- #
# Results import
# --------------------------------------------------------------------------- #

_FRAME_KEYS = ("frame_number", "frame_offset", "frame_id", "frame")
_OUTPUT_WRAPPERS = ("outputs", "output", "result", "results", "predictions")


def _find_frame_index(record: dict[str, Any]) -> int | None:
    for k in _FRAME_KEYS:
        if k in record and record[k] is not None:
            try:
                return int(record[k])
            except (TypeError, ValueError):
                continue
    return None


def _outputs_of(record: dict[str, Any]) -> dict[str, Any]:
    """The dict holding our workflow output fields, wherever it nests."""
    for k in _OUTPUT_WRAPPERS:
        inner = record.get(k)
        if isinstance(inner, dict) and (
            "tracked_players" in inner or "ball" in inner
        ):
            return inner
    return record


def _predictions_list(det: Any) -> list[dict[str, Any]]:
    """Serialized detections -> list of prediction dicts (tolerant)."""
    if det is None:
        return []
    if isinstance(det, dict):
        preds = det.get("predictions", det.get("detections"))
        if isinstance(preds, list):
            return [p for p in preds if isinstance(p, dict)]
        return []
    if isinstance(det, list):
        return [p for p in det if isinstance(p, dict)]
    return []


def _bbox_of(pred: dict[str, Any]) -> BBox | None:
    """Roboflow serialization: center x/y + width/height."""
    try:
        w, h = float(pred["width"]), float(pred["height"])
        return BBox(float(pred["x"]) - w / 2.0, float(pred["y"]) - h / 2.0, w, h)
    except (KeyError, TypeError, ValueError):
        return None


def _observation_from_record(
    record: dict[str, Any],
    frame_index: int,
    timestamp_s: float,
    frame_w: int,
    frame_h: int,
    frame_bgr: np.ndarray | None,
    cfg: ReelcutConfig,
) -> FrameObservation:
    from .workflow_client import _canonical_class  # shared class mapping

    outputs = _outputs_of(record)
    players: list[PlayerObs] = []
    ocr: list[OcrRead] = []
    texts = outputs.get("jersey_texts") or []
    min_box_h = cfg.min_player_box_h_frac * frame_h

    for k, pred in enumerate(_predictions_list(outputs.get("tracked_players"))):
        bbox = _bbox_of(pred)
        tid = pred.get("tracker_id")
        if bbox is None or tid is None or bbox.h < min_box_h:
            continue
        tid = int(tid)
        hist = (
            torso_histogram(frame_bgr, (bbox.x, bbox.y, bbox.w, bbox.h))
            if frame_bgr is not None
            else None
        )
        velocity = pred.get("velocity") or pred.get("smoothed_velocity")
        players.append(PlayerObs(
            track_id=tid,
            bbox=bbox,
            class_name=_canonical_class(pred.get("class"), cfg),
            confidence=float(pred.get("confidence") or 0.0),
            speed=(
                float(pred["smoothed_speed"]) if pred.get("smoothed_speed") is not None
                else float(pred["speed"]) if pred.get("speed") is not None else None
            ),
            velocity=(
                (float(velocity[0]), float(velocity[1]))
                if isinstance(velocity, (list, tuple)) and len(velocity) >= 2
                else None
            ),
            torso_hsv=hist,
        ))
        text = texts[k] if k < len(texts) else None
        if text:
            ocr.append(OcrRead(track_id=tid, text=str(text), confidence=1.0))

    ball: BallObs | None = None
    ball_preds = _predictions_list(outputs.get("ball"))
    if ball_preds:
        best = max(ball_preds, key=lambda p: float(p.get("confidence") or 0.0))
        bb = _bbox_of(best)
        if bb is not None:
            ball = BallObs(bbox=bb, confidence=float(best.get("confidence") or 0.0))

    goal_boxes = tuple(
        b for b in (_bbox_of(p) for p in _predictions_list(outputs.get("goal_detections")))
        if b is not None
    )

    return FrameObservation(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        frame_w=frame_w,
        frame_h=frame_h,
        players=tuple(players),
        ball=ball,
        ocr=tuple(ocr),
        goal_boxes=goal_boxes,
    )


class BatchResultsClient:
    """Stage-1 client over exported Batch Processing results.

    Reads every ``*.jsonl`` under ``results_dir`` (one record per processed
    frame), pairs records with locally-decoded source frames to compute torso
    histograms, and yields FrameObservations in frame order. If the source
    video is missing, histograms are None (identity falls back to seed + OCR
    + kinematics).
    """

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)

    def _records(self) -> list[tuple[int, dict[str, Any]]]:
        rows: list[tuple[int, dict[str, Any]]] = []
        files = sorted(self.results_dir.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(
                f"no .jsonl batch results under {self.results_dir}"
            )
        for path in files:
            for i, line in enumerate(path.read_text().splitlines()):
                if not line.strip():
                    continue
                record = json.loads(line)
                idx = _find_frame_index(record)
                rows.append((idx if idx is not None else i, record))
        first = rows[0][1]
        if _outputs_of(first) is first and "tracked_players" not in first:
            raise ValueError(
                "unrecognized batch record shape; top-level keys: "
                f"{sorted(first.keys())[:12]} — adjust batch._OUTPUT_WRAPPERS/"
                "_FRAME_KEYS to match this export."
            )
        rows.sort(key=lambda r: r[0])
        return rows

    def run(self, video: Path, cfg: ReelcutConfig) -> Iterator[FrameObservation]:
        rows = self._records()
        if video.exists():
            meta = probe(video)
            fps, frame_w, frame_h = meta.fps, meta.width, meta.height
            cap: cv2.VideoCapture | None = cv2.VideoCapture(str(video))
        else:
            fps, frame_w, frame_h = 30.0, 1280, 720
            cap = None
        try:
            for frame_index, record in rows:
                frame_bgr = None
                if cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, img = cap.read()
                    if ok:
                        frame_bgr = img
                        frame_h, frame_w = img.shape[:2]
                yield _observation_from_record(
                    record,
                    frame_index=frame_index,
                    timestamp_s=frame_index / fps,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    frame_bgr=frame_bgr,
                    cfg=cfg,
                )
        finally:
            if cap is not None:
                cap.release()
