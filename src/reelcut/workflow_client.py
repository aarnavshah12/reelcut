"""Stage-1 clients: turn a video into per-frame FrameObservations.

Two implementations behind one Protocol:

* StubWorkflowClient — deterministic synthetic "game" (seeded RNG); lets every
  downstream stage run end-to-end with no Roboflow account, network, or GPU.
* RoboflowWorkflowClient — the real thing: `inference.InferencePipeline`
  running our saved Roboflow workflow (detection -> filters -> BoT-SORT ->
  class lock -> stabilizer -> velocity -> OCR branch) at cfg.sample_fps.

The client is also where torso HSV histograms are computed (locally, from the
frame + player bbox — cheap numpy, no extra model), because identity stitching
needs an appearance descriptor and the frames are already in hand here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from .config import ReelcutConfig
from .types import FrameObservation

# Histogram binning shared with stitching.team_color_reference:
# HSV, H in 12 bins x S in 4 bins (V dropped for lighting robustness),
# flattened to 48 floats, L1-normalized.
HIST_H_BINS = 12
HIST_S_BINS = 4


def torso_histogram(frame_bgr: np.ndarray, bbox_xywh: tuple[float, float, float, float]) -> tuple[float, ...]:
    """HSV histogram of the upper-torso region of a player bbox.

    Torso region: central 60% width, rows 15%..50% of bbox height (skips head
    and legs). Returns the flattened normalized histogram; all-zeros if the
    region is degenerate/out of frame.
    """
    raise NotImplementedError


class WorkflowClient(Protocol):
    def run(self, video: Path, cfg: ReelcutConfig) -> Iterator[FrameObservation]:
        """Yield FrameObservations in frame order at ~cfg.sample_fps."""
        ...


class StubWorkflowClient:
    """Synthetic 22-player + ball game, deterministic per cfg.seed.

    * If ``video`` exists, duration/size come from ffmpeg.probe; otherwise a
      120 s / 1280x720 game is synthesized (so unit tests need no video file).
    * Players do smooth random walks (bounded to the frame); one designated
      "target" player (track_id 7) wears the jersey number the stub is
      configured to emit in OCR reads (~30% of frames, conf 0.5-0.95).
    * The ball bounces between random players, occasionally missing
      (None ~25% of frames) to exercise interpolation.
    * Tracks fragment (id shifts) every ~20 s to exercise stitching.
    """

    def __init__(self, seed: int | None = None, target_jersey: str = "10") -> None:
        self.seed = seed
        self.target_jersey = target_jersey

    def run(self, video: Path, cfg: ReelcutConfig) -> Iterator[FrameObservation]:
        raise NotImplementedError


class RoboflowWorkflowClient:
    """InferencePipeline over the saved workspace workflow.

    Contract with the workflow outputs (see workflows/reelcut_tracking.json):
      * "tracked_players": sv.Detections with tracker_id, class, confidence,
        and data keys "velocity"/"speed" from the Velocity block
      * "ball": sv.Detections (0 or 1 efter highest-confidence filtering)
      * "jersey_texts": list[str] aligned with tracked_players order (may be
        shorter/None-padded; treat missing as no read)
    Frames are sampled by InferencePipeline at max_fps=cfg.sample_fps; each
    emitted FrameObservation carries the SOURCE frame index and timestamp.
    """

    def __init__(
        self,
        api_key: str,
        workspace: str,
        workflow_id: str,
        api_url: str | None = None,
        image_input_name: str = "image",
    ) -> None:
        self.api_key = api_key
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.api_url = api_url
        self.image_input_name = image_input_name

    def run(self, video: Path, cfg: ReelcutConfig) -> Iterator[FrameObservation]:
        """Run InferencePipeline.init_with_workflow, collect results via an
        internal queue sink, convert each workflow result + video frame into a
        FrameObservation (computing torso_histogram locally per player), and
        yield in order. Blocks until the video is exhausted."""
        raise NotImplementedError
