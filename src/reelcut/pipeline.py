"""Stage orchestration: INFER -> IDENTITY -> SCORE -> CUT (+ optional debug).

Each stage: pure core (other modules) + thin cached I/O here. One-line health
summary printed per stage, e.g.:
  [stage1 infer]    3012 frames, ball in 61.2% of frames, 48 track ids
  [stage2 identity] 39 tracklets -> 6 target / 21 not_target / 12 unknown, coverage 71%
  [stage3 score]    14 events, mean score 0.44
  [stage4 cut]      9 clips, 96.5s total (8.0% of source)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Paths, ReelcutConfig
from .types import (
    Clip,
    FrameObservation,
    IdentityPoint,
    InvolvementEvent,
    LabeledTracklet,
    ScorePoint,
    TargetSpec,
)
from .workflow_client import WorkflowClient


@dataclass
class PipelineResult:
    frames: list[FrameObservation]
    labeled: list[LabeledTracklet]
    identity: list[IdentityPoint]
    scores: list[ScorePoint]
    events: list[InvolvementEvent]
    clips: list[Clip]


def run_pipeline(
    video: Path,
    spec: TargetSpec,
    cfg: ReelcutConfig,
    paths: Paths,
    client: WorkflowClient,
    force_stage: int | None = None,
    debug_video: bool = False,
) -> PipelineResult:
    """Run all stages with caching.

    * cache key from cache.video_cache_key(video, cfg.sample_fps, workflow ref)
      — for the stub client the workflow ref is "stub:<seed>".
    * if force_stage is given, cache.invalidate_from(force_stage) first.
    * stage 1 payload name "observations"; 2 "identity" (labeled + timeline);
      3 "scores" (points + events); 4 "clips".
    * stage 4 also writes highlights.mp4 + highlights.json via clipcutter.
    * debug_video -> debugviz.render_debug_video into paths.debug_mp4.
    """
    raise NotImplementedError
