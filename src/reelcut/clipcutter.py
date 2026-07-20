"""Clip planning (pure) + reel writing (ffmpeg wrapper).

Events -> padded/merged/split clips -> highlights.mp4 + highlights.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import ReelcutConfig
from .types import Clip, IdentityPoint, InvolvementEvent, ScorePoint


def plan_clips(
    events: list[InvolvementEvent],
    scores: list[ScorePoint],
    identity: list[IdentityPoint],
    video_duration_s: float,
    cfg: ReelcutConfig,
) -> list[Clip]:
    """Pure planning core.

    * pad each event by cfg.clip_pad_s both sides, clamp to [0, duration]
    * merge clips whose gap < cfg.clip_merge_gap_s (reasons/tags unioned,
      score = max)
    * enforce min length cfg.clip_min_s (extend symmetrically, clamped)
    * enforce max length cfg.clip_max_s by splitting at the lowest smoothed
      score valley inside the clip (each piece re-checked against max)
    * clip.confidence = mean identity confidence over the clip's span
    * output sorted, non-overlapping
    """
    raise NotImplementedError


def write_highlights_json(
    clips: list[Clip], out_path: Path, source_video: Path, cfg: ReelcutConfig
) -> None:
    """Schema:
    {"source": str, "generated_by": "reelcut", "clips": [
        {"start_s": float, "end_s": float, "score": float,
         "reasons": [...], "confidence": float}]}
    """
    raise NotImplementedError


def cut_reel(
    clips: list[Clip], source_video: Path, out_path: Path, work_dir: Path
) -> None:
    """Cut each clip from the source at original quality and concat.

    Use ffmpeg stream copy (-c copy) with the concat demuxer where possible;
    fall back to re-encode (libx264, crf 18) if stream copy fails (e.g.
    cut points not on keyframes producing broken output is acceptable for
    v1 ONLY via re-encode fallback — prefer accurate cuts over speed:
    use -ss before -i with re-encode for frame-accurate cuts).
    Segments go under work_dir; concat list file too.
    """
    raise NotImplementedError
