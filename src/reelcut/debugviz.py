"""Debug video renderer (local, from cached stage outputs).

Draws on sampled frames: all tracked players (thin gray box + track id),
ball (yellow circle), target (THICK VIOLET box — Roboflow Violet 600
#7C3AED -> BGR (237, 58, 124)), identity confidence + involvement score bar,
active tags. Output plays at the sampling fps so time maps 1:1 to analysis.
"""
from __future__ import annotations

from pathlib import Path

from .config import ReelcutConfig
from .types import FrameObservation, IdentityPoint, ScorePoint

VIOLET_BGR = (237, 58, 124)   # Roboflow Violet 600 (#7C3AED) in BGR


def render_debug_video(
    video: Path,
    out_path: Path,
    frames: list[FrameObservation],
    identity: list[IdentityPoint],
    scores: list[ScorePoint] | None,
    cfg: ReelcutConfig,
) -> None:
    """Seek through the source with cv2.VideoCapture, draw overlays for each
    cached FrameObservation, write with cv2.VideoWriter (mp4v). Missing
    scores (None) -> render identity info only."""
    raise NotImplementedError
