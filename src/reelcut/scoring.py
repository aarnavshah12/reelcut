"""Involvement scoring — pure functions, no I/O.

Input: identity timeline (target presence per sampled timestamp) + per-frame
observations (ball position). Output: smoothed 0..1 score per timestamp with
reason tags, then thresholded into events.

All distances are normalized by the target bbox height at that timestamp
("player-heights"), so constants are resolution-independent.
"""
from __future__ import annotations

from .config import ReelcutConfig
from .types import (
    BallObs,
    FrameObservation,
    IdentityPoint,
    InvolvementEvent,
    ScorePoint,
)


def interpolate_ball(
    frames: list[FrameObservation], max_gap_s: float
) -> list[FrameObservation]:
    """Fill short ball gaps by linear interpolation.

    For runs of consecutive frames with ``ball is None`` bounded on both sides
    by real ball detections and shorter than ``max_gap_s``, insert interpolated
    ``BallObs(..., interpolated=True)`` with confidence linearly decayed toward
    the middle of the gap. Longer gaps stay None. Returns new list; input
    unmodified.
    """
    raise NotImplementedError


def score_timeline(
    identity: list[IdentityPoint],
    frames: list[FrameObservation],
    cfg: ReelcutConfig,
) -> list[ScorePoint]:
    """Compute the raw involvement score at every identity timestamp.

    Components (see config.SportConfig for constants):
      * proximity: 1.0 inside ``ball_near_dist`` player-heights, linear falloff
        to 0 at ``ball_far_dist``; tag "near_ball" when > 0.5.
      * possession: ball inside ``possession_radius`` for >= ``possession_min_s``
        AND ball speed correlates with target speed -> add w_possession,
        tag "possession".
      * touch: ball direction change > ``touch_direction_change_deg`` while
        inside ``touch_radius`` -> add w_touch, tag "touch".
      * sprint fallback: target speed > ``sprint_speed`` player-heights/s ->
        add w_sprint, tag "sprint". Works when ball is unknown.

    Raw score = clamp01(weighted sum) * identity confidence at that timestamp.
    Ball speed/direction are derived from consecutive ball positions; frames
    where the ball is interpolated still count for proximity but not for touch.
    Timestamps where identity confidence is 0 (target absent) score 0.
    """
    raise NotImplementedError


def smooth_scores(
    points: list[ScorePoint], window_s: float
) -> list[ScorePoint]:
    """Centered rolling-mean over ``window_s``; tags are unioned over the window."""
    raise NotImplementedError


def extract_events(
    points: list[ScorePoint], threshold: float, min_duration_s: float
) -> list[InvolvementEvent]:
    """Contiguous runs of score >= threshold lasting >= min_duration_s.

    Event tags = union of member tags; peak/mean over member scores.
    """
    raise NotImplementedError
