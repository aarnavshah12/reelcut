"""Identity stitching — label tracklets target / not_target / unknown.

Pure functions, no I/O. Evidence sources, in trust order:
  1. seed click (pins one tracklet as TARGET)
  2. jersey OCR (matching number = strong positive; confident different
     number = strong negative)
  3. team-color histogram (cheap NOT_TARGET veto for opponents/referees)
  4. kinematic continuity (plausible hand-off between tracklets in time)

Ambiguity stays UNKNOWN: a reel of the wrong kid is worse than a missed play.
Never flip a tracklet to TARGET on appearance similarity alone.
"""
from __future__ import annotations

from .config import ReelcutConfig
from .types import (
    BBox,
    FrameObservation,
    IdentityLabel,
    IdentityPoint,
    LabeledTracklet,
    TargetSpec,
    Tracklet,
)


def build_tracklets(frames: list[FrameObservation]) -> list[Tracklet]:
    """Group per-frame PlayerObs into Tracklets by track_id.

    A gap of any length within one track_id stays one tracklet (the workflow
    tracker already decided identity); points are time-ordered. OCR reads are
    attached to their tracklet. Referee-class tracks are still returned (the
    labeling step will mark them NOT_TARGET).
    """
    raise NotImplementedError


def find_seed_tracklet(
    tracklets: list[Tracklet], spec: TargetSpec, frames: list[FrameObservation]
) -> int | None:
    """Return track_id of the tracklet best matching the user's seed click.

    Match = highest IoU with ``spec.seed_box`` among observations at (or
    nearest to, within +-5 frames of) ``spec.seed_frame_index``. Require
    IoU > 0.2; else None (caller errors out with a helpful message).
    """
    raise NotImplementedError


def hist_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Symmetric 0..1 distance between two normalized histograms
    (1 - intersection)."""
    raise NotImplementedError


def team_color_reference(color_name: str) -> tuple[float, ...]:
    """Synthesize the reference torso histogram for a named kit color using
    config.TEAM_COLORS bins. Same binning as the runner's torso extractor
    (workflow_client.torso_histogram)."""
    raise NotImplementedError


def label_tracklets(
    tracklets: list[Tracklet],
    spec: TargetSpec,
    frames: list[FrameObservation],
    cfg: ReelcutConfig,
) -> list[LabeledTracklet]:
    """Assign labels and confidences per the evidence rules.

    Rules (applied in order, first hit wins for hard labels):
      * seed tracklet -> TARGET, confidence 1.0, evidence {"seed": 1}.
      * >= cfg.ocr_neg_votes confident reads of a DIFFERENT number
        -> NOT_TARGET (evidence "ocr_neg").
      * referee class -> NOT_TARGET.
      * mean torso-hist distance to team reference > cfg.color_veto_dist
        -> NOT_TARGET (evidence "color").
      * >= cfg.ocr_pos_votes confident reads of the matching number
        -> TARGET, confidence ~0.9 (evidence "ocr_pos").
      * else UNKNOWN with a soft prior in evidence for the chaining step.
    """
    raise NotImplementedError


def chain_target_tracklets(
    labeled: list[LabeledTracklet], cfg: ReelcutConfig
) -> list[LabeledTracklet]:
    """Greedy kinematic chaining of UNKNOWN tracklets onto the TARGET chain.

    Sort TARGET tracklets by time. For gaps between consecutive confirmed
    TARGET tracklets, consider UNKNOWN tracklets that fit inside the gap and
    whose endpoints are kinematically plausible (distance between boxes <=
    cfg.link_max_dist player-heights scaled by the time gap; implied speed <=
    cfg.max_plausible_speed). Promote at most one non-overlapping chain per
    gap to TARGET with confidence = 0.5 * plausibility (never 1.0). Tracklets
    overlapping in time with a confirmed TARGET tracklet are never promoted.
    Returns a new list.
    """
    raise NotImplementedError


def identity_timeline(
    labeled: list[LabeledTracklet],
    frames: list[FrameObservation],
    cfg: ReelcutConfig,
) -> list[IdentityPoint]:
    """One IdentityPoint per frame timestamp.

    If a TARGET tracklet covers the timestamp: confidence = tracklet
    confidence, bbox/speed from its point at that frame. If two TARGET
    tracklets claim the same timestamp (shouldn't happen post-chaining),
    confidence 0 — conflicting identity is unknown identity. Otherwise
    confidence 0, bbox None. Target off-screen is NORMAL: zero, not an error.
    """
    raise NotImplementedError
