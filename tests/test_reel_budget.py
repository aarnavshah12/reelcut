"""Adaptive reel budget + goal-box sanity filtering."""
from __future__ import annotations

from reelcut.config import ReelcutConfig
from reelcut.scoring import adaptive_threshold, score_opportunities
from reelcut.types import BallObs, BBox, ScorePoint

from fixtures import make_frame

CFG = ReelcutConfig()


def points(scores):
    return [ScorePoint(i / 5.0, s, ()) for i, s in enumerate(scores)]


def test_threshold_unchanged_when_coverage_is_sane():
    pts = points([0.9] * 10 + [0.0] * 90)      # 10% hot
    assert adaptive_threshold(pts, 0.30, 0.12, 0.20) == 0.30


def test_threshold_rises_when_scores_saturate():
    pts = points([0.8] * 80 + [0.1] * 20)      # 80% hot -> way over max
    t = adaptive_threshold(pts, 0.30, 0.12, 0.20)
    assert t > 0.30
    passing = sum(1 for p in pts if p.score >= t) / len(pts)
    assert passing <= 0.25                      # brought near the target band


def test_threshold_keeps_score_ordering_top_slice():
    # varied scores: the top ~12% must be the survivors
    vals = [i / 100 for i in range(100)]
    pts = points(vals)
    t = adaptive_threshold(pts, 0.30, 0.12, 0.20)
    survivors = [p.score for p in pts if p.score >= t]
    assert len(survivors) <= 15 and min(survivors) >= 0.85


def test_giant_goal_boxes_are_ignored():
    ball = BallObs(bbox=BBox(300, 200, 14, 14), confidence=0.8)
    huge_goal = BBox(0, 0, 640, 350)            # h ~ 97% of frame: nonsense
    frame = make_frame(0, [], ball)
    frame = type(frame)(**{**{f: getattr(frame, f) for f in frame.__dataclass_fields__},
                           "goal_boxes": (huge_goal,)})
    pts = score_opportunities([frame], CFG)
    assert pts[0].score == 0.0 and pts[0].tags == ()


def test_reasonable_goal_boxes_still_score():
    ball = BallObs(bbox=BBox(100, 300, 14, 14), confidence=0.8)
    goal = BBox(60, 260, 90, 100)               # h=100 of 720: fine
    frame = make_frame(0, [], ball)
    frame = type(frame)(**{**{f: getattr(frame, f) for f in frame.__dataclass_fields__},
                           "goal_boxes": (goal,)})
    pts = score_opportunities([frame], CFG)
    assert pts[0].score > 0.5 and "goal_chance" in pts[0].tags
