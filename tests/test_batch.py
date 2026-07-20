"""Batch Processing results import — record parsing and client iteration."""
from __future__ import annotations

import json

import pytest

from reelcut.batch import BatchResultsClient, _observation_from_record
from reelcut.config import ReelcutConfig

CFG = ReelcutConfig()


def record(frame=10, tracked=None, ball=None, goals=None, texts=None, wrap=None):
    outputs = {
        "tracked_players": {"predictions": tracked or []},
        "ball": {"predictions": ball or []},
        "goal_detections": {"predictions": goals or []},
        "jersey_texts": texts or [],
    }
    if wrap:
        return {"frame_number": frame, wrap: outputs}
    return {"frame_number": frame, **outputs}


def pred(x=100, y=100, w=40, h=90, cls="player", conf=0.8, tid=3, **extra):
    return {"x": x, "y": y, "width": w, "height": h, "class": cls,
            "confidence": conf, "tracker_id": tid, **extra}


def test_record_parses_players_ball_goals_ocr():
    rec = record(
        tracked=[pred(tid=3, smoothed_speed=12.5, velocity=[3.0, -4.0]),
                 pred(x=300, tid=None)],          # untracked -> dropped
        ball=[{"x": 50, "y": 60, "width": 12, "height": 12, "confidence": 0.4},
              {"x": 55, "y": 60, "width": 12, "height": 12, "confidence": 0.7}],
        goals=[{"x": 20, "y": 300, "width": 40, "height": 120, "confidence": 0.9}],
        texts=["10"],
    )
    obs = _observation_from_record(rec, 10, 10 / 30.0, 1280, 720, None, CFG)
    assert len(obs.players) == 1
    p = obs.players[0]
    assert p.track_id == 3 and p.speed == 12.5 and p.velocity == (3.0, -4.0)
    assert p.torso_hsv is None                     # no frame supplied
    assert obs.ball is not None and obs.ball.confidence == 0.7  # max conf wins
    assert len(obs.goal_boxes) == 1 and obs.goal_boxes[0].h == 120
    assert obs.ocr and obs.ocr[0].text == "10" and obs.ocr[0].track_id == 3


def test_record_output_wrapper_variants():
    for wrap in (None, "outputs", "result"):
        rec = record(tracked=[pred()], wrap=wrap)
        obs = _observation_from_record(rec, 1, 0.033, 1280, 720, None, CFG)
        assert len(obs.players) == 1, f"wrapper {wrap!r} not handled"


def test_client_orders_by_frame_and_errors_on_alien_shape(tmp_path):
    rows = [record(frame=20, tracked=[pred(tid=2)]),
            record(frame=5, tracked=[pred(tid=1)])]
    (tmp_path / "video_a.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    client = BatchResultsClient(tmp_path)
    obs = list(client.run(tmp_path / "missing.mp4", CFG))
    assert [o.frame_index for o in obs] == [5, 20]
    assert obs[0].players[0].track_id == 1

    alien = tmp_path / "alien"
    alien.mkdir()
    (alien / "r.jsonl").write_text(json.dumps({"foo": 1, "bar": 2}) + "\n")
    with pytest.raises(ValueError, match="unrecognized batch record shape"):
        list(BatchResultsClient(alien).run(tmp_path / "missing.mp4", CFG))


def test_client_requires_results_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(BatchResultsClient(tmp_path).run(tmp_path / "v.mp4", CFG))
