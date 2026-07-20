"""Stage-output caching.

Each pipeline stage persists its output under
``<out>/cache/<video_key>/stageN_<name>.json.gz`` where ``video_key`` is a
sha256 over (first 8 MiB of the video, file size, sample fps, workflow ref).
Downstream stages re-run without re-inference; ``--force-stage N`` recomputes
stage N and everything after it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from .types import from_jsonable, to_jsonable


def video_cache_key(video: Path, sample_fps: float, workflow_ref: str) -> str:
    """Hex digest identifying (video content, sampling, workflow version).

    Reads only the first 8 MiB of the file plus its size, so hashing a 90-min
    4K video stays instant. Must be deterministic across runs.
    """
    raise NotImplementedError


class StageCache:
    """Load/save JSON-able payloads per (video_key, stage name)."""

    def __init__(self, cache_dir: Path, video_key: str) -> None:
        self.dir = cache_dir / video_key
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, stage_index: int, name: str) -> Path:
        return self.dir / f"stage{stage_index}_{name}.json.gz"

    def has(self, stage_index: int, name: str) -> bool:
        raise NotImplementedError

    def save(self, stage_index: int, name: str, payload: Any) -> Path:
        """``payload`` is any structure accepted by ``types.to_jsonable``."""
        raise NotImplementedError

    def load(self, stage_index: int, name: str) -> Any:
        """Inverse of :meth:`save` (via ``types.from_jsonable``)."""
        raise NotImplementedError

    def invalidate_from(self, stage_index: int) -> None:
        """Delete cached outputs for ``stage_index`` and every later stage."""
        raise NotImplementedError
