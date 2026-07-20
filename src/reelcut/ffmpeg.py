"""ffmpeg/video utilities.

No system ffmpeg is assumed: resolve the binary from PATH first, else fall
back to the static binary bundled with imageio-ffmpeg. Probing uses cv2
(bundled with inference) — imageio-ffmpeg ships no ffprobe.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary (PATH lookup, else imageio_ffmpeg)."""
    raise NotImplementedError


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(video: Path) -> VideoMeta:
    """Read fps/frames/size via cv2.VideoCapture. Raises FileNotFoundError /
    ValueError on unreadable input."""
    raise NotImplementedError


def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg with ``args`` (no leading binary), raising CalledProcessError
    with captured stderr tail on failure. Always pass -y and -loglevel error."""
    raise NotImplementedError
