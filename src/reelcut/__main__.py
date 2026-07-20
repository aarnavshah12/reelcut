"""CLI: python -m reelcut --video game.mp4 --jersey 10 --team-color blue \
--target-frame 1500 --target-box x,y,w,h --out ./output/

Flags:
  --video PATH (required)      --jersey STR (required)
  --team-color STR (required)  --target-frame INT (required)
  --target-box X,Y,W,H (required, pixels)
  --out DIR (default ./output)
  --debug-video                render annotated debug.mp4
  --force-stage N              recompute stage N and later from cache
  --stub                       use StubWorkflowClient (no Roboflow needed)
  --workflow ID                workspace workflow slug (default from config)
  --model-id ID                override detection model (workflow parameter)
  --api-url URL                inference server URL (default in-process)
  --fps FLOAT                  sampling fps (default 5.0)
  --seed INT                   determinism seed
  --sport NAME                 sport preset (default soccer)

Reads ROBOFLOW_API_KEY from env / .env (python-dotenv). Exits 2 on bad args,
1 with a clear message when the seed click matches no tracklet.
"""
from __future__ import annotations

import sys


def parse_box(s: str):
    """'x,y,w,h' -> BBox; raises argparse-friendly ValueError."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
