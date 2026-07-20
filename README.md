# reelcut

Turn a full-length phone recording of a kids' soccer game into a highlight reel
of every play one specific kid was involved in. Built Roboflow-first: detection,
tracking, class-locking, box smoothing, velocity, and jersey OCR all run inside a
[Roboflow Workflow](workflows/reelcut_tracking.json); the local app only
orchestrates, stitches identity, scores involvement, and cuts clips.

```
video → [InferencePipeline × Roboflow Workflow] → per-frame observations
      → identity stitching → involvement scoring → ClipCutter → highlights.mp4
```

## Setup

```bash
uv sync                                  # Python 3.12 env, installs `inference`
echo 'ROBOFLOW_API_KEY=...' > .env       # gitignored
```

ffmpeg is not required on the system — the binary bundled with `imageio-ffmpeg`
is used automatically (probing goes through OpenCV).

## Usage

```bash
uv run python -m reelcut \
  --video game.mp4 --jersey 10 --team-color blue \
  --target-frame 1500 --target-box 640,320,40,90 \
  --out ./output/ [--debug-video] [--force-stage N]
```

- `--stub` runs a deterministic synthetic game instead of Roboflow — every
  downstream stage works offline (Phase 0 acceptance mode).
- Stage outputs cache under `out/cache/<video-key>/`; `--force-stage N`
  recomputes stage N and everything after. Stages: 1 infer, 2 identity,
  3 score, 4 cut.
- `--model-id` / `--workflow` override the detection model / workflow slug
  (defaults in [config.py](src/reelcut/config.py)).

## The workflow

[workflows/reelcut_tracking.json](workflows/reelcut_tracking.json) is the source
of truth, mirrored in the `aarnavs-space` workspace as `reelcut-tracking`
(update path: edit JSON → `workflow_specs_validate` → `workflows_update`).

detector (model_id param) → per-class confidence filter → split people/ball
(ball: highest-confidence only) → **BoT-SORT with camera-motion compensation** →
track class lock → detections stabilizer → velocity → dynamic crop → DocTR OCR →
stitch OCR. Outputs per frame: `tracked_players` (with `smoothed_speed`/
`smoothed_velocity` in data), `ball`, `jersey_texts` (aligned with player
order), `raw_detections`.

The detection model is a workflow parameter: currently a public Universe
stand-in (`my-first-project-gsrpg/18`) until our RF-DETR-small finishes
training; swap = one `--model-id` flag / one default change.

## GPU offload (Roboflow Batch Processing)

Local in-process inference is ~20x slower than realtime on CPU (OCR-heavy).
For full games, run stage 1 on a Roboflow cloud GPU instead:

```bash
uv run python scripts/run_batch.py --video game.mp4 --jersey 10 \
    --team-color blue --target-frame 1500 --target-box x,y,w,h --dry-run
```

Drop `--dry-run` to actually submit (costs Roboflow credits per job). The
script stages the video, runs `reelcut-tracking` on GPU via Batch Processing,
exports JSONL results, and finishes identity/scoring/cutting locally in
seconds. `python -m reelcut --batch-results DIR` re-runs downstream stages
against already-exported results. This is the same API surface a future web
backend would call from an upload handler (webhook support included), so the
CLI-vs-web-app question is just about who calls these functions.

## Tests

```bash
uv run pytest        # unit + stub end-to-end, no network, no video needed
```

## Known platform notes

- **Apple Silicon + torch 2.13**: `inference-models` crashes with
  `torch.mps has no attribute 'current_device'`; `workflow_client._shim_torch_mps()`
  patches it before importing `inference` (bug reported to Roboflow).
- **Serverless can't run this workflow**: the tracker/velocity blocks are
  stateful video blocks; run via `InferencePipeline` locally (default) or a
  local `inference server start` — not per-request serverless.
- OCR runs on every player crop every sampled frame and dominates CPU time;
  `scripts/compare_trackers.py` strips the OCR branch when benchmarking
  trackers.

## Fixture

`data/fixtures/game_2min.mp4` — real spectator-filmed match footage,
CC BY-SA 4.0, attribution + provenance in
[data/fixtures/SOURCE.md](data/fixtures/SOURCE.md) (file is gitignored; the
attribution doc is not).
