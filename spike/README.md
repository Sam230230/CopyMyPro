# CopyMyPro — Phase 1 Feasibility Spike

**Goal:** validate that pose estimation works well enough on amateur tennis-swing video to make the CopyMyPro concept viable.

- If it works → proceed with the Swift + Apple Vision MVP plan.
- If it doesn't → rethink the concept before investing weeks of build time.

This is **throwaway code**. It exists only to answer the question above. Do not optimize for elegance, tests, or architecture.

## Setup (one-time)

```bash
# needs Python 3.11 (mediapipe has no 3.13+ wheels yet) — brew install python@3.11
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

In VSCode: `Cmd+Shift+P` → "Python: Select Interpreter" → pick `spike/.venv`.

Activating (`source .venv/bin/activate`) is optional — running `.venv/bin/python`
directly works without it.

### Models

Two model files live in this directory and are **not** checked in — download both:

```bash
# Pose: 33 body landmarks (heavy = most accurate variant)
curl -sL -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

# Objects: COCO's 80 classes, one of which is "tennis racket" (optional —
# without it the racket overlay falls back to guessing from the hand)
curl -sL -O https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite
```

## Run

Put a tennis-swing video in `samples/` (mp4 or mov), then:

```bash
.venv/bin/python spike.py samples/<your_video>.mp4
.venv/bin/python spike.py samples/<your_video>.mp4 --lefty      # left-handed player
.venv/bin/python spike.py samples/<your_video>.mp4 --no-racket  # body skeleton only
```

### As an MCP tool

`spike_mcp.py` exposes the same analysis as one MCP tool (`analyze_swing`) so
Claude can run it on a clip and read the report back:

```bash
claude mcp add copymypro-spike -- "$(pwd)/.venv/bin/python" "$(pwd)/spike_mcp.py"
```

The annotated video is written to `output/` with the pose skeleton drawn over each frame.

### Reading the racket overlay

The racket is drawn as a diamond head + string cross + handle. Its colour tells
you where the shaft angle came from, and that distinction matters:

- **Yellow** — the COCO detector found the racket this frame. Position and shaft direction are real measurements.
- **Orange** — the detector missed. Direction is extrapolated from the hand landmarks and drifts 30–45°. Decoration, not data.
- **Dim olive box** — the raw detection bounding box, for debugging.

Neither mode recovers racket **face** angle (roll). An axis-aligned box has no
orientation, so open-vs-closed face is invisible to this spike. If you need to
score face angle, that's a separate problem — an oriented-box or keypoint model.

## Sample footage to test on

Aim for two contrasting clips:

1. **Pro slow-mo** — search YouTube for "tennis forehand slow motion" and screen-record or download a short clip. This is the easy case; pose estimation should crush it.
2. **Amateur phone clip** — film yourself (or someone) hitting a forehand on your iPhone. This is the realistic case; if pose breaks here, the whole product premise needs rethinking.

## What to look for

- Does the skeleton track the whole swing, or does it lose the player at impact?
- Is the racket arm tracked accurately during the fast contact phase?
- Does it handle the player turning sideways (not facing camera)?
- How does quality differ between pro slow-mo and amateur full-speed footage?
- How does it cope with shaky handheld vs tripod-mounted video?

Capture your observations in `findings.md` as you go.

## Decision criteria

Spike succeeds if amateur phone footage produces a clean, mostly-correct skeleton through the swing — good enough that a comparison to a reference pose would be meaningful.

Spike fails if pose is wildly noisy, drops frames, or completely loses the arm during the swing. In that case we look at alternatives (Apple Vision's built-in pose detector, more constrained camera setups, or a different product framing).
