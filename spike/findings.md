# Spike findings

Capture observations here as you run the spike on different clips. Be specific — these notes feed directly into the MVP plan decisions.

## Clip 1: <name / description>

- Source:
- Length:
- Resolution / framerate:
- Camera setup (handheld / tripod / phone propped up):
- Lighting:

**Observations:**

- Pose tracking through the swing:
- Racket arm at contact:
- Player orientation (side-on / front-on):
- Notable failures:

## Clip 2: <name / description>

(repeat the same template)

---

## Racket detection (added 2026-08-12)

The racket overlay uses COCO's `tennis racket` class via EfficientDet-Lite2.
Nothing tennis-specific was trained; it's off-the-shelf.

Measured on `samples/Pro/Wawrinka_backhand.mov` (76 frames):

| signal | frames | rate |
| --- | --- | --- |
| racket detected at score ≥ 0.30 | 26 / 76 | 34% |
| racket detected at score ≥ 0.15 | 29 / 76 | 38% |
| right wrist visible ≥ 0.5 | 47 / 76 | 62% |
| **both in the same frame** | **9 / 76** | **12%** |

**The key finding: those two signals are anti-correlated.** If they were
independent you'd expect ~16 overlapping frames; there are 9. The racket is
most detectable when it's extended away from the body at speed — which is
exactly when the wrist motion-blurs and pose drops it. The frames where the
racket is easy are the frames where the arm is hard.

Consequences already acted on:
- Lowering the detection threshold does **not** help (34% → 38% for 2× the false positives). The detector is missing the racket, not scoring it low.
- The proximity filter is not the bottleneck — it rejected 0 of 9 candidates. Box-centre-to-wrist distances all landed at 0.23–0.52 racket-lengths, comfortably inside the cutoff.
- So the racket is now anchored to the **detection box** rather than the wrist, and direction is taken from the nearest tracked arm joint (wrist → elbow → shoulder). The shoulder tracks at 98.7%, so a detection no longer goes to waste when the wrist blurs.

Still unsolved: **racket face angle (roll) is not recoverable this way.** An
axis-aligned box has no orientation. Open-vs-closed face — arguably the most
coachable thing about a stroke — is invisible to this pipeline. Getting it
needs an oriented-box or racket-keypoint model, i.e. custom training data.
Worth deciding whether v1 scores racket path at all, or sticks to body pose.

## Overall verdict

- [ ] Spike succeeded — proceed with Swift + Apple Vision MVP
- [ ] Spike failed — rethink

**Decision rationale:**
