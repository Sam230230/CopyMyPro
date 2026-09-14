"""
CopyMyPro — Phase 1 feasibility spike.

Question this answers: can pose estimation track a tennis swing well enough
that comparing two swings would be meaningful?

Usage:
    .venv/bin/python spike.py samples/my_swing.mov
    python spike.py samples/my_swing.mov --lefty      # left-handed player
    python spike.py samples/my_swing.mov --no-racket  # body skeleton only

Writes an annotated video to output/ with the skeleton drawn on every frame,
and prints a tracking-quality report to the terminal.

This is throwaway code. Do not optimize it.
"""

import sys
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# MediaPipe 0.10.3x dropped the old `mp.solutions.pose` API, so this uses the
# Tasks API instead. Same 33-landmark model underneath — the .task file is the
# "heavy" (most accurate) variant, which is what model_complexity=2 used to mean.
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "pose_landmarker_heavy.task")

# COCO's 80 classes include "tennis racket", so an off-the-shelf detector finds
# the racket without any training. Optional: if the file is missing the racket
# falls back to being inferred from the hand. Download with:
#   curl -sL -o efficientdet_lite2.tflite https://storage.googleapis.com/\
# mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite
DETECTOR_PATH = os.path.join(HERE, "efficientdet_lite2.tflite")
RACKET_CLASS = "tennis racket"
RACKET_SCORE_THRESHOLD = 0.3

# The joints that matter most for a tennis forehand. If these track badly,
# the whole comparison idea is in trouble — everything else is secondary.
KEY_JOINTS = {
    "right_shoulder": vision.PoseLandmark.RIGHT_SHOULDER.value,
    "right_elbow": vision.PoseLandmark.RIGHT_ELBOW.value,
    "right_wrist": vision.PoseLandmark.RIGHT_WRIST.value,
    "left_shoulder": vision.PoseLandmark.LEFT_SHOULDER.value,
    "right_hip": vision.PoseLandmark.RIGHT_HIP.value,
    "right_knee": vision.PoseLandmark.RIGHT_KNEE.value,
}

# MediaPipe reports a visibility score 0-1 per landmark. Below this we treat
# the joint as "not really tracked" rather than trusting the coordinates.
VISIBILITY_THRESHOLD = 0.5

# --- Racket overlay ---------------------------------------------------------
# The racket shaft direction comes from the COCO detector when it fires, and
# falls back to the hand direction (wrist -> index knuckle) when it doesn't.
# The fallback is a guess and drifts 30-45 degrees off — it's there so the
# overlay doesn't blink out, not because it's trustworthy. Length is always
# torso-derived: a detection box is axis-aligned, so it pins down where the
# racket is and which way it points, but not how long it is.
RACKET_COLOR = (0, 255, 255)          # yellow (BGR), detector-driven
RACKET_GUESS_COLOR = (0, 165, 255)    # orange, hand-inferred fallback
BOX_COLOR = (90, 90, 60)              # dim — the raw detection box
RACKET_THICKNESS = 2

# Proportions measured off racket_skeleton.png, as fractions of total length.
HANDLE_FRAC = 0.42                    # butt -> throat
HEAD_FRAC = 0.58                      # throat -> tip
HEAD_WIDTH_FRAC = 0.78                # head width as a fraction of head length
GRIP_OFFSET_FRAC = 0.08               # butt sits slightly behind the wrist

# A real racket is ~68cm; an adult shoulder-to-hip is ~50cm. Scaling off the
# torso keeps the racket proportionate as the player moves toward the camera.
RACKET_TO_TORSO = 1.35


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def make_direction_smoother(alpha=0.6):
    """Exponential smoothing on the shaft direction vector.

    alpha is the weight on the current frame — high enough that a fast swing
    still tracks, low enough to take the buzz off the detection box.
    """
    prev = {}

    def smooth(dx, dy):
        if "v" in prev:
            ox, oy = prev["v"]
            dx, dy = alpha * dx + (1 - alpha) * ox, alpha * dy + (1 - alpha) * oy
            n = (dx * dx + dy * dy) ** 0.5
            if n > 1e-6:
                dx, dy = dx / n, dy / n
        prev["v"] = (dx, dy)
        return dx, dy

    return smooth


def pick_racket_box(detections, wrist, max_dist):
    """Closest 'tennis racket' box whose centre is plausibly in the hand's
    reach. Rejects rackets lying on the court or held by someone else."""
    best, best_d = None, max_dist
    for det in detections:
        cat = det.categories[0]
        if cat.category_name != RACKET_CLASS or cat.score < RACKET_SCORE_THRESHOLD:
            continue
        b = det.bounding_box
        cx, cy = b.origin_x + b.width / 2.0, b.origin_y + b.height / 2.0
        d = ((cx - wrist[0]) ** 2 + (cy - wrist[1]) ** 2) ** 0.5
        if d < best_d:
            best, best_d = (b, (cx, cy)), d
    return best


def draw_racket(frame, landmarks, pts, lefty=False, detections=(), smooth=None):
    """Draw the racket_skeleton.png shape on the racket hand.

    Returns "detected" if the shaft direction came from the object detector,
    "guessed" if it fell back to the hand landmarks, or None if the hand
    wasn't tracked well enough to place a racket at all.
    """
    L = vision.PoseLandmark
    if lefty:
        wrist_i, elbow_i, index_i = L.LEFT_WRIST, L.LEFT_ELBOW, L.LEFT_INDEX
        shoulder_i = L.LEFT_SHOULDER
    else:
        wrist_i, elbow_i, index_i = L.RIGHT_WRIST, L.RIGHT_ELBOW, L.RIGHT_INDEX
        shoulder_i = L.RIGHT_SHOULDER

    wrist = pts[wrist_i.value]
    wrist_ok = landmarks[wrist_i.value].visibility >= VISIBILITY_THRESHOLD

    # Anchor for working out which way the racket points: the arm joint nearest
    # the hand that's actually tracked. On a fast swing the wrist blurs out far
    # more often than the shoulder, and the racket points away from the body
    # either way, so the shoulder still gives a usable direction.
    anchor = None
    for idx in (wrist_i, elbow_i, shoulder_i):
        if landmarks[idx.value].visibility >= VISIBILITY_THRESHOLD:
            anchor = pts[idx.value]
            break
    if anchor is None:
        return None

    # Scale off the torso. Falls back to a fixed guess if hips aren't visible.
    shoulder_mid = _mid(pts[L.LEFT_SHOULDER.value], pts[L.RIGHT_SHOULDER.value])
    hip_mid = _mid(pts[L.LEFT_HIP.value], pts[L.RIGHT_HIP.value])
    torso = ((shoulder_mid[0] - hip_mid[0]) ** 2
             + (shoulder_mid[1] - hip_mid[1]) ** 2) ** 0.5
    if torso < 20:
        torso = frame.shape[0] * 0.22
    length = torso * RACKET_TO_TORSO

    # Shaft direction. The detector's box is axis-aligned so it carries no
    # angle of its own, but the racket runs from the hand to the far end of
    # its own box — so wrist -> box centre recovers the direction. Using the
    # centre rather than the far corner keeps it stable when the box is thin.
    source = None
    centre = None
    box = pick_racket_box(detections, anchor, max_dist=2.0 * length)
    if box is not None:
        b, centre = box
        cv2.rectangle(frame, (b.origin_x, b.origin_y),
                      (b.origin_x + b.width, b.origin_y + b.height),
                      BOX_COLOR, 1)
        dx, dy = centre[0] - anchor[0], centre[1] - anchor[1]
        if (dx * dx + dy * dy) ** 0.5 > 0.1 * length:
            source = "detected"

    if source is None:
        # No usable detection: point the racket along the hand, or failing
        # that straight out from the forearm. Both are guesses, and both need
        # a real wrist to hang off.
        if not wrist_ok:
            return None
        if landmarks[index_i.value].visibility >= VISIBILITY_THRESHOLD:
            tgt = pts[index_i.value]
            dx, dy = tgt[0] - wrist[0], tgt[1] - wrist[1]
        elif landmarks[elbow_i.value].visibility >= VISIBILITY_THRESHOLD:
            elbow = pts[elbow_i.value]
            dx, dy = wrist[0] - elbow[0], wrist[1] - elbow[1]
        else:
            return None
        source = "guessed"

    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-3:
        return None
    dx, dy = dx / norm, dy / norm

    # Per-frame detection boxes jitter by a few degrees; smooth the direction
    # so the racket doesn't twitch. Light enough not to lag a real swing.
    # Only detected frames go through the filter — feeding a hand-guessed
    # direction into it would drag the next real detection off with it.
    if smooth is not None and source == "detected":
        dx, dy = smooth(dx, dy)

    px, py = -dy, dx  # perpendicular, in the image plane
    colour = RACKET_COLOR if source == "detected" else RACKET_GUESS_COLOR

    # Where the butt sits. With a detection, work back from the box centre —
    # that's a measurement of where the racket actually is, and it doesn't
    # need the wrist, which is the landmark most likely to be blurred out.
    # Without one, hang it off the wrist as before.
    if source == "detected":
        butt = (centre[0] - dx * 0.5 * length, centre[1] - dy * 0.5 * length)
    else:
        butt = (wrist[0] - dx * GRIP_OFFSET_FRAC * length,
                wrist[1] - dy * GRIP_OFFSET_FRAC * length)

    def along(d, side=0.0):
        """Point d px up the shaft from the butt, side px off the axis."""
        return (int(butt[0] + dx * d + px * side),
                int(butt[1] + dy * d + py * side))

    throat = along(HANDLE_FRAC * length)
    tip = along(length)
    half_w = HEAD_WIDTH_FRAC * HEAD_FRAC * length / 2.0
    waist = (HANDLE_FRAC + HEAD_FRAC / 2.0) * length  # widest point of the head
    left = along(waist, -half_w)
    right = along(waist, +half_w)

    cv2.line(frame, (int(butt[0]), int(butt[1])), throat,
             colour, RACKET_THICKNESS)                # handle
    for p, q in ((throat, left), (left, tip), (tip, right), (right, throat)):
        cv2.line(frame, p, q, colour, RACKET_THICKNESS)  # diamond head
    cv2.line(frame, throat, tip, colour, 1)           # strings, long axis
    cv2.line(frame, left, right, colour, 1)           # strings, short axis
    return source


def draw_pose(frame, landmarks, width, height):
    """Draw the skeleton by hand — the Tasks API has no drawing_utils."""
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (255, 255, 255), 2)
    for i, pt in enumerate(pts):
        # Green = trusted, orange = below the visibility threshold.
        color = ((0, 255, 0) if landmarks[i].visibility >= VISIBILITY_THRESHOLD
                 else (0, 165, 255))
        cv2.circle(frame, pt, 3, color, -1)
    return pts


def analyze(video_path, lefty=False, racket=True):
    if not os.path.exists(video_path):
        sys.exit(f"File not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"OpenCV could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nInput:  {video_path}")
    print(f"Size:   {width}x{height} @ {fps:.1f} fps, {total} frames "
          f"({total / fps:.1f}s)\n")

    os.makedirs("output", exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join("output", f"{stem}_pose.mp4")

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frames = 0
    frames_with_pose = 0
    racket_counts = {"detected": 0, "guessed": 0}
    joint_hits = {name: 0 for name in KEY_JOINTS}

    use_detector = racket and os.path.exists(DETECTOR_PATH)
    if racket and not use_detector:
        print(f"No {os.path.basename(DETECTOR_PATH)} — the racket will be")
        print("inferred from the hand instead of detected. See DETECTOR_PATH.\n")
    smooth = make_direction_smoother() if racket else None

    # The "heavy" model is the most accurate (and slowest) variant. For a
    # fast swing we want accuracy over speed — this runs offline, not live.
    # VIDEO running mode lets it track between frames instead of re-detecting.
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Same 80-class COCO detector everyone uses; "tennis racket" is class 43.
    # Nothing tennis-specific was trained here — it just happens to be in COCO.
    det_options = vision.ObjectDetectorOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=DETECTOR_PATH),
        running_mode=vision.RunningMode.VIDEO,
        score_threshold=RACKET_SCORE_THRESHOLD,
        max_results=10,
    )

    with vision.PoseLandmarker.create_from_options(options) as pose:
        detector = (vision.ObjectDetector.create_from_options(det_options)
                    if use_detector else None)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            # MediaPipe wants RGB; OpenCV gives BGR.
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            # VIDEO mode needs a monotonically increasing timestamp in ms.
            ts = int(frames / fps * 1000)
            result = pose.detect_for_video(image, ts)
            detections = (detector.detect_for_video(image, ts).detections
                          if detector else ())

            if result.pose_landmarks:
                frames_with_pose += 1
                lms = result.pose_landmarks[0]
                for name, idx in KEY_JOINTS.items():
                    if lms[idx].visibility >= VISIBILITY_THRESHOLD:
                        joint_hits[name] += 1

                pts = draw_pose(frame, lms, width, height)
                if racket:
                    src = draw_racket(frame, lms, pts, lefty, detections, smooth)
                    if src:
                        racket_counts[src] += 1
            else:
                # Loud red banner so dropped frames are obvious on playback.
                cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 255), -1)
                cv2.putText(frame, "NO POSE DETECTED", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.putText(frame, f"{frames}", (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            writer.write(frame)

            if frames % 30 == 0:
                print(f"  ...{frames} frames", flush=True)

        if detector:
            detector.close()

    cap.release()
    writer.release()
    report(frames, frames_with_pose, joint_hits, out_path,
           racket_counts if racket else None)


def report(frames, frames_with_pose, joint_hits, out_path, racket_counts):
    if frames == 0:
        sys.exit("No frames were read. Is the file a valid video?")

    body_pct = 100.0 * frames_with_pose / frames

    print("\n" + "=" * 52)
    print("TRACKING REPORT")
    print("=" * 52)
    print(f"Frames processed:        {frames}")
    print(f"Frames with a skeleton:  {frames_with_pose} ({body_pct:.1f}%)")
    if racket_counts is not None:
        det, guess = racket_counts["detected"], racket_counts["guessed"]
        print(f"Racket from detection:   {det} "
              f"({100.0 * det / frames:.1f}%)   [yellow]")
        print(f"Racket guessed from hand:{guess:>5} "
              f"({100.0 * guess / frames:.1f}%)   [orange]")
    print("\nPer-joint tracking (% of frames above visibility 0.5):")
    for name, hits in joint_hits.items():
        pct = 100.0 * hits / frames
        flag = "ok" if pct >= 80 else ("weak" if pct >= 50 else "BAD")
        print(f"  {name:<16} {pct:5.1f}%   {flag}")

    arm = min(
        joint_hits["right_shoulder"],
        joint_hits["right_elbow"],
        joint_hits["right_wrist"],
    ) / frames * 100.0

    print("\n" + "-" * 52)
    if body_pct >= 90 and arm >= 80:
        print("VERDICT: green light. Tracking is solid enough to compare swings.")
    elif body_pct >= 70 and arm >= 60:
        print("VERDICT: borderline. Usable, but try a tripod / side-on angle")
        print("         / better lighting and re-run before deciding.")
    else:
        print("VERDICT: poor tracking. Before giving up, try: a tripod, filming")
        print("         side-on from ~5m, brighter light, and plain background.")
    print("-" * 52)
    print("\nNow WATCH the annotated video — the numbers can look fine while")
    print("the skeleton is visibly wrong at contact. Trust your eyes:")
    print(f"  open {out_path}\n")

    if racket_counts is not None:
        print("Racket overlay: YELLOW = shaft direction from the COCO detector,")
        print("ORANGE = detector missed, direction guessed from the hand (treat")
        print("those frames as decoration). The dim box is the raw detection.")
        print("Neither gives racket FACE angle — a box has no roll.")
        print("If the racket hangs off the wrong hand, use --lefty.\n")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        sys.exit("Usage: python spike.py samples/<video>.mov [--lefty] "
                 "[--no-racket]")
    analyze(args[0], lefty="--lefty" in flags, racket="--no-racket" not in flags)
