# CopyMyPro

An experimental tennis-motion analysis app combining an iOS video/pose interface
with a Python computer-vision feasibility study. Personal project begun in May 2026.

## Demo video


<img width="800" height="449" alt="ezgif-4286b9911272c09c" src="https://github.com/user-attachments/assets/f42a4c15-146b-4b2a-a8ef-3d7cc895287e" />





## Explore the project

- **iOS:** open `CopyMyPro/CopyMyPro.xcodeproj` in Xcode. The Swift sources analyze
  pose, render annotated video and provide side-by-side playback.
- **Python:** start with [the spike setup guide](spike/README.md). It describes
  Python dependencies, MediaPipe model downloads and command-line video analysis.
- **Evidence:** [spike/findings.md](spike/findings.md) records the feasibility observations.

## Status and limits

This is a prototype, not a validated coaching or injury-prevention product.
One recorded 76-frame clip had 26 racket detections, 47 wrist-visible frames and
only 9 frames with both. Those observations are not a multi-video benchmark.
The axis-aligned detector cannot measure racket-face angle; hand-derived racket
overlays are estimates. Generalized similarity and coaching quality remain unvalidated.

Model weights and sample/pro footage are excluded. Download models as described
in the spike guide and supply footage you have permission to use. The optional
iOS reference video can be added as `pro_serve.mov` or `.mp4`; the app tolerates
its absence. No personal footage is included.

## Validation

See [VALIDATION.md](VALIDATION.md) for checks performed on this published snapshot.
