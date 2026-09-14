//
//  PoseAnalyzer.swift
//  CopyMyPro
//
//  Runs Apple Vision body-pose detection over every frame of a video.
//  This is the in-app version of the Python feasibility spike — if pose
//  tracks well here, the spike isn't needed.
//

import AVFoundation
import CoreMedia
import ImageIO
import Vision

struct PoseFrame: Sendable {
    let time: CMTime
    /// Joint locations in Vision's normalized space: [0, 1] with origin at bottom-left.
    let joints: [VNHumanBodyPoseObservation.JointName: CGPoint]
}

enum PoseAnalyzerError: Error {
    case noVideoTrack
    case readerSetupFailed
}

actor PoseAnalyzer {
    /// All joint names we care about for a tennis-swing skeleton.
    private static let trackedJoints: [VNHumanBodyPoseObservation.JointName] = [
        .nose, .neck, .root,
        .leftShoulder, .rightShoulder,
        .leftElbow, .rightElbow,
        .leftWrist, .rightWrist,
        .leftHip, .rightHip,
        .leftKnee, .rightKnee,
        .leftAnkle, .rightAnkle,
    ]

    private static let minConfidence: Float = 0.3

    func analyze(
        videoURL: URL,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> [PoseFrame] {
        let asset = AVURLAsset(url: videoURL)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { throw PoseAnalyzerError.noVideoTrack }

        let duration = try await asset.load(.duration)
        let preferredTransform = try await track.load(.preferredTransform)
        let orientation = imageOrientation(for: preferredTransform)

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        )
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw PoseAnalyzerError.readerSetupFailed }
        reader.add(output)
        guard reader.startReading() else {
            throw reader.error ?? PoseAnalyzerError.readerSetupFailed
        }

        let totalSeconds = duration.seconds.isFinite ? duration.seconds : 0
        let request = VNDetectHumanBodyPoseRequest()
        var frames: [PoseFrame] = []

        while let sample = output.copyNextSampleBuffer() {
            defer { CMSampleBufferInvalidate(sample) }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            let time = CMSampleBufferGetPresentationTimeStamp(sample)

            let handler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: orientation,
                options: [:]
            )
            var joints: [VNHumanBodyPoseObservation.JointName: CGPoint] = [:]
            do {
                try handler.perform([request])
                if let observation = request.results?.first {
                    for name in Self.trackedJoints {
                        if let point = try? observation.recognizedPoint(name),
                           point.confidence > Self.minConfidence {
                            joints[name] = point.location
                        }
                    }
                }
            } catch {
                // Skip this frame; pose detection occasionally fails on noisy frames.
            }
            frames.append(PoseFrame(time: time, joints: joints))

            if totalSeconds > 0 {
                progress(min(1.0, time.seconds / totalSeconds))
            }
        }

        if reader.status == .failed {
            throw reader.error ?? PoseAnalyzerError.readerSetupFailed
        }
        progress(1.0)
        return frames
    }

    /// Maps an AVFoundation `preferredTransform` to the orientation Vision needs
    /// so the returned points line up with what AVPlayerLayer displays.
    private func imageOrientation(for transform: CGAffineTransform) -> CGImagePropertyOrientation {
        let angleDegrees = Int(round(atan2(transform.b, transform.a) * 180 / .pi))
        switch ((angleDegrees % 360) + 360) % 360 {
        case 90:  return .right
        case 180: return .down
        case 270: return .left
        default:  return .up
        }
    }
}
