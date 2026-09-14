//
//  PoseVideoRenderer.swift
//  CopyMyPro
//
//  Bakes a pose skeleton + synthesized racket into a new MP4 file using
//  AVAssetExportSession + AVVideoCompositionCoreAnimationTool.
//

import AVFoundation
import CoreMedia
import QuartzCore
import UIKit
import Vision

enum PoseVideoRendererError: Error {
    case noVideoTrack
    case exporterUnavailable
    case exportFailed(Error?)
}

actor PoseVideoRenderer {
    func render(
        inputURL: URL,
        frames: [PoseFrame],
        outputURL: URL
    ) async throws -> URL {
        let asset = AVURLAsset(url: inputURL)
        let duration = try await asset.load(.duration)
        let durationSec = duration.seconds
        guard durationSec.isFinite, durationSec > 0 else { throw PoseVideoRendererError.noVideoTrack }

        let videoComposition = try await AVMutableVideoComposition.videoComposition(withPropertiesOf: asset)
        let renderSize = videoComposition.renderSize

        // Layer hierarchy: parent contains the video layer (filled by AVF) and our shape overlays.
        let videoLayer = CALayer()
        videoLayer.frame = CGRect(origin: .zero, size: renderSize)
        let parentLayer = CALayer()
        parentLayer.frame = CGRect(origin: .zero, size: renderSize)
        // Match UIKit's top-left origin so our coordinate math is intuitive.
        parentLayer.isGeometryFlipped = true
        parentLayer.addSublayer(videoLayer)

        let skeletonLayer = CAShapeLayer()
        skeletonLayer.frame = parentLayer.bounds
        skeletonLayer.strokeColor = UIColor.systemGreen.cgColor
        skeletonLayer.fillColor = UIColor.systemGreen.withAlphaComponent(0.9).cgColor
        skeletonLayer.lineWidth = max(3, renderSize.width / 240)
        skeletonLayer.lineCap = .round
        skeletonLayer.lineJoin = .round
        parentLayer.addSublayer(skeletonLayer)

        let racketLayer = CAShapeLayer()
        racketLayer.frame = parentLayer.bounds
        racketLayer.strokeColor = UIColor.systemYellow.cgColor
        racketLayer.fillColor = UIColor.systemYellow.withAlphaComponent(0.35).cgColor
        racketLayer.lineWidth = max(4, renderSize.width / 180)
        racketLayer.lineCap = .round
        racketLayer.lineJoin = .round
        parentLayer.addSublayer(racketLayer)

        // Map Vision (bottom-left origin, normalized) → pixel coords (top-left origin) in render size.
        let mapping: (CGPoint) -> CGPoint = { p in
            CGPoint(x: p.x * renderSize.width, y: (1 - p.y) * renderSize.height)
        }

        var skeletonValues: [CGPath] = []
        var racketValues: [CGPath] = []
        var keyTimes: [NSNumber] = []
        let emptyPath = CGPath(rect: .zero, transform: nil)
        for frame in frames {
            skeletonValues.append(skeletonPath(frame: frame, mapping: mapping))
            racketValues.append(racketPath(frame: frame, mapping: mapping) ?? emptyPath)
            let t = max(0, min(1, frame.time.seconds / durationSec))
            keyTimes.append(NSNumber(value: t))
        }

        func attachKeyframeAnimation(to layer: CAShapeLayer, values: [CGPath]) {
            let anim = CAKeyframeAnimation(keyPath: "path")
            anim.values = values
            anim.keyTimes = keyTimes
            anim.duration = durationSec
            anim.beginTime = AVCoreAnimationBeginTimeAtZero
            anim.fillMode = .both
            anim.isRemovedOnCompletion = false
            anim.calculationMode = .discrete
            layer.add(anim, forKey: "path")
        }
        attachKeyframeAnimation(to: skeletonLayer, values: skeletonValues)
        attachKeyframeAnimation(to: racketLayer, values: racketValues)

        videoComposition.animationTool = AVVideoCompositionCoreAnimationTool(
            postProcessingAsVideoLayer: videoLayer,
            in: parentLayer
        )

        try? FileManager.default.removeItem(at: outputURL)
        guard let exporter = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetHighestQuality) else {
            throw PoseVideoRendererError.exporterUnavailable
        }
        exporter.videoComposition = videoComposition
        exporter.shouldOptimizeForNetworkUse = true

        do {
            try await exporter.export(to: outputURL, as: .mp4)
        } catch {
            throw PoseVideoRendererError.exportFailed(error)
        }
        return outputURL
    }

    // MARK: - Path builders (mirrors PoseOverlayPlayerView, but in pixel space)

    private static let connections: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
        (.neck, .nose),
        (.leftShoulder, .rightShoulder),
        (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
        (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
        (.leftShoulder, .leftHip), (.rightShoulder, .rightHip),
        (.leftHip, .rightHip),
        (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
        (.rightHip, .rightKnee), (.rightKnee, .rightAnkle),
    ]

    private func skeletonPath(frame: PoseFrame, mapping: (CGPoint) -> CGPoint) -> CGPath {
        let path = CGMutablePath()
        for (a, b) in Self.connections {
            guard let pa = frame.joints[a], let pb = frame.joints[b] else { continue }
            path.move(to: mapping(pa))
            path.addLine(to: mapping(pb))
        }
        for (_, p) in frame.joints {
            let pt = mapping(p)
            path.addEllipse(in: CGRect(x: pt.x - 4, y: pt.y - 4, width: 8, height: 8))
        }
        return path
    }

    private func racketPath(frame: PoseFrame, mapping: (CGPoint) -> CGPoint) -> CGPath? {
        let candidates: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
            (.rightElbow, .rightWrist),
            (.leftElbow, .leftWrist),
        ]
        for (elbowKey, wristKey) in candidates {
            guard let elbowN = frame.joints[elbowKey], let wristN = frame.joints[wristKey] else { continue }
            let elbow = mapping(elbowN)
            let wrist = mapping(wristN)
            let dx = wrist.x - elbow.x
            let dy = wrist.y - elbow.y
            let forearm = (dx * dx + dy * dy).squareRoot()
            guard forearm > 1 else { continue }
            let ux = dx / forearm
            let uy = dy / forearm
            let handleLen = 1.0 * forearm
            let headLong = 1.1 * forearm
            let headShort = 0.55 * forearm
            let handleEnd = CGPoint(x: wrist.x + ux * handleLen, y: wrist.y + uy * handleLen)
            let headCenter = CGPoint(
                x: handleEnd.x + ux * headLong / 2,
                y: handleEnd.y + uy * headLong / 2
            )
            let angle = atan2(uy, ux)

            let path = CGMutablePath()
            path.move(to: wrist)
            path.addLine(to: handleEnd)
            let transform = CGAffineTransform.identity
                .translatedBy(x: headCenter.x, y: headCenter.y)
                .rotated(by: angle)
            let ovalRect = CGRect(
                x: -headLong / 2,
                y: -headShort / 2,
                width: headLong,
                height: headShort
            )
            path.addEllipse(in: ovalRect, transform: transform)
            return path
        }
        return nil
    }
}
