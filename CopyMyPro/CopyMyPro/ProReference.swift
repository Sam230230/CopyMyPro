//
//  ProReference.swift
//  CopyMyPro
//
//  Looks up an optional bundled pro reference video, runs the same
//  pose-analysis + render pipeline on it, and caches the result so it
//  only happens once per install.
//
//  To enable: drop a file named `pro_serve.mov` (or `.mp4`) into the
//  CopyMyPro target. Per CLAUDE.md, the in-app label stays generic
//  ("Pro Reference") — never name a real pro in product UI.
//

import Foundation

enum ProReference {
    private static let bundleBaseName = "pro_serve"
    private static let bundleExtensions = ["mov", "mp4", "MOV", "MP4"]
    private static let cacheFilename = "pro_serve_pose.mp4"

    static func bundledSourceURL() -> URL? {
        for ext in bundleExtensions {
            if let url = Bundle.main.url(forResource: bundleBaseName, withExtension: ext) {
                return url
            }
        }
        return nil
    }

    /// Returns the processed (skeleton-overlaid) pro reference video URL, or nil if
    /// no `pro_serve.mov`/`.mp4` is bundled. Caches across launches.
    static func processedURL() async throws -> URL? {
        guard let sourceURL = bundledSourceURL() else { return nil }

        let cachesDir = try FileManager.default.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let cachedURL = cachesDir.appendingPathComponent(cacheFilename)
        if FileManager.default.fileExists(atPath: cachedURL.path) {
            return cachedURL
        }

        let analyzer = PoseAnalyzer()
        let frames = try await analyzer.analyze(videoURL: sourceURL) { _ in }

        let renderer = PoseVideoRenderer()
        return try await renderer.render(inputURL: sourceURL, frames: frames, outputURL: cachedURL)
    }
}
