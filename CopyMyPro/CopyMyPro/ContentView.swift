//
//  ContentView.swift
//  CopyMyPro
//
//  Created by Sam Lee on 2026-06-03.
//

import SwiftUI
import PhotosUI

struct ContentView: View {
    @State private var pickerItem: PhotosPickerItem?
    @State private var pickedVideoURL: URL?
    @State private var isLoadingVideo = false

    var body: some View {
        VStack(spacing: 20) {
            Spacer()

            Text("CopyMyPro")
                .font(.largeTitle.bold())

            Spacer()

            PhotosPicker(
                selection: $pickerItem,
                matching: .videos,
                preferredItemEncoding: .current
            ) {
                Label("Upload Serve", systemImage: "square.and.arrow.up.on.square")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(isLoadingVideo)

            if isLoadingVideo {
                ProgressView()
            }

            Spacer()
        }
        .padding()
        .onChange(of: pickerItem) { _, newItem in
            guard let newItem else { return }
            loadVideo(from: newItem)
        }
        .fullScreenCover(item: $pickedVideoURL) { url in
            ProcessScreen(inputURL: url)
        }
    }

    private func loadVideo(from item: PhotosPickerItem) {
        isLoadingVideo = true
        Task {
            defer { isLoadingVideo = false }
            do {
                if let movie = try await item.loadTransferable(type: PickedMovie.self) {
                    pickedVideoURL = movie.url
                }
            } catch {
                print("Failed to load video: \(error)")
            }
        }
    }
}

struct PickedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let destination = URL.temporaryDirectory.appendingPathComponent(
                "swing-\(UUID().uuidString).\(received.file.pathExtension)"
            )
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.copyItem(at: received.file, to: destination)
            return PickedMovie(url: destination)
        }
    }
}

struct ProcessScreen: View {
    let inputURL: URL
    @Environment(\.dismiss) private var dismiss

    @State private var stage: Stage = .analyzing(progress: 0)
    @State private var userProcessedURL: URL?
    @State private var proProcessedURL: URL?

    enum Stage {
        case analyzing(progress: Double)
        case rendering
        case preparingPro
        case ready
        case failed(String)
    }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.ignoresSafeArea()

            switch stage {
            case .ready:
                if let userProcessedURL {
                    SideBySidePlayerView(userURL: userProcessedURL, proURL: proProcessedURL)
                        .ignoresSafeArea()
                }
            case .failed(let message):
                VStack(spacing: 12) {
                    Text("Couldn't process video")
                        .font(.headline)
                        .foregroundStyle(.white)
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center)
                }
                .padding()
            case .analyzing(let progress):
                progressPane(
                    title: "Analyzing pose",
                    detail: "\(Int(progress * 100))%",
                    value: progress
                )
            case .rendering:
                progressPane(
                    title: "Rendering overlay video",
                    detail: "Baking skeleton into MP4…",
                    value: nil
                )
            case .preparingPro:
                progressPane(
                    title: "Preparing pro reference",
                    detail: "One-time processing…",
                    value: nil
                )
            }

            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title)
                    .symbolRenderingMode(.palette)
                    .foregroundStyle(.white, .black.opacity(0.6))
            }
            .padding()
        }
        .task(id: inputURL) {
            await runPipeline()
        }
    }

    @ViewBuilder
    private func progressPane(title: String, detail: String, value: Double?) -> some View {
        VStack(spacing: 16) {
            if let value {
                ProgressView(value: value)
                    .progressViewStyle(.linear)
                    .tint(.white)
            } else {
                ProgressView()
                    .progressViewStyle(.circular)
                    .tint(.white)
            }
            Text(title)
                .font(.headline)
                .foregroundStyle(.white)
            Text(detail)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.7))
        }
        .padding()
        .frame(maxWidth: 280)
    }

    private func runPipeline() async {
        do {
            let analyzer = PoseAnalyzer()
            let frames = try await analyzer.analyze(videoURL: inputURL) { value in
                Task { @MainActor in
                    if case .analyzing = self.stage {
                        self.stage = .analyzing(progress: value)
                    }
                }
            }

            stage = .rendering
            let renderer = PoseVideoRenderer()
            let outputURL = URL.temporaryDirectory.appendingPathComponent(
                "\(inputURL.deletingPathExtension().lastPathComponent)_pose.mp4"
            )
            let userBaked = try await renderer.render(
                inputURL: inputURL,
                frames: frames,
                outputURL: outputURL
            )
            userProcessedURL = userBaked

            stage = .preparingPro
            proProcessedURL = try? await ProReference.processedURL()

            stage = .ready
        } catch {
            stage = .failed(String(describing: error))
        }
    }
}

extension URL: @retroactive Identifiable {
    public var id: String { absoluteString }
}

#Preview {
    ContentView()
}
