//
//  SideBySidePlayerView.swift
//  CopyMyPro
//
//  Plays the user's processed clip next to a pro reference clip. Both
//  videos auto-loop together. Container shape decides whether to split
//  horizontally (landscape) or stack vertically (portrait).
//

import AVFoundation
import AVKit
import SwiftUI

struct SideBySidePlayerView: View {
    let userURL: URL
    let proURL: URL?

    @State private var userPlayer: AVPlayer
    @State private var proPlayer: AVPlayer?

    init(userURL: URL, proURL: URL?) {
        self.userURL = userURL
        self.proURL = proURL
        _userPlayer = State(initialValue: AVPlayer(url: userURL))
        if let proURL {
            _proPlayer = State(initialValue: AVPlayer(url: proURL))
        } else {
            _proPlayer = State(initialValue: nil)
        }
    }

    var body: some View {
        GeometryReader { proxy in
            let isLandscape = proxy.size.width > proxy.size.height
            stack(isHorizontal: isLandscape)
                .frame(width: proxy.size.width, height: proxy.size.height)
        }
        .background(Color.black)
        .onAppear { restartAll() }
        .onDisappear { pauseAll() }
    }

    @ViewBuilder
    private func stack(isHorizontal: Bool) -> some View {
        let layout = isHorizontal
            ? AnyLayout(HStackLayout(spacing: 4))
            : AnyLayout(VStackLayout(spacing: 4))
        layout {
            paneLabel("You", player: userPlayer)
            if let proPlayer {
                paneLabel("Pro Reference", player: proPlayer)
            } else {
                missingProPane
            }
        }
    }

    private func paneLabel(_ title: String, player: AVPlayer) -> some View {
        ZStack(alignment: .topLeading) {
            VideoPlayer(player: player)
                .background(Color.black)
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.black.opacity(0.55), in: Capsule())
                .padding(8)
        }
    }

    private var missingProPane: some View {
        ZStack {
            Color.black
            VStack(spacing: 8) {
                Image(systemName: "tennis.racket")
                    .font(.largeTitle)
                    .foregroundStyle(.white.opacity(0.7))
                Text("No pro reference loaded")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                Text("Add `pro_serve.mov` to the app target\nto compare side by side.")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
                    .multilineTextAlignment(.center)
            }
            .padding()
        }
    }

    private func restartAll() {
        userPlayer.seek(to: .zero)
        userPlayer.play()
        if let proPlayer {
            proPlayer.seek(to: .zero)
            proPlayer.play()
        }
    }

    private func pauseAll() {
        userPlayer.pause()
        proPlayer?.pause()
    }
}
