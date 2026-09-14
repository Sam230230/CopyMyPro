"""
MCP wrapper around spike.py.

Exposes the Phase 1 tracking spike as one MCP tool so Claude can run it
directly on a clip and read the tracking report back. The heavy lifting still
lives in spike.py — this file only captures its stdout and hands it over.

Run the server:
    .venv/bin/python spike_mcp.py

Register it with Claude Code (from this directory):
    claude mcp add copymypro-spike -- "$(pwd)/.venv/bin/python" "$(pwd)/spike_mcp.py"
"""

import io
import os
import contextlib

from mcp.server.mcpserver import MCPServer  # mcp 2.x (was FastMCP in mcp 1.x)

import spike

# spike.analyze resolves samples/ and output/ against the cwd, and an MCP client
# starts the server from wherever it likes. Pin the cwd to this directory so
# relative paths mean the same thing here as they do on the command line.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

mcp = MCPServer("copymypro-spike")


@mcp.tool()
def analyze_swing(video_path: str, lefty: bool = False, racket: bool = True) -> str:
    """Run pose tracking on a tennis-swing video and return the tracking-quality
    report (frame coverage, per-joint visibility, verdict).

    Also writes an annotated video to output/<name>_pose.mp4 — watch it, the
    numbers can look fine while the skeleton is wrong at contact.

    Args:
        video_path: path to the swing video (.mov/.mp4).
        lefty: set for a left-handed player.
        racket: draw the racket overlay (needs efficientdet_lite2.tflite for
            detection; otherwise the racket is guessed from the hand).
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            spike.analyze(video_path, lefty=lefty, racket=racket)
    except SystemExit as e:  # spike.analyze bails with sys.exit on bad input
        return f"{buf.getvalue()}\nERROR: {e}"
    return buf.getvalue()


if __name__ == "__main__":
    mcp.run()
