"""
Audio to MIDI Converter using Spotify's Basic-Pitch
=====================================================
Converts any audio file (MP3, OGG, WAV, FLAC) to MIDI using
Spotify's Basic-Pitch neural network model (via the CLI).

Uses the CLI tool instead of the Python API to work around a
pkg_resources incompatibility in Python 3.12.

Usage:
    # Single file
    python audio_to_midi.py path/to/song.ogg

    # Whole song directory (processes all audio files found)
    python audio_to_midi.py path/to/songs/ --batch

    # With custom frequency range (e.g., bass guitar only)
    python audio_to_midi.py song.mp3 --min-freq 40 --max-freq 300

Install:
    pip install 'basic-pitch[onnx]'
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess

AUDIO_EXTENSIONS = (".mp3", ".ogg", ".wav", ".flac", ".aiff", ".m4a")


def check_basic_pitch():
    """Verify basic-pitch CLI is on PATH."""
    if shutil.which("basic-pitch") is None:
        print("❌  'basic-pitch' CLI not found.")
        print("    Install with:  pip install 'basic-pitch[onnx]'")
        sys.exit(1)


def convert_single(audio_path: str, out_dir: str,
                   min_freq: float = None, max_freq: float = None,
                   onset_threshold: float = 0.5, save_notes_csv: bool = True):
    """Convert one audio file to MIDI using the basic-pitch CLI."""
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(audio_path))[0]
    print(f"\n🎵  Converting: {os.path.basename(audio_path)}")
    print(f"    → Output dir: {out_dir}")

    cmd = ["basic-pitch", out_dir, audio_path, "--model-serialization", "onnx"]

    if save_notes_csv:
        cmd.append("--save-note-events")

    if onset_threshold != 0.5:
        cmd += ["--onset-threshold", str(onset_threshold)]

    if min_freq is not None:
        cmd += ["--minimum-frequency", str(min_freq)]

    if max_freq is not None:
        cmd += ["--maximum-frequency", str(max_freq)]

    result = subprocess.run(cmd)   # stream output directly — no capture

    if result.returncode != 0:
        print(f"    ⚠️   basic-pitch exited with code {result.returncode}")
        return None

    midi_out = os.path.join(out_dir, f"{name}_basic_pitch.mid")
    if os.path.exists(midi_out):
        print(f"    ✅  MIDI saved  → {midi_out}")
    else:
        # basic-pitch may use a slightly different filename; find it
        candidates = glob.glob(os.path.join(out_dir, f"{name}*.mid"))
        if candidates:
            print(f"    ✅  MIDI saved  → {candidates[0]}")
        else:
            print(f"    ⚠️   MIDI file not found in {out_dir}")

    return midi_out


def batch_convert(audio_dir: str, out_dir: str, min_freq=None, max_freq=None,
                  onset_threshold=0.5, save_notes_csv=True):
    """Walk audio_dir, convert every audio file found."""
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(
            glob.glob(os.path.join(audio_dir, "**", f"*{ext}"), recursive=True)
        )
    audio_files = sorted(audio_files)

    if not audio_files:
        print(f"❌  No audio files found in: {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio file(s) to convert.\n")
    for i, audio_path in enumerate(audio_files):
        rel = os.path.relpath(os.path.dirname(audio_path), audio_dir)
        song_out = os.path.join(out_dir, rel)
        print(f"[{i+1}/{len(audio_files)}] ", end="")
        try:
            convert_single(audio_path, song_out, min_freq, max_freq,
                           onset_threshold, save_notes_csv)
        except Exception as e:
            print(f"    ⚠️   Failed: {e}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files to MIDI using Spotify Basic-Pitch."
    )
    parser.add_argument("input", help="Audio file or directory (use --batch for directory)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: midi_output/ next to input)")
    parser.add_argument("--batch", action="store_true",
                        help="Process all audio files in the input directory")
    parser.add_argument("--min-freq", type=float, default=None,
                        help="Minimum note frequency in Hz (e.g. 40 for bass)")
    parser.add_argument("--max-freq", type=float, default=None,
                        help="Maximum note frequency in Hz (e.g. 2000 for melody)")
    parser.add_argument("--onset-threshold", type=float, default=0.5,
                        help="Note onset sensitivity 0.0-1.0 (lower = more notes)")
    parser.add_argument("--no-csv", action="store_true",
                        help="Skip saving note events CSV")

    args = parser.parse_args()

    check_basic_pitch()

    # Resolve output directory
    if args.out_dir:
        out_dir = args.out_dir
    elif args.batch:
        out_dir = os.path.join(args.input, "midi")
    else:
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.input)), "midi"
        )

    if args.batch:
        if not os.path.isdir(args.input):
            print(f"❌  Not a directory: {args.input}")
            sys.exit(1)
        batch_convert(args.input, out_dir, args.min_freq, args.max_freq,
                      args.onset_threshold, not args.no_csv)
    else:
        if not os.path.isfile(args.input):
            print(f"❌  File not found: {args.input}")
            sys.exit(1)
        convert_single(args.input, out_dir, args.min_freq, args.max_freq,
                       args.onset_threshold, not args.no_csv)


if __name__ == "__main__":
    main()
