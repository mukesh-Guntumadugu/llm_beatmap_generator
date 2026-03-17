"""
Audio to MIDI Converter using Spotify's Basic-Pitch
=====================================================
Converts any audio file (MP3, OGG, WAV, FLAC) to MIDI using
Spotify's Basic-Pitch neural network model.

Usage:
    # Single file
    python audio_to_midi.py path/to/song.mp3

    # Whole song directory (processes all audio files found)
    python audio_to_midi.py path/to/songs/ --batch

    # With custom frequency range (e.g., bass guitar only)
    python audio_to_midi.py song.mp3 --min-freq 40 --max-freq 300

Install:
    pip install basic-pitch
"""

import os
import sys
import glob
import argparse

# ── Install check ──────────────────────────────────────────────────────────────
try:
    from basic_pitch.inference import predict, predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH
except ImportError:
    print("❌  basic-pitch is not installed.")
    print("    Run:  pip install basic-pitch")
    sys.exit(1)

AUDIO_EXTENSIONS = (".mp3", ".ogg", ".wav", ".flac", ".aiff", ".m4a")


def convert_single(audio_path: str, out_dir: str, min_freq: float = None,
                   max_freq: float = None, onset_threshold: float = 0.5,
                   save_notes_csv: bool = True):
    """
    Convert one audio file to MIDI using Basic-Pitch.
    Saves .mid (and optionally a note-events .csv) to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(audio_path))[0]

    print(f"\n🎵  Converting: {os.path.basename(audio_path)}")
    print(f"    → Output dir: {out_dir}")

    kwargs = dict(
        onset_threshold=onset_threshold,
    )
    if min_freq is not None:
        kwargs["minimum_frequency"] = min_freq
    if max_freq is not None:
        kwargs["maximum_frequency"] = max_freq

    model_output, midi_data, note_events = predict(audio_path, **kwargs)

    # Save MIDI
    midi_out = os.path.join(out_dir, f"{name}.mid")
    midi_data.write(midi_out)
    print(f"    ✅  MIDI saved  → {midi_out}")

    # Save note events CSV (time, pitch, velocity, start, end)
    if save_notes_csv and note_events:
        csv_out = os.path.join(out_dir, f"{name}_note_events.csv")
        with open(csv_out, "w", encoding="utf-8") as f:
            f.write("start_s,end_s,pitch_midi,velocity,pitch_bend\n")
            for ev in note_events:
                # note_events tuples: (start_time, end_time, pitch, amplitude, pitch_bend)
                start, end, pitch, amp, bend = ev
                velocity = int(amp * 127)
                f.write(f"{start:.4f},{end:.4f},{pitch},{velocity},{bend}\n")
        print(f"    ✅  Notes CSV  → {csv_out}")

    return midi_out, note_events


def batch_convert(audio_dir: str, out_dir: str, min_freq=None, max_freq=None,
                  onset_threshold=0.5, save_notes_csv=True):
    """
    Walk audio_dir, convert every audio file found.
    """
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(glob.glob(os.path.join(audio_dir, "**", f"*{ext}"), recursive=True))

    audio_files = sorted(audio_files)
    if not audio_files:
        print(f"❌  No audio files found in: {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio file(s) to convert.\n")

    for i, audio_path in enumerate(audio_files):
        # Mirror directory structure under out_dir
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

    # Resolve output directory
    if args.out_dir:
        out_dir = args.out_dir
    elif args.batch:
        out_dir = os.path.join(args.input, "midi_output")
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)), "midi_output")

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
