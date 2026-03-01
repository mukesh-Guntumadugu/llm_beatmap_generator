#!/usr/bin/env python3
"""
evaluate_llm_beatmap.py  –  CLI for Issue #7 metrics

Runs all 7 evaluation metrics on a generated beatmap and writes a JSON report.

Examples
--------
# Minimal (no audio):
python evaluate_llm_beatmap.py generated.txt

# With audio + BPM (enables beat/onset alignment):
python evaluate_llm_beatmap.py generated.txt \\
    --audio src/musicForBeatmap/Springtime/Springtime.ogg \\
    --bpm 181.685 --offset -0.028

# With original beatmap for step-count comparison:
python evaluate_llm_beatmap.py generated.txt \\
    --audio src/musicForBeatmap/Springtime/Springtime.ogg \\
    --bpm 181.685 --offset -0.028 \\
    --original src/musicForBeatmap/Springtime/beatmap_easy.text \\
    --out results/metrics.json
"""

import argparse
import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from analysis.llm_beatmap_metrics import compute_all_metrics, print_report, save_metrics_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-generated DDR beatmaps (Issue #7 metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("beatmap", help="Path to generated beatmap (.txt / .text)")
    parser.add_argument("--audio", "-a", default=None, metavar="PATH",
                        help="Audio file path (.ogg/.mp3/.wav).  "
                             "Required for beat/onset alignment metrics.")
    parser.add_argument("--bpm", "-b", type=float, default=None,
                        help="Song BPM (e.g. 181.685).  "
                             "Required for timing-based metrics.")
    parser.add_argument("--offset", "-O", type=float, default=0.0,
                        help="Song start offset in seconds (default: 0.0).")
    parser.add_argument("--original", "-r", default=None, metavar="PATH",
                        help="Ground-truth beatmap for step-count comparison.")
    parser.add_argument("--out", "-o", default=None, metavar="PATH",
                        help="Where to save the JSON report.  "
                             "Defaults to <beatmap_stem>_metrics.json.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.path.isfile(args.beatmap):
        print(f"Error: beatmap file not found: {args.beatmap}", file=sys.stderr)
        sys.exit(1)

    if args.audio and not os.path.isfile(args.audio):
        print(f"Warning: audio file not found: {args.audio} – alignment metrics skipped.",
              file=sys.stderr)
        args.audio = None

    if args.original and not os.path.isfile(args.original):
        print(f"Warning: original beatmap not found: {args.original} – comparison skipped.",
              file=sys.stderr)
        args.original = None

    print(f"Evaluating: {args.beatmap}")
    if args.audio:
        print(f"Audio     : {args.audio}  BPM={args.bpm}  offset={args.offset}s")
    else:
        print("Audio     : (not provided – beat/onset alignment skipped)")
    if args.original:
        print(f"Original  : {args.original}")

    metrics = compute_all_metrics(
        beatmap_path=args.beatmap,
        audio_path=args.audio,
        bpm=args.bpm,
        offset=args.offset,
        original_path=args.original,
    )

    print_report(metrics)

    # Determine output path
    out_path = args.out
    if out_path is None:
        stem = os.path.splitext(os.path.basename(args.beatmap))[0]
        out_path = f"{stem}_metrics.json"

    save_metrics_json(metrics, out_path)


if __name__ == "__main__":
    main()
