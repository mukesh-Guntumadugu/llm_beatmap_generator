"""
score_onset_detection.py
========================
Compares Qwen-predicted onsets against librosa ground-truth onsets and
reports per-song Precision, Recall, and F1 scores.

A predicted onset is considered CORRECT if it lands within ±TOLERANCE_MS
of any ground-truth onset (default tolerance = 50 ms).

Usage
-----
  # Score a single pair:
  python score_onset_detection.py \\
      --ref  path/to/original_onsets_SongName_*.csv \\
      --pred path/to/Qwen_onsets_SongName_*.csv

  # Batch mode — auto-match all pairs inside the Fraxtil directory:
  python score_onset_detection.py --batch

  # Batch mode with custom tolerance (e.g. 100 ms):
  python score_onset_detection.py --batch --tolerance 100
"""

import os
import re
import csv
import glob
import argparse
import datetime
import numpy as np
from typing import Optional, List

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

DEFAULT_TOLERANCE_MS = 50.0   # ±50 ms window for a correct match

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_onsets_csv(csv_path: str) -> np.ndarray:
    """Load onset_ms column from a CSV file and return a sorted numpy array."""
    onsets = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                onsets.append(float(row["onset_ms"]))
            except (KeyError, ValueError):
                pass
    return np.array(sorted(onsets))


def find_latest_file(directory: str, pattern: str) -> Optional[str]:
    """Return the most recently created file matching a glob pattern."""
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


# ── Scoring logic ─────────────────────────────────────────────────────────────

def score_onsets(
    ref_onsets: np.ndarray,
    pred_onsets: np.ndarray,
    tolerance_ms: float = DEFAULT_TOLERANCE_MS
) -> dict:
    """
    Compare predicted onsets to reference onsets using a greedy matching strategy.

    Each predicted onset may match at most ONE reference onset (closest within
    the tolerance window). This mirrors standard onset detection evaluation
    (as used in mir_eval / MIREX).

    Returns a dict with:
        tp         – true positives (correctly predicted onsets)
        fp         – false positives (predicted onsets with no ref match)
        fn         – false negatives (ref onsets missed by the model)
        precision  – tp / (tp + fp)
        recall     – tp / (tp + fn)
        f1         – 2 * precision * recall / (precision + recall)
    """
    if len(ref_onsets) == 0 and len(pred_onsets) == 0:
        return dict(tp=0, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)
    if len(ref_onsets) == 0:
        return dict(tp=0, fp=len(pred_onsets), fn=0, precision=0.0, recall=1.0, f1=0.0)
    if len(pred_onsets) == 0:
        return dict(tp=0, fp=0, fn=len(ref_onsets), precision=1.0, recall=0.0, f1=0.0)

    ref_used   = np.zeros(len(ref_onsets), dtype=bool)
    pred_matched = np.zeros(len(pred_onsets), dtype=bool)

    # Greedy: for each predicted onset (in order), find closest unmatched ref onset
    for pi, p in enumerate(pred_onsets):
        diffs = np.abs(ref_onsets - p)
        diffs[ref_used] = np.inf        # mask already-matched ref onsets
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= tolerance_ms:
            pred_matched[pi] = True
            ref_used[min_idx] = True

    tp = int(pred_matched.sum())
    fp = int((~pred_matched).sum())
    fn = int((~ref_used).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return dict(
        tp=tp, fp=fp, fn=fn,
        precision=round(precision * 100, 2),
        recall=round(recall * 100, 2),
        f1=round(f1 * 100, 2)
    )


def print_score(song_name: str, metrics: dict, n_ref: int, n_pred: int):
    print(f"\n  Song    : {song_name}")
    print(f"  Ref     : {n_ref:,} librosa onsets  |  Predicted: {n_pred:,} Qwen onsets")
    print(f"  TP={metrics['tp']:,}  FP={metrics['fp']:,}  FN={metrics['fn']:,}")
    print(f"  Precision : {metrics['precision']:.2f}%")
    print(f"  Recall    : {metrics['recall']:.2f}%")
    print(f"  F1 Score  : {metrics['f1']:.2f}%")


# ── Single-pair mode ──────────────────────────────────────────────────────────

def score_pair(ref_path: str, pred_path: str, tolerance_ms: float):
    song_name = os.path.basename(os.path.dirname(ref_path))
    print(f"\n{'='*70}")
    print(f"  Reference : {os.path.basename(ref_path)}")
    print(f"  Predicted : {os.path.basename(pred_path)}")
    print(f"  Tolerance : ±{tolerance_ms:.0f} ms")
    print(f"{'='*70}")

    ref_onsets  = load_onsets_csv(ref_path)
    pred_onsets = load_onsets_csv(pred_path)
    metrics     = score_onsets(ref_onsets, pred_onsets, tolerance_ms)
    print_score(song_name, metrics, len(ref_onsets), len(pred_onsets))


# ── Batch mode ────────────────────────────────────────────────────────────────

def batch_score(tolerance_ms: float):
    """
    For each song in the Fraxtil dataset, find the latest original_onsets_*.csv
    and the latest Qwen_onsets_*.csv in the same song directory, then score them.
    Saves a summary CSV to the project root.
    """
    if not os.path.isdir(BASE_DIR):
        print(f"❌ Dataset directory not found:\n   {BASE_DIR}")
        return

    song_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
        and not d.startswith("_") and not d.startswith(".")
    ])

    results = []
    skipped = []

    print(f"\n{'='*110}")
    print(f"  Batch Onset Scoring  |  Tolerance: ±{tolerance_ms:.0f} ms  |  Songs: {len(song_dirs)}")
    print(f"{'='*110}")

    header = f"{'Song':<45} {'#Ref':>6} {'#Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec%':>7} {'Rec%':>7} {'F1%':>7}"
    print(header)
    print("─" * 110)

    for song_name in song_dirs:
        song_dir = os.path.join(BASE_DIR, song_name)

        ref_path  = find_latest_file(song_dir, "original_onsets_*.csv")
        pred_path = find_latest_file(song_dir, "Qwen_onsets_*.csv")

        if ref_path is None:
            skipped.append((song_name, "No librosa onset CSV found (run extract_librosa_onsets.py first)"))
            continue
        if pred_path is None:
            skipped.append((song_name, "No Qwen onset CSV found (run extract_qwen_onsets.py first)"))
            continue

        try:
            ref_onsets  = load_onsets_csv(ref_path)
            pred_onsets = load_onsets_csv(pred_path)
            metrics     = score_onsets(ref_onsets, pred_onsets, tolerance_ms)

            row_str = (
                f"  {song_name:<43} {len(ref_onsets):>6,} {len(pred_onsets):>6,} "
                f"{metrics['tp']:>6} {metrics['fp']:>6} {metrics['fn']:>6} "
                f"{metrics['precision']:>7.2f} {metrics['recall']:>7.2f} {metrics['f1']:>7.2f}"
            )
            print(row_str)

            results.append({
                "song":           song_name,
                "n_ref":          len(ref_onsets),
                "n_pred":         len(pred_onsets),
                "tp":             metrics["tp"],
                "fp":             metrics["fp"],
                "fn":             metrics["fn"],
                "precision_pct":  metrics["precision"],
                "recall_pct":     metrics["recall"],
                "f1_pct":         metrics["f1"],
                "tolerance_ms":   tolerance_ms,
                "ref_file":       os.path.basename(ref_path),
                "pred_file":      os.path.basename(pred_path),
            })

        except Exception as e:
            skipped.append((song_name, str(e)))

    print("─" * 110)

    # ── Aggregate summary ────────────────────────────────────────────────────
    if results:
        avg_prec = np.mean([r["precision_pct"] for r in results])
        avg_rec  = np.mean([r["recall_pct"]    for r in results])
        avg_f1   = np.mean([r["f1_pct"]        for r in results])
        best     = max(results, key=lambda r: r["f1_pct"])
        worst    = min(results, key=lambda r: r["f1_pct"])

        print(f"\n  Scored {len(results)}/{len(song_dirs)} songs")
        print(f"  Average Precision : {avg_prec:.2f}%")
        print(f"  Average Recall    : {avg_rec:.2f}%")
        print(f"  Average F1        : {avg_f1:.2f}%")
        print(f"  Best  F1 → {best['song']} ({best['f1_pct']:.2f}%)")
        print(f"  Worst F1 → {worst['song']} ({worst['f1_pct']:.2f}%)")

        # Save summary CSV
        ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
        summary_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"onset_score_summary_{ts}.csv"
        )
        fieldnames = list(results[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  📄 Summary saved → {summary_path}")

    if skipped:
        print(f"\n  ⚠️  Skipped {len(skipped)} song(s):")
        for song, reason in skipped:
            print(f"     • {song}: {reason}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score Qwen onset detection against librosa ground truth."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--batch", action="store_true",
        help="Auto-discover and score all songs in the Fraxtil dataset."
    )
    mode.add_argument(
        "--ref", type=str,
        help="Path to reference (librosa) onset CSV (single-pair mode)."
    )
    parser.add_argument(
        "--pred", type=str,
        help="Path to predicted (Qwen) onset CSV (required with --ref)."
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE_MS,
        help=f"Matching tolerance in ms (default: {DEFAULT_TOLERANCE_MS})."
    )
    args = parser.parse_args()

    if args.batch:
        batch_score(args.tolerance)
    elif args.ref:
        if not args.pred:
            parser.error("--pred is required when using --ref")
        if not os.path.isfile(args.ref):
            print(f"❌ Reference file not found: {args.ref}")
            return
        if not os.path.isfile(args.pred):
            print(f"❌ Predicted file not found: {args.pred}")
            return
        score_pair(args.ref, args.pred, args.tolerance)
    else:
        # Default: run batch mode
        print("No arguments given — running in batch mode (--batch).")
        batch_score(args.tolerance)


if __name__ == "__main__":
    main()
