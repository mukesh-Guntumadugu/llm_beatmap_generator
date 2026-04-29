#!/usr/bin/env python3
"""
plot_qwen_convergence.py
========================
Parses the Qwen Director training log and plots the loss convergence curve.

Usage:
  python scripts/plot_qwen_convergence.py \
      --log /data/mg546924/llm_beatmap_generator/logs/train_hierarchical_qwen_26683.log \
      --out /data/mg546924/llm_beatmap_generator/logs/qwen_convergence.png
"""

import re
import argparse
import json

def parse_log(log_path):
    epochs, losses, eval_epochs, eval_losses = [], [], [], []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            # Match training step lines: {'loss': X, ..., 'epoch': Y}
            if line.startswith("{'loss':") or line.startswith('{"loss":'):
                try:
                    d = json.loads(line.replace("'", '"'))
                    if "loss" in d and "epoch" in d:
                        losses.append(d["loss"])
                        epochs.append(d["epoch"])
                except Exception:
                    pass
            # Match eval lines: {'eval_loss': X, ..., 'epoch': Y}
            if "eval_loss" in line:
                try:
                    d = json.loads(line.replace("'", '"'))
                    if "eval_loss" in d and "epoch" in d:
                        eval_losses.append(d["eval_loss"])
                        eval_epochs.append(d["epoch"])
                except Exception:
                    pass
    return epochs, losses, eval_epochs, eval_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument("--out", default="qwen_convergence.png", help="Output PNG path")
    args = parser.parse_args()

    epochs, losses, eval_epochs, eval_losses = parse_log(args.log)

    if not losses:
        print("ERROR: No loss values found in log file. Check the path.")
        return

    print(f"Parsed {len(losses)} training steps across {max(epochs):.1f} epochs.")
    print(f"Loss range: {min(losses):.4f} → {max(losses):.4f}")
    print(f"Final train loss: {losses[-1]:.4f}")
    if eval_losses:
        print(f"Final eval loss:  {eval_losses[-1]:.4f}")

    # ── Compute per-epoch average loss ──
    from collections import defaultdict
    epoch_buckets = defaultdict(list)
    for e, l in zip(epochs, losses):
        bucket = round(e * 4) / 4  # round to nearest 0.25 epoch
        epoch_buckets[bucket].append(l)
    avg_epochs = sorted(epoch_buckets.keys())
    avg_losses  = [sum(epoch_buckets[e]) / len(epoch_buckets[e]) for e in avg_epochs]

    # ── Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0f0f17")
    ax.set_facecolor("#0f0f17")

    # Raw per-step (light, low opacity)
    ax.plot(epochs, losses, color="#4a7fd4", alpha=0.18, linewidth=0.6, label="Step loss (raw)")

    # Smoothed per-0.25-epoch average (bright)
    ax.plot(avg_epochs, avg_losses, color="#4a7fd4", linewidth=2.5, label="Avg loss / 0.25 epoch")

    # Eval loss dots
    if eval_losses:
        ax.scatter(eval_epochs, eval_losses, color="#f0b429", s=80, zorder=5,
                   label="Eval loss (checkpoint)", marker="D")

    # Epoch dividers
    for ep in range(1, 6):
        ax.axvline(ep, color="#ffffff", alpha=0.08, linewidth=0.8, linestyle="--")
        ax.text(ep - 0.5, max(losses) * 0.95, f"Epoch {ep}", color="#888",
                fontsize=8, ha="center", va="top")

    # Highlight final loss
    ax.annotate(
        f"Final train loss\n{losses[-1]:.4f}",
        xy=(epochs[-1], losses[-1]),
        xytext=(epochs[-1] - 0.6, losses[-1] + 0.3),
        color="#4af0b4",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#4af0b4"),
    )

    ax.set_xlabel("Epoch", color="#aaa", fontsize=12)
    ax.set_ylabel("Loss", color="#aaa", fontsize=12)
    ax.set_title("Qwen2-Audio Director — Training Convergence (5 Epochs)", color="#eee", fontsize=14, pad=16)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    ax.legend(facecolor="#1a1a2e", labelcolor="#ccc", fontsize=9, framealpha=0.8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[OK] Plot saved to: {args.out}")


if __name__ == "__main__":
    main()
