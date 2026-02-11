#!/usr/bin/env python3
"""
Plot Training Loss from Log File
Usage: python3 src/analysis/plot_loss.py [log_file_path]
If no log file specified, uses the most recent one.
"""
import re
import sys
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime

def parse_log_file(log_path):
    """Extract epoch and loss from log file."""
    epochs = []
    losses = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match lines like: "Epoch 10/300 Completed. Avg Loss: 1.2595"
            match = re.search(r'Epoch (\d+)/\d+ Completed\. Avg Loss: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)
    
    return epochs, losses

def plot_loss(epochs, losses, log_path, output_path=None):
    """Create and save loss plot."""
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, linewidth=2, color='#2563eb')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add final loss annotation
    if epochs and losses:
        final_epoch = epochs[-1]
        final_loss = losses[-1]
        plt.annotate(
            f'Final: {final_loss:.4f}',
            xy=(final_epoch, final_loss),
            xytext=(final_epoch * 0.7, final_loss * 1.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10,
            color='red'
        )
    
    plt.tight_layout()
    
    # Determine output path
    if output_path is None:
        log_dir = os.path.dirname(log_path)
        log_basename = os.path.basename(log_path).replace('.log', '')
        output_path = os.path.join(log_dir, f'{log_basename}_graph.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Graph saved to: {output_path}")
    
    # Also save to a standard location for easy access
    simple_output = "src/training/loss_graph.png"
    plt.savefig(simple_output, dpi=150, bbox_inches='tight')
    print(f"✓ Also saved to: {simple_output}")
    
    plt.close()

def get_latest_log():
    """Find the most recent log file."""
    log_dir = "src/training/logs"
    log_files = glob.glob(os.path.join(log_dir, "training_run_*.log"))
    if not log_files:
        print(f"Error: No log files found in {log_dir}")
        sys.exit(1)
    
    # Sort by modification time
    latest = max(log_files, key=os.path.getmtime)
    return latest

def main():
    # Determine log file
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = get_latest_log()
        print(f"Using latest log: {log_path}")
    
    if not os.path.exists(log_path):
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)
    
    # Parse and plot
    print("Parsing log file...")
    epochs, losses = parse_log_file(log_path)
    
    if not epochs:
        print("Error: No epoch data found in log file")
        sys.exit(1)
    
    print(f"Found {len(epochs)} epochs")
    print(f"Loss range: {min(losses):.4f} to {max(losses):.4f}")
    
    plot_loss(epochs, losses, log_path)
    print("\nDone!")

if __name__ == "__main__":
    main()
