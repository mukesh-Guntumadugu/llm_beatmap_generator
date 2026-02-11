"""
Beatmap Comparison Utility

This module provides functions to compare generated beatmaps with original beatmaps
and calculate various accuracy metrics for W&B tracking.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from typing import List, Tuple, Dict

def load_beatmap(filepath: str) -> List[str]:
    """
    Load beatmap from .text file.
    
    Args:
        filepath: Path to beatmap .text file
        
    Returns:
        List of note rows (e.g., ['0000', '1000', '0200', ...])
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Remove empty lines and measure separators (commas)
    beatmap_lines = [line for line in lines if line and line != ',']
    return beatmap_lines


def load_beatmap_with_measures(filepath: str) -> List[List[str]]:
    """
    Load beatmap from .text file, preserving measure structure.
    
    Each comma (,) represents a measure boundary (temporal segment).
    0000 lines are preserved as they represent meaningful timing data.
    
    Args:
        filepath: Path to beatmap .text file
        
    Returns:
        List of measures, where each measure is a list of note rows
        Example: [['1000', '0001', '0000', '0001'], ['1000', '0200'], ...]
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    measures = []
    current_measure = []
    
    for line in lines:
        if not line:  # Skip empty lines
            continue
        elif line == ',':  # Measure boundary
            if current_measure:  # Only add non-empty measures
                measures.append(current_measure)
                current_measure = []
        else:
            current_measure.append(line)
    
    # Add final measure if exists
    if current_measure:
        measures.append(current_measure)
    
    return measures


def normalize_beatmap_length(generated: List[str], original: List[str]) -> Tuple[List[str], List[str]]:
    """
    Normalize two beatmaps to the same length by padding or truncating.
    
    Args:
        generated: Generated beatmap lines
        original: Original beatmap lines
        
    Returns:
        Tuple of (normalized_generated, normalized_original)
    """
    max_len = max(len(generated), len(original))
    
    # Pad with empty notes if needed
    gen_padded = generated + ['0000'] * (max_len - len(generated))
    orig_padded = original + ['0000'] * (max_len - len(original))
    
    return gen_padded[:max_len], orig_padded[:max_len]


def calculate_note_accuracy(generated: List[str], original: List[str]) -> Dict[str, float]:
    """
    Calculate note-level accuracy metrics.
    
    Args:
        generated: Generated beatmap lines
        original: Original beatmap lines
        
    Returns:
        Dictionary with accuracy metrics
    """
    gen_norm, orig_norm = normalize_beatmap_length(generated, original)
    
    # Exact match accuracy
    exact_matches = sum(1 for g, o in zip(gen_norm, orig_norm) if g == o)
    exact_accuracy = (exact_matches / len(gen_norm) * 100) if gen_norm else 0
    
    # Per-position accuracy (each of 4 arrows)
    position_correct = [0, 0, 0, 0]
    for g, o in zip(gen_norm, orig_norm):
        # Skip if either line is not exactly 4 characters (safety check)
        if len(g) != 4 or len(o) != 4:
            continue
        for i in range(4):
            if g[i] == o[i]:
                position_correct[i] += 1
    
    position_accuracy = [(count / len(gen_norm) * 100) if gen_norm else 0 
                         for count in position_correct]
    
    # Overall note accuracy (any correct note in the row)
    any_note_correct = sum(1 for g, o in zip(gen_norm, orig_norm) 
                          if len(g) == 4 and len(o) == 4 and 
                          any(g[i] == o[i] and  o[i] != '0' for i in range(4)))
    valid_orig = [o for o in orig_norm if len(o) == 4 and o != '0000']
    any_note_accuracy = (any_note_correct / len(valid_orig) * 100) if valid_orig else 0
    
    return {
        'exact_match_accuracy': exact_accuracy,
        'left_accuracy': position_accuracy[0],
        'down_accuracy': position_accuracy[1],
        'up_accuracy': position_accuracy[2],
        'right_accuracy': position_accuracy[3],
        'any_note_accuracy': any_note_accuracy,
        'total_lines': len(gen_norm),
        'exact_matches': exact_matches
    }


def calculate_density_metrics(generated: List[str], original: List[str]) -> Dict[str, any]:
    """
    Calculate density-related metrics (notes per measure, etc.).
    
    Args:
        generated: Generated beatmap lines
        original: Original beatmap lines
        
    Returns:
        Dictionary with density metrics
    """
    def count_notes(beatmap):
        """Count total notes in beatmap"""
        return sum(sum(1 for c in line if c != '0') for line in beatmap)
    
    gen_notes = count_notes(generated)
    orig_notes = count_notes(original)
    
    # Note density difference
    density_diff = abs(gen_notes - orig_notes)
    density_ratio = (gen_notes / orig_notes) if orig_notes > 0 else 0
    
    return {
        'generated_total_notes': gen_notes,
        'original_total_notes': orig_notes,
        'density_difference': density_diff,
        'density_ratio': density_ratio,
        'density_accuracy': (1 - abs(1 - density_ratio)) * 100 if orig_notes > 0 else 0
    }


def calculate_per_direction_metrics(generated: List[str], original: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, and F1 for each arrow direction.
    
    Args:
        generated: Generated beatmap lines
        original: Original beatmap lines
        
    Returns:
        Dictionary with per-direction metrics
    """
    gen_norm, orig_norm = normalize_beatmap_length(generated, original)
    
    directions = ['left', 'down', 'up', 'right']
    metrics = {}
    
    for i, direction in enumerate(directions):
        # Binary classification: 0 (no note) vs non-zero (note present)
        y_true = [1 if len(orig_norm[j]) == 4 and orig_norm[j][i] != '0' else 0 for j in range(len(orig_norm))]
        y_pred = [1 if len(gen_norm[j]) == 4 and gen_norm[j][i] != '0' else 0 for j in range(len(gen_norm))]
        
        precision = precision_score(y_true, y_pred, zero_division=0) * 100
        recall = recall_score(y_true, y_pred, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100
        
        metrics[direction] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    return metrics


def generate_confusion_matrix_data(generated: List[str], original: List[str]) -> Dict[str, any]:
    """
    Generate confusion matrix for note predictions.
    
    Args:
        generated: Generated beatmap lines
        original: Original beatmap lines
        
    Returns:
        Dictionary with confusion matrix data
    """
    gen_norm, orig_norm = normalize_beatmap_length(generated, original)
    
    # Flatten all positions into single arrays
    all_true = []
    all_pred = []
    
    for g, o in zip(gen_norm, orig_norm):
        # Only process lines with exactly 4 characters
        if len(g) == 4 and len(o) == 4:
            for i in range(4):
                all_true.append(int(o[i]))
                all_pred.append(int(g[i]))
    
    # Create confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3, 4])
    
    return {
        'confusion_matrix': cm.tolist(),
        'labels': ['Empty', 'Left', 'Down', 'Up', 'Right'],
        'y_true': all_true,
        'y_pred': all_pred
    }


def generate_concurrency_confusion_matrix(generated: List[str], original: List[str]) -> Dict[str, any]:
    """
    Generate confusion matrix for note concurrency (simultaneous notes per line).
    
    Classifies each line by number of simultaneous notes:
    - 0 notes: "0000" (empty)
    - 1 note: "1000", "0100", "0010", "0001" (single tap)
    - 2 notes: "1100", "1010", etc. (double/jump)
    - 3 notes: "1110", etc. (triple/hand)
    - 4 notes: "1111" (quad)
    
    Args:
        generated: Generated beatmap lines
        original: Original beatmap lines
        
    Returns:
        Dictionary with concurrency confusion matrix data
    """
    gen_norm, orig_norm = normalize_beatmap_length(generated, original)
    
    def count_notes(line: str) -> int:
        """Count number of non-zero notes in a line."""
        if len(line) != 4:
            return 0
        return sum(1 for c in line if c != '0')
    
    # Count simultaneous notes for each line
    orig_concurrency = [count_notes(line) for line in orig_norm]
    gen_concurrency = [count_notes(line) for line in gen_norm]
    
    # Create confusion matrix
    cm = confusion_matrix(orig_concurrency, gen_concurrency, labels=[0, 1, 2, 3, 4])
    
    # Class labels
    class_labels = ['Empty (0)', 'Single (1)', 'Double (2)', 'Triple (3)', 'Quad (4)']
    
    return {
        'confusion_matrix': cm.tolist(),
        'labels': class_labels,
        'y_true': orig_concurrency,
        'y_pred': gen_concurrency
    }



def calculate_timing_alignment(generated: List[str], original: List[str], 
                               tolerance: int = 2) -> Dict[str, float]:
    """
    Calculate timing alignment score allowing small offsets.
    
    Args:
        generated: Generated beatmap lines
        original: Original beatmap lines
        tolerance: Number of line offsets allowed (±2 lines)
        
    Returns:
        Dictionary with timing metrics
    """
    gen_norm, orig_norm = normalize_beatmap_length(generated, original)
    
    aligned_matches = 0
    total_notes_orig = 0
    
    # For each note in original, check if it appears within tolerance in generated
    for orig_idx, orig_line in enumerate(orig_norm):
        if orig_line == '0000':
            continue
            
        total_notes_orig += 1
        
        # Check within tolerance window
        start_idx = max(0, orig_idx - tolerance)
        end_idx = min(len(gen_norm), orig_idx + tolerance + 1)
        
        # Check if any line in window matches
        if any(gen_norm[i] == orig_line for i in range(start_idx, end_idx)):
            aligned_matches += 1
    
    timing_accuracy = (aligned_matches / total_notes_orig * 100) if total_notes_orig > 0 else 0
    
    return {
        'timing_accuracy': timing_accuracy,
        'aligned_matches': aligned_matches,
        'total_original_notes': total_notes_orig,
        'tolerance': tolerance
    }


def calculate_measure_accuracy(gen_measure: List[str], orig_measure: List[str]) -> Dict[str, any]:
    """
    Calculate accuracy for a single measure.
    
    Preserves temporal structure - does NOT normalize lengths with padding.
    If measures have different lengths, that's a temporal mismatch.
    
    Args:
        gen_measure: Generated measure (list of note lines)
        orig_measure: Original measure (list of note lines)
        
    Returns:
        Dictionary with measure-level metrics
    """
    # Exact match count (line-by-line)
    min_len = min(len(gen_measure), len(orig_measure))
    exact_matches = sum(1 for i in range(min_len) if gen_measure[i] == orig_measure[i])
    
    # Length difference penalty
    length_diff = abs(len(gen_measure) - len(orig_measure))
    max_len = max(len(gen_measure), len(orig_measure))
    
    # Accuracy: exact matches / max length (penalizes extra or missing lines)
    accuracy = (exact_matches / max_len * 100) if max_len > 0 else 100.0
    
    # Note counts
    gen_notes = sum(sum(1 for c in line if c != '0') for line in gen_measure)
    orig_notes = sum(sum(1 for c in line if c != '0') for line in orig_measure)
    
    return {
        'accuracy': accuracy,
        'exact_matches': exact_matches,
        'gen_lines': len(gen_measure),
        'orig_lines': len(orig_measure),
        'length_diff': length_diff,
        'gen_notes': gen_notes,
        'orig_notes': orig_notes,
        'is_perfect': gen_measure == orig_measure
    }


def calculate_temporal_metrics(generated_filepath: str, original_filepath: str) -> Dict[str, any]:
    """
    Calculate measure-by-measure temporal accuracy metrics.
    
    This preserves the temporal structure of the beatmap and shows
    accuracy progression over time (by measure).
    
    Args:
        generated_filepath: Path to generated beatmap
        original_filepath: Path to original beatmap
        
    Returns:
        Dictionary with temporal metrics and per-measure data
    """
    # Load with measure structure
    gen_measures = load_beatmap_with_measures(generated_filepath)
    orig_measures = load_beatmap_with_measures(original_filepath)
    
    # Calculate per-measure accuracy
    measure_accuracies = []
    measure_details = []
    
    max_measures = max(len(gen_measures), len(orig_measures))
    
    for i in range(max_measures):
        # Get measures, use empty list if index out of bounds
        gen_m = gen_measures[i] if i < len(gen_measures) else []
        orig_m = orig_measures[i] if i < len(orig_measures) else []
        
        # Calculate measure accuracy
        m_acc = calculate_measure_accuracy(gen_m, orig_m)
        measure_accuracies.append(m_acc['accuracy'])
        measure_details.append({
            'measure_num': i + 1,
            **m_acc
        })
    
    # Overall statistics
    avg_accuracy = np.mean(measure_accuracies) if measure_accuracies else 0
    std_accuracy = np.std(measure_accuracies) if measure_accuracies else 0
    
    # Find best/worst measures
    if measure_accuracies:
        best_idx = np.argmax(measure_accuracies)
        worst_idx = np.argmin(measure_accuracies)
    else:
        best_idx = worst_idx = 0
    
    # Count perfect measures
    perfect_measures = sum(1 for m in measure_details if m['is_perfect'])
    
    return {
        'temporal_accuracy': avg_accuracy,
        'temporal_std': std_accuracy,
        'total_measures': max_measures,
        'perfect_measures': perfect_measures,
        'perfect_measure_ratio': (perfect_measures / max_measures * 100) if max_measures > 0 else 0,
        'best_measure': {
            'measure_num': best_idx + 1,
            'accuracy': measure_accuracies[best_idx] if measure_accuracies else 0
        },
        'worst_measure': {
            'measure_num': worst_idx + 1,
            'accuracy': measure_accuracies[worst_idx] if measure_accuracies else 0
        },
        'measure_accuracies': measure_accuracies,
        'measure_details': measure_details
    }


def generate_density_confusion_matrix(generated_filepath: str, original_filepath: str) -> Dict[str, any]:
    """
    Generate confusion matrix for measure density (line count per measure).
    
    Shows how well the model predicts the number of lines in each measure:
    - 4 lines (Low density)
    - 8 lines (Medium density)  
    - 12 lines (High density)
    - 16+ lines (Very high density)
    
    Args:
        generated_filepath: Path to generated beatmap
        original_filepath: Path to original beatmap
        
    Returns:
        Dictionary with density confusion matrix data
    """
    gen_measures = load_beatmap_with_measures(generated_filepath)
    orig_measures = load_beatmap_with_measures(original_filepath)
    
    # Get exact line counts for each measure
    gen_line_counts = [len(m) for m in gen_measures]
    orig_line_counts = [len(m) for m in orig_measures]
    
    # Ensure same length for comparison
    max_measures = max(len(gen_line_counts), len(orig_line_counts))
    
    # Pad shorter one with zeros (empty measures)
    while len(gen_line_counts) < max_measures:
        gen_line_counts.append(0)
    while len(orig_line_counts) < max_measures:
        orig_line_counts.append(0)
    
    # Use exact line counts (4, 8, 12, 16, 20, 24)
    # Create confusion matrix with all possible line counts
    possible_counts = sorted(list(set(gen_line_counts + orig_line_counts)))
    
    # Map actual counts to class indices for W&B
    count_to_idx = {count: idx for idx, count in enumerate(possible_counts)}
    orig_indices = [count_to_idx[count] for count in orig_line_counts]
    gen_indices = [count_to_idx[count] for count in gen_line_counts]
    
    cm = confusion_matrix(orig_indices, gen_indices, labels=list(range(len(possible_counts))))
    
    # Class labels showing exact line counts
    class_labels = [f'{count} lines' for count in possible_counts]
    
    return {
        'confusion_matrix': cm.tolist(),
        'labels': class_labels,
        'y_true': orig_indices,  # Use indices for W&B
        'y_pred': gen_indices,   # Use indices for W&B
        'total_measures': max_measures
    }


def plot_accuracy_over_time(measure_accuracies: List[float], 
                             measure_details: List[Dict] = None,
                             output_path: str = 'accuracy_over_time.png',
                             title: str = 'Beatmap Accuracy Over Time') -> str:
    """
    Create a comprehensive visualization of accuracy and metrics progression over time.
    
    Similar to training loss graphs, shows multiple metrics on dual axes.
    
    Args:
        measure_accuracies: List of accuracy percentages for each measure
        measure_details: List of detailed metrics per measure (optional)
        output_path: Path to save the plot
        title: Title for the plot
        
    Returns:
        Path to saved plot
    """
    if not measure_accuracies:
        return None
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    measures = list(range(1, len(measure_accuracies) + 1))
    
    # Primary axis: Overall accuracy
    ax1.set_xlabel('Measure Number (Time Progression)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy / Error (%)', fontsize=12, fontweight='bold')
    
    # Calculate error rate (inverse of accuracy = "loss")
    error_rates = [100.0 - acc for acc in measure_accuracies]
    
    # Plot measure accuracy
    line1 = ax1.plot(measures, measure_accuracies, linewidth=2.5, color='#2563eb', 
                     marker='o', markersize=4, label='Measure Accuracy', zorder=3)
    
    # Plot error rate (loss) in pink
    line_error = ax1.plot(measures, error_rates, linewidth=2, color='#ec4899', 
                          linestyle='--', marker='x', markersize=3, 
                          label='Error Rate (Loss)', alpha=0.8, zorder=3)
    
    # Highlight low accuracy sections (< 50%)
    low_accuracy_threshold = 50.0
    for i, acc in enumerate(measure_accuracies):
        if acc < low_accuracy_threshold:
            ax1.axvspan(i + 0.5, i + 1.5, alpha=0.15, color='red', zorder=1)
    
    # Add average line
    avg_accuracy = np.mean(measure_accuracies)
    ax1.axhline(y=avg_accuracy, color='#10b981', linestyle='--', 
                linewidth=2, label=f'Average: {avg_accuracy:.1f}%', zorder=2)
    
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax1.set_ylim(-5, 105)
    
    # If we have detailed metrics, add per-direction note counts
    if measure_details:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Notes per Measure', fontsize=12, fontweight='bold')
        
        # Extract note counts
        gen_notes = [m['gen_notes'] for m in measure_details]
        orig_notes = [m['orig_notes'] for m in measure_details]
        
        # Plot note density
        line2 = ax2.plot(measures, orig_notes, linewidth=1.5, color='#f59e0b', 
                        linestyle='-.', marker='s', markersize=3, 
                        label='Original Notes', alpha=0.7)
        line3 = ax2.plot(measures, gen_notes, linewidth=1.5, color='#8b5cf6', 
                        linestyle='-.', marker='d', markersize=3, 
                        label='Generated Notes', alpha=0.7)
        
        ax2.tick_params(axis='y')
        max_notes = max(max(gen_notes) if gen_notes else 0, max(orig_notes) if orig_notes else 0)
        ax2.set_ylim(0, max_notes + 5)
        
        # Combine legends
        lines = line1 + line_error + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=9, ncol=2)
    else:
        lines = line1 + line_error
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    # Add title
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add statistics annotation
    std_dev = np.std(measure_accuracies)
    perfect_count = sum(1 for acc in measure_accuracies if acc == 100.0)
    low_count = sum(1 for acc in measure_accuracies if acc < 50.0)
    
    stats_text = (f'Statistics:\n'
                 f'Avg: {avg_accuracy:.1f}%\n'
                 f'Std: {std_dev:.1f}%\n'
                 f'Perfect: {perfect_count}\n'
                 f'Low (<50%): {low_count}\n'
                 f'Total: {len(measure_accuracies)}')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.6)
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', bbox=props, zorder=5)
    
    # Annotate best and worst sections
    if len(measure_accuracies) > 0:
        best_idx = measure_accuracies.index(max(measure_accuracies))
        worst_idx = measure_accuracies.index(min(measure_accuracies))
        
        # Mark best
        ax1.plot(best_idx + 1, measure_accuracies[best_idx], marker='*', 
                color='gold', markersize=20, zorder=10, markeredgecolor='#b45309', markeredgewidth=1.5)
        ax1.annotate(f'Best\n({measure_accuracies[best_idx]:.0f}%)', 
                    xy=(best_idx + 1, measure_accuracies[best_idx]),
                    xytext=(best_idx + 1, measure_accuracies[best_idx] - 15),
                    fontsize=9, color='#b45309', fontweight='bold',
                    ha='center', zorder=10)
        
        # Mark worst if significantly different
        if measure_accuracies[worst_idx] < avg_accuracy - 20:
            ax1.plot(worst_idx + 1, measure_accuracies[worst_idx], marker='v', 
                    color='red', markersize=12, zorder=10)
            ax1.annotate(f'Worst\n({measure_accuracies[worst_idx]:.0f}%)', 
                        xy=(worst_idx + 1, measure_accuracies[worst_idx]),
                        xytext=(worst_idx + 1, max(measure_accuracies[worst_idx] + 15, 15)),
                        fontsize=9, color='#dc2626', fontweight='bold',
                        ha='center', zorder=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def calculate_all_metrics(generated: List[str], original: List[str]) -> Dict[str, any]:
    """
    Calculate all comparison metrics at once.
    
    Args:
        generated: Generated beatmap lines
        original: Original beatmap lines
        
    Returns:
        Comprehensive dictionary with all metrics
    """
    metrics = {}
    
    # Basic accuracy
    metrics.update(calculate_note_accuracy(generated, original))
    
    # Density metrics
    metrics.update(calculate_density_metrics(generated, original))
    
    # Per-direction metrics
    metrics['per_direction'] = calculate_per_direction_metrics(generated, original)
    
    # Confusion matrix
    metrics['confusion_matrix'] = generate_confusion_matrix_data(generated, original)
    
    # Timing alignment
    metrics.update(calculate_timing_alignment(generated, original))
    
    return metrics


if __name__ == "__main__":
    # Hardcoded file paths for easy testing
    gen_file = "generated_springtime_beatmap_20260210_010532.text"  # Latest generated beatmap
    orig_file = "src/musicForBeatmap/Springtime/beatmap_easy.text"   # Original beatmap
    
    print(f"Loading beatmaps...")
    print(f"Generated: {gen_file}")
    print(f"Original: {orig_file}")
    
    generated = load_beatmap(gen_file)
    original = load_beatmap(orig_file)
    
    print(f"\nGenerated: {len(generated)} lines")
    print(f"Original: {len(original)} lines")
    
    print(f"\nCalculating metrics...")
    metrics = calculate_all_metrics(generated, original)
    
    print(f"\n{'='*60}")
    print(f"COMPARISON METRICS")
    print(f"{'='*60}")
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%")
    print(f"Timing Accuracy (±2 lines): {metrics['timing_accuracy']:.2f}%")
    print(f"Density Accuracy: {metrics['density_accuracy']:.2f}%")
    print(f"\nPer-Direction Metrics:")
    for direction, scores in metrics['per_direction'].items():
        print(f"  {direction.capitalize()}: P={scores['precision']:.1f}% R={scores['recall']:.1f}% F1={scores['f1_score']:.1f}%")
    print(f"{'='*60}")
