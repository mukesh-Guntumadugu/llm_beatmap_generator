#!/usr/bin/env python3
"""
Beatmap Comparison Visualization
Compares human-made beatmap with AI-generated beatmap
Shows accuracy with color coding: Green = Correct, Red = Incorrect
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np

def load_beatmap(filepath):
    """Load beatmap file and parse into measures"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    measures = []
    current_measure = []
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        
        if line == ',' or line == ';':
            if current_measure:
                measures.append(current_measure)
                current_measure = []
        else:
            # Only include lines with exactly 4 characters (note data)
            if len(line) == 4 and all(c in '01234M' for c in line):
                current_measure.append(line)
    
    if current_measure:
        measures.append(current_measure)
    
    return measures

def compare_measures(human_measures, generated_measures):
    """Compare two beatmaps and calculate accuracy"""
    total_lines = 0
    correct_lines = 0
    
    max_measures = min(len(human_measures), len(generated_measures))
    
    for i in range(max_measures):
        human = human_measures[i]
        generated = generated_measures[i]
        
        max_lines = min(len(human), len(generated))
        
        for j in range(max_lines):
            total_lines += 1
            if human[j] == generated[j]:
                correct_lines += 1
    
    accuracy = (correct_lines / total_lines * 100) if total_lines > 0 else 0
    return accuracy, correct_lines, total_lines

def analyze_beatmap_statistics(human_measures, generated_measures):
    """Analyze and compare beatmap statistics"""
    
    def count_notes_in_measures(measures):
        """Count various statistics from measures"""
        stats = {
            'measure_lengths': {},  # e.g., {4: 10, 8: 20, 12: 5, 16: 30}
            'note_counts': {
                '1000': 0,  # Left
                '0100': 0,  # Down
                '0010': 0,  # Up
                '0001': 0,  # Right
            },
            'hold_counts': {
                '2': 0,  # Hold starts
                '3': 0,  # Hold bodies/ends
            },
            'double_notes': 0,  # Two simultaneous notes
            'triple_notes': 0,  # Three simultaneous notes
            'quad_notes': 0,    # Four simultaneous notes
            'total_notes': 0,
            'empty_lines': 0,
        }
        
        for measure in measures:
            # Count measure length
            length = len(measure)
            stats['measure_lengths'][length] = stats['measure_lengths'].get(length, 0) + 1
            
            for line in measure:
                # Count empty vs non-empty lines
                if line == '0000':
                    stats['empty_lines'] += 1
                    continue
                
                # Count notes per line (for doubles/triples/quads)
                notes_in_line = sum(1 for c in line if c in '1234')
                if notes_in_line == 2:
                    stats['double_notes'] += 1
                elif notes_in_line == 3:
                    stats['triple_notes'] += 1
                elif notes_in_line == 4:
                    stats['quad_notes'] += 1
                
                # Count specific note patterns
                if line == '1000':
                    stats['note_counts']['1000'] += 1
                elif line == '0100':
                    stats['note_counts']['0100'] += 1
                elif line == '0010':
                    stats['note_counts']['0010'] += 1
                elif line == '0001':
                    stats['note_counts']['0001'] += 1
                
                # Count holds (2s and 3s in any position)
                for char in line:
                    if char == '2':
                        stats['hold_counts']['2'] += 1
                    elif char == '3':
                        stats['hold_counts']['3'] += 1
                    elif char in '14':
                        stats['total_notes'] += 1
        
        return stats
    
    print("\n" + "="*60)
    print("DETAILED BEATMAP STATISTICS")
    print("="*60)
    
    human_stats = count_notes_in_measures(human_measures)
    gen_stats = count_notes_in_measures(generated_measures)
    
    # Print Measure Lengths Distribution
    print("\n📊 Measure Lengths Distribution:")
    print("-" * 60)
    print(f"{'Lines/Measure':<15} {'Human':<15} {'Generated':<15}")
    print("-" * 60)
    
    all_lengths = sorted(set(list(human_stats['measure_lengths'].keys()) + 
                            list(gen_stats['measure_lengths'].keys())))
    
    for length in all_lengths:
        h_count = human_stats['measure_lengths'].get(length, 0)
        g_count = gen_stats['measure_lengths'].get(length, 0)
        print(f"{length} lines{'':<8} {h_count:<15} {g_count:<15}")
    
    # Print Note Type Distribution
    print("\n🎯 Single Note Distribution:")
    print("-" * 60)
    print(f"{'Note Type':<15} {'Human':<15} {'Generated':<15} {'Match':<10}")
    print("-" * 60)
    
    note_labels = {
        '1000': 'Left (←)',
        '0100': 'Down (↓)',
        '0010': 'Up (↑)',
        '0001': 'Right (→)'
    }
    
    for pattern, label in note_labels.items():
        h_count = human_stats['note_counts'][pattern]
        g_count = gen_stats['note_counts'][pattern]
        match_pct = (min(h_count, g_count) / max(h_count, g_count) * 100) if max(h_count, g_count) > 0 else 100
        print(f"{label:<15} {h_count:<15} {g_count:<15} {match_pct:>6.1f}%")
    
    # Print Hold Statistics
    print("\n⏸️  Hold Note Statistics:")
    print("-" * 60)
    print(f"{'Hold Type':<15} {'Human':<15} {'Generated':<15}")
    print("-" * 60)
    print(f"{'Starts (2)':<15} {human_stats['hold_counts']['2']:<15} {gen_stats['hold_counts']['2']:<15}")
    print(f"{'Bodies (3)':<15} {human_stats['hold_counts']['3']:<15} {gen_stats['hold_counts']['3']:<15}")
    
    # Print Concurrency Statistics
    print("\n🎵 Note Concurrency (Simultaneous Notes):")
    print("-" * 60)
    print(f"{'Type':<15} {'Human':<15} {'Generated':<15}")
    print("-" * 60)
    print(f"{'Doubles':<15} {human_stats['double_notes']:<15} {gen_stats['double_notes']:<15}")
    print(f"{'Triples':<15} {human_stats['triple_notes']:<15} {gen_stats['triple_notes']:<15}")
    print(f"{'Quads':<15} {human_stats['quad_notes']:<15} {gen_stats['quad_notes']:<15}")
    
    # Print Overall Summary
    print("\n📈 Overall Summary:")
    print("-" * 60)
    h_total = human_stats['total_notes'] + human_stats['hold_counts']['2']
    g_total = gen_stats['total_notes'] + gen_stats['hold_counts']['2']
    print(f"{'Total Notes':<15} {h_total:<15} {g_total:<15}")
    print(f"{'Empty Lines':<15} {human_stats['empty_lines']:<15} {gen_stats['empty_lines']:<15}")
    
    print("="*60 + "\n")


def visualize_comparison(human_file, generated_file, start_measure=0, num_measures=8):
    """
    Create visual comparison of beatmaps
    
    Args:
        human_file: Path to human/ground truth beatmap
        generated_file: Path to AI-generated beatmap
        start_measure: Which measure to start from (0-indexed)
        num_measures: How many measures to display
    """
    
    # Load beatmaps
    print(f"Loading human beatmap: {human_file}")
    human_measures = load_beatmap(human_file)
    
    print(f"Loading generated beatmap: {generated_file}")
    generated_measures = load_beatmap(generated_file)
    
    print(f"Human: {len(human_measures)} measures")
    print(f"Generated: {len(generated_measures)} measures")
    
    # Analyze detailed statistics
    analyze_beatmap_statistics(human_measures, generated_measures)
    
    # Calculate overall accuracy
    accuracy, correct, total = compare_measures(human_measures, generated_measures)
    print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{total} lines correct)")
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f'Beatmap Comparison - Overall Accuracy: {accuracy:.1f}%', 
                 fontsize=16, fontweight='bold')
    
    # Arrow symbols for each column (L, D, U, R)
    arrow_map = {
        '0': '',
        '1': '←',
        '2': '↓',
        '3': '↑',
        '4': '→',
        'M': '✕'
    }
    
    colors = {
        '0': '#f0f0f0',  # Empty
        '1': '#FF6B9D',  # Left - Pink
        '2': '#4ECDC4',  # Down - Cyan
        '3': '#95E1D3',  # Up - Light teal
        '4': '#FFE66D',  # Right - Yellow
        'M': '#FF0000'   # Mine - Red
    }
    
    end_measure = min(start_measure + num_measures, 
                     len(human_measures), 
                     len(generated_measures))
    
    for ax_idx, (ax, title) in enumerate(zip(axes, 
                                             ['Human Choreography (Ground Truth)',
                                              'Prediction Accuracy (Green=Correct, Red=Wrong)',
                                              'Generated Choreography (AI Model)'])):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(0, num_measures * 4 + num_measures)  # 4 columns + spacing
        ax.set_ylim(0, 20)
        ax.axis('off')
        
        x_offset = 0
        
        for m_idx in range(start_measure, end_measure):
            human_measure = human_measures[m_idx]
            gen_measure = generated_measures[m_idx]
            
            max_lines = max(len(human_measure), len(gen_measure))
            
            # Draw measure separator
            if m_idx > start_measure:
                ax.axvline(x=x_offset - 0.3, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            for line_idx in range(max_lines):
                y_pos = 19 - line_idx  # Start from top
                
                # Get note data
                human_line = human_measure[line_idx] if line_idx < len(human_measure) else '0000'
                gen_line = gen_measure[line_idx] if line_idx < len(gen_measure) else '0000'
                
                for col_idx in range(4):
                    x_pos = x_offset + col_idx
                    
                    if ax_idx == 0:  # Human beatmap
                        note = human_line[col_idx]
                        color = colors.get(note, '#f0f0f0')
                        rect = Rectangle((x_pos, y_pos), 0.9, 0.9, 
                                       facecolor=color, edgecolor='black', linewidth=1)
                        ax.add_patch(rect)
                        
                        if note != '0':
                            ax.text(x_pos + 0.45, y_pos + 0.45, 
                                   arrow_map.get(note, note),
                                   ha='center', va='center', 
                                   fontsize=14, fontweight='bold',
                                   color='#2d2d2d')
                    
                    elif ax_idx == 1:  # Comparison
                        human_note = human_line[col_idx]
                        gen_note = gen_line[col_idx]
                        
                        is_correct = (human_note == gen_note)
                        
                        # Color code: Green if correct, Red if wrong
                        if is_correct:
                            color = '#00ff00' if human_note != '0' else '#e8e8e8'
                            edge_color = '#00aa00' if human_note != '0' else '#cccccc'
                        else:
                            color = '#ff0000'
                            edge_color = '#aa0000'
                        
                        rect = Rectangle((x_pos, y_pos), 0.9, 0.9,
                                       facecolor=color, edgecolor=edge_color, 
                                       linewidth=2 if not is_correct else 1)
                        ax.add_patch(rect)
                        
                        # Show what was generated (if wrong)
                        if not is_correct:
                            ax.text(x_pos + 0.45, y_pos + 0.45,
                                   arrow_map.get(gen_note, gen_note),
                                   ha='center', va='center',
                                   fontsize=12, fontweight='bold',
                                   color='white')
                    
                    elif ax_idx == 2:  # Generated beatmap
                        note = gen_line[col_idx]
                        color = colors.get(note, '#f0f0f0')
                        rect = Rectangle((x_pos, y_pos), 0.9, 0.9,
                                       facecolor=color, edgecolor='black', linewidth=1)
                        ax.add_patch(rect)
                        
                        if note != '0':
                            ax.text(x_pos + 0.45, y_pos + 0.45,
                                   arrow_map.get(note, note),
                                   ha='center', va='center',
                                   fontsize=14, fontweight='bold',
                                   color='#2d2d2d')
            
            x_offset += 5  # 4 columns + spacing
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='#00ff00', edgecolor='#00aa00', label='Correct Prediction'),
        patches.Patch(facecolor='#ff0000', edgecolor='#aa0000', label='Incorrect Prediction'),
        patches.Patch(facecolor='#e8e8e8', edgecolor='#cccccc', label='Correct Empty')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11)
    
    plt.tight_layout()
    
    # Save
    output_file = f"beatmap_comparison_measures_{start_measure}-{end_measure-1}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    # Configuration
    HUMAN_BEATMAP = "src/musicForBeatmap/Springtime/beatmap_easy.text"
    GENERATED_BEATMAP = "generated_springtime_beatmap_20260208_181657.text"
    
    # You can change these values
    START_MEASURE = 0  # Start from measure 5 (where notes begin)
    NUM_MEASURES = 5   # Show 8 measures
    
    print("="*60)
    print("BEATMAP COMPARISON VISUALIZATION")
    print("="*60)
    
    if len(sys.argv) >= 3:
        HUMAN_BEATMAP = sys.argv[1]
        GENERATED_BEATMAP = sys.argv[2]
        if len(sys.argv) >= 4:
            START_MEASURE = int(sys.argv[3])
        if len(sys.argv) >= 5:
            NUM_MEASURES = int(sys.argv[4])
    
    visualize_comparison(HUMAN_BEATMAP, GENERATED_BEATMAP, START_MEASURE, NUM_MEASURES)
