import os
import re

def parse_ssc_easy(file_path):
    """Parses the 'Easy' difficulty chart from an .ssc file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all NOTEDATA sections
    notedata_sections = re.split(r'#NOTEDATA:;', content)
    
    easy_notes = []
    
    for section in notedata_sections:
        if "#DIFFICULTY:Easy;" in section or "#DIFFICULTY:Easy" in section: # varied formatting
             # Extract #NOTES tag content
            match = re.search(r'#NOTES:\s*([^;]+);', section, re.DOTALL)
            if match:
                raw_notes = match.group(1).strip()
                easy_notes = parse_chart_data(raw_notes)
                return easy_notes
    
    if not easy_notes:
        print("[WARNING] 'Easy' difficulty not found in .ssc file.")
        
    return easy_notes

def parse_generated_file(file_path):
    """Parses the generated beatmap file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove single-line comments
    content = re.sub(r'//.*', '', content)
    return parse_chart_data(content)

def parse_chart_data(raw_data):
    """
    Parses raw chart data (measures separated by commas) into a list of events.
    Returns: List of tuples (beat, column, note_type)
    """
    measures = raw_data.split(',')
    events = []
    
    for measure_idx, measure_content in enumerate(measures):
        lines = measure_content.strip().split()
        if not lines:
            continue
            
        num_lines = len(lines)
        for line_idx, line in enumerate(lines):
            # Calculate beat position
            # 4 beats per measure. line_idx / num_lines gives fraction of measure.
            beat = measure_idx * 4 + (line_idx / num_lines) * 4
            
            for col, char in enumerate(line):
                if char != '0':
                    events.append((beat, col, char))
                    
    return events

def compare_beatmaps(original, generated):
    """Compares two lists of beatmap events."""
    original_set = set(original)
    generated_set = set(generated)
    
    common = original_set.intersection(generated_set)
    only_in_original = original_set - generated_set
    only_in_generated = generated_set - original_set
    
    precision = len(common) / len(generated_set) if generated_set else 0
    recall = len(common) / len(original_set) if original_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    print(f"Total Notes in Original (Easy): {len(original_set)}")
    print(f"Total Notes in Generated:       {len(generated_set)}")
    print(f"Exact Matches:                  {len(common)}")
    print(f"Missed Notes (FN):              {len(only_in_original)}")
    print(f"Extra Notes (FP):               {len(only_in_generated)}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Sort events to find first notes
    orig_sorted = sorted(list(original_set))
    gen_sorted = sorted(list(generated_set))
    
    if orig_sorted and gen_sorted:
        print("\n--- Timing Analysis ---")
        print(f"First Note Time - Original:  Beat {orig_sorted[0][0]:.3f} (Col {orig_sorted[0][1]})")
        print(f"First Note Time - Generated: Beat {gen_sorted[0][0]:.3f} (Col {gen_sorted[0][1]})")
        diff = gen_sorted[0][0] - orig_sorted[0][0]
        print(f"Start Offset Diff:           {diff:.3f} beats")
        
        # Check for matching patterns despite offset?
        # (Naive check: check if shifting by offset helps)
        shifted_matches = 0
        for note in generated_set:
            shifted_note = (note[0] - diff, note[1], note[2])
            if shifted_note in original_set:
                shifted_matches += 1
        print(f"Matches if shifted by start diff: {shifted_matches}")

    if only_in_generated:
        print("\n[Sample Extra Notes in Generated] (Beat, Col, Type)")
        for item in sorted(list(only_in_generated))[:10]:
            print(f"  {item}")

    if only_in_original:
        print("\n[Sample Missed Notes from Original] (Beat, Col, Type)")
        for item in sorted(list(only_in_original))[:10]:
            print(f"  {item}")

if __name__ == "__main__":
    original_path = "../src/musicForBeatmap/Springtime/Springtime.ssc"
    generated_path = "../src/beatmap_generated/generated_chart_beatmap_Kommisar - Springtime_20260126140345.txt"
    
    print(f"Loading Original: {original_path}")
    original_events = parse_ssc_easy(original_path)
    
    print(f"Loading Generated: {generated_path}")
    generated_events = parse_generated_file(generated_path)
    
    if not original_events:
        print("Error: Could not load original events.")
    elif not generated_events:
        print("Error: Could not load generated events.")
    else:
        print("\n--- Comparison Results ---")
        compare_beatmaps(original_events, generated_events)
