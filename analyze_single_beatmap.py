import os

"""
(.venv1) mukeshguntumadugu@Mac llm_beatmap_generator % /Users/mukeshguntumadugu/llm_beatmap_generator/.venv1/bin/python /Users/mukeshguntumadugu/llm_beatmap_generator/an
alyze_single_beatmap.py
--- Single Beatmap Analyzer ---
Enter the file path to analyze:  /Users/mukeshguntumadugu/llm_beatmap_generator/src/Goin' Under_20260204_123041_Easy_full.csv


"""


def analyze_single_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print("\n" + "="*40)
    print(f"ANALYZING: {os.path.basename(file_path)}")
    print("="*40)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_rows = 0
    empty_rows = 0  # "0000"
    filled_rows = 0
    
    # Concurrency Stats
    singles = 0
    doubles = 0
    triples = 0
    quads = 0
    more_than_quad = 0
    
    # Store Locations of heavy notes for inspection
    quad_locations = []
    
    col_counts = [0, 0, 0, 0] # L, D, U, R (assuming 4 cols)
    total_zeros = 0 # Individual '0' characters
    
    # Heuristic to detect valid note lines: length >= 4, mostly digits
    valid_lines = []
    
    # We need index relative to file for debugging
    for idx, line in enumerate(lines):
        line = line.strip()
        # Clean inline comments or semicolons
        clean_line = line.split("//")[0].replace(";", "").replace(",", "").strip()
        
        if not clean_line: continue
        
        # Check if it looks like a note row (e.g. "0000", "1001", "01000")
        # Must be digits
        if not all(c in '01234M' for c in clean_line):
            continue
            
        # If length is wildly different, skip (unless it's 4 or 5 panels)
        if len(clean_line) < 4: continue
        
        valid_lines.append(clean_line)
        
        total_rows += 1
        
        # Check for Empty Row
        if all(c == '0' for c in clean_line):
            empty_rows += 1
        else:
            filled_rows += 1
            # Count concurrency (how many notes active at once)
            active_notes = sum(1 for c in clean_line if c != '0')
            if active_notes == 1: singles += 1
            elif active_notes == 2: doubles += 1
            elif active_notes == 3: triples += 1
            elif active_notes == 4: 
                quads += 1
                quad_locations.append((idx + 1, clean_line))
            elif active_notes > 4: more_than_quad += 1
            
        # Count Zeros and Notes
        for i, char in enumerate(clean_line):
            if char == '0':
                total_zeros += 1
            else:
                # It's a note (1, 2, 3, M, etc)
                if i < 4: # Standard 4 columns
                    col_counts[i] += 1
                # If 5 columns, we could track it, but let's stick to standard summary
    
    # --- OUTPUT ---
    print(f"Total Note Rows Scanned: {total_rows}")
    print("-" * 30)
    
    if total_rows > 0:
        empty_pct = (empty_rows / total_rows) * 100
        filled_pct = (filled_rows / total_rows) * 100
        
        print(f"Empty Rows ('0000'):   {empty_rows:<6} ({empty_pct:.1f}%)")
        print(f"Filled Rows (Notes):   {filled_rows:<6} ({filled_pct:.1f}%)")
        print("-" * 30)
        
        print(f"Total Zero Characters: {total_zeros}")
        print("-" * 30)
        
        print("Note Concurrency (Chords/Jumps):")
        print(f"  Singles (1 note):  {singles}")
        print(f"  Doubles (2 notes): {doubles}")
        print(f"  Triples (3 notes): {triples}")
        print(f"  Quads   (4 notes): {quads}")
        if more_than_quad > 0:
            print(f"  5+ Notes:          {more_than_quad}")
        print("-" * 30)
        
        if quads > 0:
            print(f"  [!] LIST OF QUADS (Lines where 4 notes appear):")
            for loc in quad_locations[:20]: # Show first 20
                print(f"      Line {loc[0]}: {loc[1]}")
            if len(quad_locations) > 20:
                print(f"      ... and {len(quad_locations) - 20} more.")
            print("-" * 30)
        
        print("Column Distribution (Notes):")
        labels = ["Left", "Down", "Up", "Right"]
        for i in range(4):
            count = col_counts[i]
            pct = count / sum(col_counts) * 100 if sum(col_counts) > 0 else 0
            print(f"  {labels[i]}: {count} ({pct:.1f}%)")
            
    else:
        print("No valid note rows found.")
        
    print("="*40 + "\n")

if __name__ == "__main__":
    print("--- Single Beatmap Analyzer ---")
    raw_input = input("Enter the file path to analyze: ").strip()
    # Remove surrounding quotes only (drag-and-drop artifact)
    # Do NOT use .replace() as it breaks filenames like "Goin' Under"
    if len(raw_input) >= 2 and raw_input.startswith("'") and raw_input.endswith("'"):
        raw_input = raw_input[1:-1]
    elif len(raw_input) >= 2 and raw_input.startswith('"') and raw_input.endswith('"'):
        raw_input = raw_input[1:-1]
        
    user_file = raw_input
    analyze_single_file(user_file)
