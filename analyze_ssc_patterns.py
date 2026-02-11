import os


"""
(.venv1) mukeshguntumadugu@Mac llm_beatmap_generator % /Users/mukeshguntumadugu/llm_beatmap_generator/.venv1/bin/python /Users/mukeshguntumadugu/llm_beatmap_generator/an
alyze_ssc_patterns.py

--- Beatmap Verification ---
Enter the GENERATED .text file path (to validate): generated_springtime_beatmap_20260204_120503.text
Enter the TARGET/TRUTH .text/.ssc file path (to compare against): /Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/Springtime/Springtime.ssc

"""


def analyze_patterns(file_path):
    # Determine file type
    is_ssc = file_path.lower().endswith('.ssc')
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print("\n" + "="*40)
    print(f"FILE: {os.path.basename(file_path)}")
    print("="*40)

    # Initialize defaults for return values
    col_counts = [0, 0, 0, 0]
    total_notes = 0
    chart_name = "Raw Text"

    if is_ssc:
        # SSC: Scan and print EACH chart found
        in_chart = False
        in_notes = False
        temp_difficulty = "Unknown"
        
        temp_counts = [0, 0, 0, 0]
        temp_total = 0
        
        charts_found = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith("#NOTEDATA:"):
                in_chart = True
                in_notes = False
                temp_counts = [0, 0, 0, 0]
                temp_total = 0
                temp_difficulty = "Unknown"
                continue
                
            if in_chart:
                if line.startswith("#DIFFICULTY:"):
                    temp_difficulty = line.replace("#DIFFICULTY:", "").replace(";", "").strip()
                    
                if line.startswith("#NOTES:"):
                    in_notes = True
                    continue
                    
                if in_notes:
                    # Check for end of chart (semicolon)
                    has_semicolon = ";" in line
                    note_line = line.replace(";", "").strip()
                    
                    # 1. Parse Note if valid
                    # Ignore comments/empty, but allow 0000;
                    is_valid_note = False
                    if len(note_line) >= 4 and not note_line.startswith("//") and not note_line.startswith(","):
                        # Check validity (digits)
                        row_dat = note_line.split("//")[0].strip() # Remove inline comments
                        if all(c in '01234M' for c in row_dat):
                            is_valid_note = True
                            # Resize temp_counts if needed (e.g. 5 columns)
                            if len(row_dat) > len(temp_counts):
                                temp_counts.extend([0] * (len(row_dat) - len(temp_counts)))
                            
                            for col_idx, char in enumerate(row_dat):
                                if col_idx < len(temp_counts) and char != '0':
                                    temp_counts[col_idx] += 1
                            temp_total = sum(temp_counts)

                    # 2. Close Chart if semicolon found
                    if has_semicolon or line.startswith(";"):
                        in_notes = False
                        in_chart = False
                        
                        charts_found += 1
                        print(f"\n--- Chart: {temp_difficulty} ---")
                        print(f"Total Notes: {temp_total}")
                        
                        # Dynamic Labels
                        default_labels = ["Left", "Down", "Up", "Right"]
                        for idx, count in enumerate(temp_counts):
                            label = default_labels[idx] if idx < len(default_labels) else f"Col {idx}"
                            pct = (count / temp_total * 100) if temp_total > 0 else 0
                            print(f"{label}: {count} ({pct:.1f}%)")
                            
                        col_counts = temp_counts
                        total_notes = temp_total
                        chart_name = temp_difficulty
        
        if charts_found == 0:
            print(f"No Valid Charts Found in SSC ({os.path.basename(file_path)}).")
            # DEBUG: Print why
            # Check if #NOTES detected
            has_notes_tag = any(l.strip().startswith("#NOTES:") for l in lines)
            print(f"DEBUG: Found #NOTES tag in file? {has_notes_tag}")
            has_notedata = any(l.strip().startswith("#NOTEDATA:") for l in lines)
            print(f"DEBUG: Found #NOTEDATA tag in file? {has_notedata}")


    else:
        # Raw Text Mode
        chart_name = "Generated/Text"
        for line in lines:
            line = line.strip()
            if len(line) < 4: continue
            if line.startswith(",") or line.startswith("//"): continue
            
            row_dat = line[:4]
            if not all(c in '01234M' for c in row_dat): continue
            
            for col_idx, char in enumerate(row_dat):
                if char != '0':
                    col_counts[col_idx] += 1
        total_notes = sum(col_counts)
        
        print(f"\n--- Chart: {chart_name} ---")
        print(f"Total Notes: {total_notes}")
        labels = ["Left", "Down", "Up", "Right"]
        for idx, count in enumerate(col_counts):
            pct = (count / total_notes * 100) if total_notes > 0 else 0
            print(f"{labels[idx]}: {count} ({pct:.1f}%)")

    # Return LAST stats found (mostly useful for text files or single comparison)
    return {
        "file": os.path.basename(file_path),
        "chart": chart_name,
        "total": total_notes,
        "counts": col_counts,
        "labels": ["Left", "Down", "Up", "Right"]
    }

def compare_files(file_path_1, file_path_2):
    print("\n" + "#"*40)
    print(f"COMPARING:\n A: {os.path.basename(file_path_1)}\n B: {os.path.basename(file_path_2)}")
    print("#"*40)
    
    stats_a = analyze_patterns(file_path_1)
    stats_b = analyze_patterns(file_path_2)
    
    if not stats_a or not stats_b:
        print("Comparison failed due to read errors.")
        return

    print(f"\n{'METRIC':<15} | {'FILE A':<15} | {'FILE B':<15} | {'DIFF (A-B)':<15}")
    print("-" * 70)
    
    # Total
    diff_total = stats_a['total'] - stats_b['total']
    print(f"{'Total Notes':<15} | {stats_a['total']:<15} | {stats_b['total']:<15} | {diff_total:<15}")
    
    # Columns
    labels = stats_a['labels']
    for i, label in enumerate(labels):
        val_a = stats_a['counts'][i]
        val_b = stats_b['counts'][i]
        diff = val_a - val_b
        print(f"{label + ' (Col '+str(i)+')':<15} | {val_a:<15} | {val_b:<15} | {diff:<15}")
    print("-" * 70)

if __name__ == "__main__":
    # Interactive Mode as requested
    print("\n--- Beatmap Verification ---")
    
    # 1. Ask for Generated File
    file_path_1 = input("Enter the GENERATED .text file path (to validate): ").strip().replace("'", "").replace('"', "")
    
    # 2. Ask for Target File
    file_path_2 = input("Enter the TARGET/TRUTH .text/.ssc file path (to compare against): ").strip().replace("'", "").replace('"', "")
    
    if os.path.exists(file_path_1) and os.path.exists(file_path_2):
        compare_files(file_path_1, file_path_2)
    else:
        if not os.path.exists(file_path_1): print(f"Error: Generated file '{file_path_1}' not found.")
        if not os.path.exists(file_path_2): print(f"Error: Target file '{file_path_2}' not found.")

    # 3. Full Directory Scan (As requested: "After that... print the SSC... all difficulties")
    print("\n" + "#"*40)
    print("FULL DIRECTORY SCAN (SSC FILES)")
    print("#"*40)
    
    base_dir = "src/musicForBeatmap"
    if not os.path.exists(base_dir): base_dir = "." # Fallback
    
    ssc_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".ssc"):
                # Avoid duplicates if we already analyzed? No, just print everything.
                ssc_files.append(os.path.join(root, file))
    
    ssc_files.sort()
    
    if not ssc_files:
        print("No SSC files found in scan.")
    else:
        for ssc in ssc_files:
            analyze_patterns(ssc)
