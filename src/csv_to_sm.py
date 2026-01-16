
import csv
import argparse
import os
import re
import math

def parse_sm_metadata(sm_file):
    """
    Parses the reference .sm file to get metadata like BPMS and OFFSET.
    Returns the full header/content of the file up to #NOTES.
    """
    metadata = {}
    header_lines = []
    
    with open(sm_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        if line.strip().startswith("//") and "dance-single" in line:
            break
        if line.strip().startswith("#NOTES:"):
            break
            
        header_lines.append(line)
        
        # Extract Offset
        if line.strip().startswith("#OFFSET:"):
            try:
                val = line.split(":")[1].split(";")[0].strip()
                metadata['offset'] = float(val)
            except ValueError:
                pass
                
        # Extract BPMs (Assume simplistic single BPM for now)
        if line.strip().startswith("#BPMS:"):
            try:
                # Format: 0.000=150.000
                val = line.split(":")[1].split(";")[0].strip()
                # Split by comma if multiple
                first_bpm = val.split(",")[0]
                bpm_val = first_bpm.split("=")[1]
                metadata['bpm'] = float(bpm_val)
            except (ValueError, IndexError):
                pass

    return header_lines, metadata

def get_quantization_idx(beat_fraction):
    """
    Returns the quantization index (row in a measure) for a given beat fraction.
    We use 192 rows per measure (standard max resolution for 4/4).
    Returns None if it doesn't align reasonably.
    """
    # Standard resolution: 192 subdivisions per measure (48 per beat)
    # 4 beats * 48 = 192
    
    # beat_fraction is 0.0 to 4.0 (float) relative to measure start
    # row = round(beat_fraction * 48)
    
    row = round(beat_fraction * 48)
    return row

def generate_notes_string(notes_data, metadata):
    """
    Converts timestamped notes into SM measure strings.
    """
    bpm = metadata.get('bpm', 120.0) # Default 120 if missing
    offset = metadata.get('offset', 0.0)
    
    print(f"Using BPM: {bpm}, Offset: {offset}")
    
    # 1. Convert notes to Beat Indices
    # Beat = (Time - Offset) * (BPM / 60)
    beat_notes = []
    for note in notes_data:
        t = float(note['time'])
        direction = note['note'].lower()
        
        beat_idx = (t - offset) * (bpm / 60.0)
        
        # Map direction
        col = -1
        if 'left' in direction: col = 0
        elif 'down' in direction: col = 1
        elif 'up' in direction: col = 2
        elif 'right' in direction: col = 3
        
        if col != -1:
            beat_notes.append({'beat': beat_idx, 'col': col})
            
    if not beat_notes:
        return ""

    # Sort by beat
    beat_notes.sort(key=lambda x: x['beat'])
    
    max_beat = beat_notes[-1]['beat']
    total_measures = math.ceil(max_beat / 4.0)
    
    # 2. Build Grid
    # We will use 192 rows per measure (48 per beat) to support 1/64 notes
    measure_grid = {} # measure_idx -> {row_idx -> [0,0,0,0]}
    
    for item in beat_notes:
        abs_beat = item['beat']
        measure_idx = int(abs_beat // 4)
        remainder = abs_beat % 4
        
        row_idx =  int(round(remainder * 48)) # 0 to 191
        
        if measure_idx not in measure_grid:
            measure_grid[measure_idx] = {}
            
        if row_idx not in measure_grid[measure_idx]:
             measure_grid[measure_idx][row_idx] = ['0','0','0','0']
             
        # Set note (1 = Tap note)
        measure_grid[measure_idx][row_idx][item['col']] = '1'
        
    # 3. Serialize to String
    sm_output = []
    
    # We need to fill up to total_measures
    for m in range(total_measures + 1):
        if m in measure_grid:
            rows = measure_grid[m]
            
            # Optimization: Determine simplification level for this measure?
            # For now, we will output standard 16th notes (wait, if we use 192 lines it's huge)
            # Let's try to simplify. 
            # Actually, standard SM usually outputs 4th, 8th, 12th, 16th. 
            # Let's iterate through 192 rows. If all zero, skip? 
            # No, StepMania expects consistent lines per measure (e.g. 4, 8, 16, 192 lines).
            # The number of lines determines the resolution FOR THAT MEASURE.
            # So check the GCD of active rows.
            
            active_rows = sorted(rows.keys())
            
            # Calculate necessary resolution
            # Standard quantizations (lines per measure): 4, 8, 12, 16, 24, 32, 48, 64, 96, 192
            # row_idx is in 0..191 range.
            # If all row_indices are divisible by 48 -> 4 lines (4ths)
            # If divisible by 24 -> 8 lines (8ths)
            # If divisible by 12 -> 16 lines (16ths)
            
            required_res = 192
            if active_rows:
                # Find greatest capability divisor
                # We need a resolution (res) such that for all r in active_rows:
                # (r * res) / 192 is an integer.
                # Equivalent to checking divisibility of 192/res?
                
                # Check 4ths (step 48)
                if all(r % 48 == 0 for r in active_rows): required_res = 4
                elif all(r % 24 == 0 for r in active_rows): required_res = 8
                elif all(r % 16 == 0 for r in active_rows): required_res = 12
                elif all(r % 12 == 0 for r in active_rows): required_res = 16
                elif all(r % 8 == 0 for r in active_rows): required_res = 24
                elif all(r % 6 == 0 for r in active_rows): required_res = 32
                elif all(r % 4 == 0 for r in active_rows): required_res = 48
                elif all(r % 3 == 0 for r in active_rows): required_res = 64
                
            
            measure_str = []
            step = 192 // required_res
            
            for i in range(required_res):
                tick = i * step
                line = rows.get(tick, ['0','0','0','0'])
                measure_str.append("".join(line))
                
            sm_output.append("\n".join(measure_str))
            
        else:
            # Empty measure
            sm_output.append("0000\n0000\n0000\n0000")
            
    return "\n,\n".join(sm_output)

def main():
    parser = argparse.ArgumentParser(description="Convert CSV beatmap to .sm file")
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument("sm_file", help="Reference .sm file (for metadata)")
    parser.add_argument("--output", help="Output .sm file name", default=None)
    
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.csv_file.replace(".csv", ".sm")
        
    print(f"Reading CSV: {args.csv_file}")
    notes_data = []
    with open(args.csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            notes_data.append(row)
            
    print(f"Reading SM Template: {args.sm_file}")
    header_lines, metadata = parse_sm_metadata(args.sm_file)
    
    if 'bpm' not in metadata or 'offset' not in metadata:
        print("Warning: Could not parse #BPMS or #OFFSET from SM file. Using defaults (120 BPM, 0.0 Offset).")
    
    note_data_str = generate_notes_string(notes_data, metadata)
    
    # Construct final file
    # We append the new charts. For now, let's create a 'Challenge' difficulty entry.
    
    chart_block = """
//---------------dance-single - ----------------
#NOTES:
     dance-single:
     :
     Challenge:
     10:
     0.0,0.0,0.0,0.0,0.0:
""" 
    chart_block += note_data_str + ";\n"
    
    final_content = "".join(header_lines) + chart_block
    
    print(f"Writing to: {args.output}")
    with open(args.output, 'w') as f:
        f.write(final_content)
        
    print("Conversion Complete.")

if __name__ == "__main__":
    main()
