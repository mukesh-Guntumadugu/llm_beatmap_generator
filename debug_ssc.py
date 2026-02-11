import os

file_path = "src/musicForBeatmap/Springtime/Springtime.ssc"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Loaded {len(lines)} lines.")
in_chart = False
in_notes = False

for i, line in enumerate(lines):
    line = line.strip()
    if line.startswith("#NOTEDATA:"):
        print(f"Line {i}: Found NOTEDATA")
        in_chart = True
        in_notes = False
    
    if in_chart:
        if line.startswith("#NOTES:"):
            print(f"Line {i}: Found NOTES start")
            in_notes = True
        
        if in_notes and ";" in line:
            print(f"Line {i}: Found Terminator (;)")
            # in_notes = False
