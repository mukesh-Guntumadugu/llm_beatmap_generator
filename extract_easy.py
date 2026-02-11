
import os

ssc_path = "src/musicForBeatmap/Springtime/Springtime.ssc"
output_path = "src/musicForBeatmap/Springtime/beatmap_easy.text"

start_line = 4744 # 1-based in view_file, so 0-based index is 4743
end_line = 5495   # inclusive

print(f"Reading from {ssc_path}...")
with open(ssc_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract chunk
# Python lists are 0-indexed. Line 4744 is index 4743.
notes_chunk = lines[start_line-1 : end_line]

processed_lines = []
for line in notes_chunk:
    line = line.strip()
    if not line or line.startswith("//"):
        continue
    
    # Remove semicolon if present (at the end of the last line)
    if line.endswith(';'):
        line = line[:-1]
    
    # validation: must be 4 char (0000) or comma
    if line == ',' or len(line) == 4:
         processed_lines.append(line)
    elif len(line) > 4: # Handling potentially 8-key or other weirdness, but 'Easy' should be 4
        # dance-single is 4 keys. If there are other things, we might need to filter.
        # But looking at the file, it looks like standard 4-key.
        pass

print(f"Extracted {len(processed_lines)} valid lines.")

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(processed_lines))

print(f"Wrote to {output_path}")
