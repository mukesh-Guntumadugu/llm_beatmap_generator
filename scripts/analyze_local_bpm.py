#!/usr/bin/env python3
"""
analyze_local_bpm.py
====================
Extracts detailed tempo change events and identifies local BPM for specific sections.
"""

import os
import glob
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "musicForBeatmap")
OUTPUT_REPORT = os.path.join(os.path.dirname(__file__), "..", "local_bpm_analysis.json")

def find_beatmap_files(directory):
    files = []
    for root, _, _ in os.walk(directory):
        sm_files = glob.glob(os.path.join(root, "*.sm"))
        ssc_files = glob.glob(os.path.join(root, "*.ssc"))
        basenames = set()
        for f in sm_files:
            files.append(f)
            basenames.add(os.path.splitext(os.path.basename(f))[0])
        for f in ssc_files:
            base = os.path.splitext(os.path.basename(f))[0]
            if base not in basenames:
                files.append(f)
    return files

def parse_bpms(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().replace('\r\n', '\n').replace('\r', '\n')
    blocks = content.split(';')
    bpms = []
    for block in blocks:
        block = block.strip()
        if block.startswith('#BPMS:'):
            data = block.split(':', 1)[1].strip()
            if data:
                for pair in data.split(','):
                    if '=' in pair:
                        beat, bpm = pair.split('=', 1)
                        try:
                            bpms.append((float(beat.strip()), float(bpm.strip())))
                        except ValueError: pass
    bpms.sort(key=lambda x: x[0])
    return bpms

def get_sections(bpms):
    sections = []
    if not bpms: return sections
    for i in range(len(bpms)):
        sections.append({
            "start_beat": bpms[i][0],
            "end_beat": bpms[i+1][0] if i+1 < len(bpms) else None,
            "bpm": bpms[i][1]
        })
    return sections

def main():
    files = find_beatmap_files(DATA_DIR)
    report = {}
    print(f"Analyzing local BPMs for {len(files)} songs...")
    for file in files:
        basename = os.path.basename(file)
        bpms = parse_bpms(file)
        sections = get_sections(bpms)
        if len(sections) > 0:
            report[basename] = {
                "file": file, "total_changes": max(0, len(sections) - 1),
                "sections": sections
            }
    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Finished! Found {len(report)} songs with tempo changes.")
    print(f"Detailed report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
