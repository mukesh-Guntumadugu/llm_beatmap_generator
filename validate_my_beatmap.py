#!/usr/bin/env python3
"""
Smart Beatmap Validator Script

This script helps you check the quality of your AI-generated beatmap.
You can manually configure paths below, or let it auto-detect from .ssc files.
"""

import sys
import os
import re
import glob

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from analysis.beatmap_validator import (
    validate_beatmap,
    compare_beatmaps,
    print_validation_report,
    print_comparison_report,
    save_results_json,
    visualize_validation
)

# ============================================================================
# USER CONFIGURATION (EDIT THIS SECTION)
# ============================================================================

# 1. GENERATED Beatmap  (The .txt file you made)
GENERATED_BEATMAP_PATH = "/Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup_Bad Ketchup_20260218_012707_Easy_gemini-pro-latest.txt"

# 2. ORIGINAL Metadata / Beatmap (The .ssc or .sm file)
# Used to get BPM, Offset, and correct difficulty data.
# If you don't have one, set to None.
ORIGINAL_METADATA_PATH = "/Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ssc"

# 3. Audio File (Set to None to auto-find from same folder, or set full path to override)
AUDIO_FILE_PATH = None  # e.g. "/path/to/song.ogg" — auto-detected from folder if None

# 4. Manual Settings (Optional - leave None to read from Metadata file)
MANUAL_BPM = None     # e.g. 145.0
MANUAL_OFFSET = None  # e.g. 0.02

# ============================================================================
# END CONFIGURATION
# ============================================================================


# Supported audio extensions (auto-detection)
AUDIO_EXTENSIONS = ['.ogg', '.mp3', '.mp4', '.wav', '.flac', '.aac', '.m4a', '.opus']


def find_audio_in_folder(folder):
    """
    Scans folder for any audio file and returns the first match.
    Checks all common audio extensions.
    """
    for ext in AUDIO_EXTENSIONS:
        matches = glob.glob(os.path.join(folder, f"*{ext}"))
        if matches:
            return matches[0]
    return None


def find_metadata_and_audio(beatmap_path):
    """
    Looks for a .ssc file in the same folder to find BPM, Offset and Audio.
    Returns: audio_path, ssc_path, bpm, offset
    """
    # This function is kept for backward compatibility if needed, 
    # but main logic now handles explicit paths.
    folder = os.path.dirname(beatmap_path)
    ssc_files = glob.glob(os.path.join(folder, "*.ssc"))
    
    if not ssc_files:
        return None, None, 120.0, 0.0
        
    ssc_path = ssc_files[0]
    bpm = 120.0
    offset = 0.0
    audio_filename = None
    
    with open(ssc_path, 'r', errors='ignore') as f:
        content = f.read()
        bpm_match = re.search(r"#BPMS:.*?=(\d+\.?\d*);", content, re.DOTALL)
        if bpm_match: bpm = float(bpm_match.group(1))
        offset_match = re.search(r"#OFFSET:(-?\d+\.?\d*);", content)
        if offset_match: offset = float(offset_match.group(1))
        music_match = re.search(r"#MUSIC:(.*?);", content)
        if music_match: audio_filename = music_match.group(1).strip()
            
    audio_path = None
    if audio_filename:
        potential_path = os.path.join(folder, audio_filename)
        if os.path.exists(potential_path): audio_path = potential_path
            
    if not audio_path:
        audio_files = glob.glob(os.path.join(folder, "*.ogg")) + glob.glob(os.path.join(folder, "*.mp3"))
        if audio_files: audio_path = audio_files[0]
            
    return audio_path, ssc_path, bpm, offset


def main():
    print("\n" + "="*60)
    print(" SMART BEATMAP VALIDATOR (Manual Config Mode)")
    print("="*60)
    
    # 1. Check Generated File
    if not GENERATED_BEATMAP_PATH or not os.path.exists(GENERATED_BEATMAP_PATH):
        print(f"\nError: Generated Beatmap file not found:\n{GENERATED_BEATMAP_PATH}")
        return

    # 2. Get Metadata (BPM, Offset, Audio)
    # Default values
    bpm = 120.0
    offset = 0.0
    audio_path = None
    
    # A. Try reading from Original Metadata (.ssc) if provided
    if ORIGINAL_METADATA_PATH and os.path.exists(ORIGINAL_METADATA_PATH):
        print(f"Reading Metadata from: {os.path.basename(ORIGINAL_METADATA_PATH)}")
        try:
            with open(ORIGINAL_METADATA_PATH, 'r', errors='ignore') as f:
                content = f.read()
                
                # Find BPM
                bpm_match = re.search(r"#BPMS:.*?=(\d+\.?\d*);", content, re.DOTALL)
                if bpm_match: bpm = float(bpm_match.group(1))
                
                # Find Offset
                offset_match = re.search(r"#OFFSET:(-?\d+\.?\d*);", content)
                if offset_match: offset = float(offset_match.group(1))
                
                # Find Audio filename
                music_match = re.search(r"#MUSIC:(.*?);", content)
                if music_match:
                    audio_filename = music_match.group(1).strip()
                    # Look for audio in same folder as .ssc
                    ssc_dir = os.path.dirname(ORIGINAL_METADATA_PATH)
                    potential_audio = os.path.join(ssc_dir, audio_filename)
                    if os.path.exists(potential_audio):
                        audio_path = potential_audio
        except Exception as e:
            print(f"Warning reading metadata: {e}")
    else:
        print("No Original Metadata file provided (or file not found). Using defaults.")

    # B. Apply Manual Overrides (if set in config)
    if MANUAL_BPM is not None: 
        bpm = MANUAL_BPM
        print(f"   Using Manual BPM: {bpm}")
        
    if MANUAL_OFFSET is not None: 
        offset = MANUAL_OFFSET
        print(f"   Using Manual Offset: {offset}")
        
    if AUDIO_FILE_PATH is not None: 
        audio_path = AUDIO_FILE_PATH
        print(f"   Using Manual Audio Path: {os.path.basename(audio_path)}")

    # 3. If audio still not found, auto-detect from the beatmap / metadata folder
    if not audio_path or not os.path.exists(audio_path):
        # Try the folder of the generated beatmap first
        search_folders = [os.path.dirname(GENERATED_BEATMAP_PATH)]
        if ORIGINAL_METADATA_PATH:
            search_folders.append(os.path.dirname(ORIGINAL_METADATA_PATH))
        
        for folder in search_folders:
            found = find_audio_in_folder(folder)
            if found:
                audio_path = found
                print(f"   Auto-detected Audio: {os.path.basename(audio_path)}")
                break

    # 4. Validation Checks
    if not audio_path or not os.path.exists(audio_path):
        print("\nError: Could not find audio file!")
        print(f"Searched for audio in: {os.path.dirname(GENERATED_BEATMAP_PATH)}")
        print(f"Supported extensions: {', '.join(AUDIO_EXTENSIONS)}")
        print("You can also set AUDIO_FILE_PATH explicitly in the script configuration.")
        return

    print(f"   Audio:  {os.path.basename(audio_path)}")
    print(f"   BPM:    {bpm}")
    print(f"   Offset: {offset}")
    
    # Check if GENERATED file is empty
    if os.path.getsize(GENERATED_BEATMAP_PATH) == 0:
        print(f"\nWarning: Generated file is empty: {GENERATED_BEATMAP_PATH}")
    
    # 4. Validate
    print("\n" + "-"*60)
    print("Running Validation...")
    results = validate_beatmap(
        audio_path=audio_path,
        beatmap_path=GENERATED_BEATMAP_PATH,
        bpm=bpm,
        offset=offset,
        tolerance_ms=50.0,
        difficulty="Easy"
    )
    
    print_validation_report(results)
    
    # Save outputs with timestamp
    from datetime import datetime
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_filename = f"validation_results_{timestamp_str}.json"
    viz_filename = f"validation_visualization_{timestamp_str}.html"
    
    save_results_json(results, json_filename)
    visualize_validation(results, viz_filename)
    
    print(f"\n Done! Check '{viz_filename}' (Open in Browser for Interactivity)")


if __name__ == "__main__":
    main()
