import re
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os

STEPFILE = 'Springtime.ssc'
AUDIOFILE = 'Kommisar - Springtime.mp3'

def parse_ssc(filepath):
    """


python3 "src/musicForBeatmap/Springtime/analyze_beatmap.py" --start 10 --end 20
python3 "src/musicForBeatmap/Springtime/analyze_beatmap.py" --start 0 --end 5
6628 in .ssc file

    Parses the StepMania SSC file to extract:
    - Offset
    - BPMs
    - Note data (for the Challenge difficulty)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract global Offset and BPMS
    offset_match = re.search(r'#OFFSET:(-?[\d\.]+);', content)
    bpms_match = re.search(r'#BPMS:(.*?);', content, re.DOTALL)
    
    if not offset_match or not bpms_match:
        print("Error: Could not find OFFSET or BPMS in SSC file.")
        return None

    global_offset = float(offset_match.group(1))
    
    # Parse BPMs: "0=181.685,304=90.843,311=181.685"
    bpm_str = bpms_match.group(1).replace('\n', '').strip()
    bpms = []
    for pair in bpm_str.split(','):
        if '=' in pair:
            beat, value = pair.split('=')
            bpms.append((float(beat), float(value)))
    
    # Extract the Challenge Chart
    charts = content.split('#NOTEDATA:;')
    challenge_chart = None
    
    for chart in charts:
        # Prioritize Easy as per user request to match the specific chart they are looking at
        # Ideally this should be an argument, but for now we switch the default.
        if '#DIFFICULTY:Easy;' in chart:
             challenge_chart = chart
             break
        elif '#DIFFICULTY:Hard;' in chart:
             challenge_chart = chart
        elif '#DIFFICULTY:Challenge;' in chart and not challenge_chart:
             challenge_chart = chart
    
    if not challenge_chart:
        print("Error: Could not find Challenge or Hard difficulty chart.")
        return None

    # Extract Notes
    notes_match = re.search(r'#NOTES:(.*?);', challenge_chart, re.DOTALL)
    if not notes_match:
        print("Error: Could not find NOTES data in chart.")
        return None
    
    note_data_str = notes_match.group(1).strip()
    return {
        'offset': global_offset,
        'bpms': bpms,
        'notes': note_data_str
    }

def get_note_times(parsed_data):
    offset = parsed_data['offset']
    bpms = parsed_data['bpms']
    note_str = parsed_data['notes']
    
    measures = re.split(r',', note_str)
    bpm_map = sorted(bpms, key=lambda x: x[0])
    
    note_times = []
    
    for measure_idx, measure in enumerate(measures):
        rows = measure.strip().split()
        if not rows: continue
        
        rows_per_measure = len(rows)
        beats_per_row = 4.0 / rows_per_measure
        
        for row_idx, row in enumerate(rows):
            current_beat = (measure_idx * 4) + (row_idx * beats_per_row)
            
            if any(c not in '0' for c in row):
                # Calculate time for this beat
                t = 0.0
                cur_b = 0.0
                cur_bpm = bpm_map[0][1]
                
                for next_b, next_bpm in bpm_map:
                    if next_b <= cur_b: 
                        cur_bpm = next_bpm
                        continue   
                    if current_beat < next_b:
                        break
                    
                    t += (next_b - cur_b) * (60.0 / cur_bpm)
                    cur_b = next_b
                    cur_bpm = next_bpm
                
                t += (current_beat - cur_b) * (60.0 / cur_bpm)
                final_time = t - offset
                
                note_times.append({
                    'time': final_time,
                    'beat': current_beat,
                    'row': row
                })
                
    return note_times

def row_to_arrows(row_str):
    """
    Converts '1000' to ['Left'], '0100' to ['Down'] etc.
    Supports 4-panel (DDR) and 5-panel (Pump).
    """
    if len(row_str) == 4:
        mapping = ['Left', 'Down', 'Up', 'Right']
    elif len(row_str) == 5:
        # Pump It Up standard: DL, UL, Center, UR, DR
        mapping = ['DownLeft', 'UpLeft', 'Center', 'UpRight', 'DownRight']
    else:
        # Fallback or unknown
        mapping = [f"Col{i}" for i in range(len(row_str))]
        
    arrows = []
    for i, char in enumerate(row_str):
        if i >= len(mapping): break
        
        if char in ['1', '2', '4']: # Tap, Hold Head, Roll Head
            arrows.append(mapping[i])
        elif char == 'M': # Mine
            arrows.append(f"Mine({mapping[i]})")
    return arrows

def analyze_and_plot(audio_path, note_times, start_arg=None, end_arg=None):
    print(f"Loading {audio_path}...")
    y, sr = librosa.load(audio_path)
    
    # Frequency analysis
    print("Computing STFT...")
    D = np.abs(librosa.stft(y))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(range(D.shape[1]), sr=sr)
    
    print("\n-----------------------------------------------------------")
    print("DETAILED STEP ANALYSIS (First 20 Steps)")
    print("-----------------------------------------------------------")
    print(f"{'Time (s)':<10} | {'Beat':<8} | {'Frequency (Hz)':<15} | {'Note':<6} | {'Arrows'}")
    print("-" * 80)
    
    for note in note_times[:20]:
        t = note['time']
        
        # Find frequency at this time
        frame_idx = (np.abs(times - t)).argmin()
        if frame_idx >= D.shape[1]: break
        
        spectrum = D[:, frame_idx]
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        
        # Convert to musical note
        note_name = librosa.hz_to_note(peak_freq) if peak_freq > 0 else "N/A"
        
        arrows = row_to_arrows(note['row'])
        arrow_str = ", ".join(arrows)
        
        print(f"{t:<10.3f} | {note['beat']:<8.2f} | {peak_freq:<15.1f} | {note_name:<6} | {arrow_str}")

    print("-" * 80)
    
    # Plotting
    print("\nGenerating Interactive Visualization...")
    plt.figure(figsize=(12, 6))
    
    # Plot Spectrogram
    librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram with Beatmap Overlay (Full Song)')
    
    # Determine Time Range
    start_time = start_arg if start_arg is not None else 0.0
    end_time = end_arg if end_arg is not None else times[-1]
    
    title_str = f'Spectrogram with Beatmap Overlay ({start_time}s - {end_time:.2f}s)'
    plt.title(title_str)
    
    # Overlay Arrows
    plt.xlim(start_time, end_time)
    plt.ylim(20, 8000) # audible range
    
    for note in note_times:
        t = note['time']
        if start_time <= t <= end_time:
            arrows = row_to_arrows(note['row'])
            for arrow in arrows:
                # Plot different markers for different arrows
                marker = 'o'
                color = 'white'
                y_pos = 100 # Base height
                
                if 'Left' == arrow: # Strict check for DDR Left
                    marker = '<'
                    color = 'cyan'
                    y_pos = 200
                elif 'Right' == arrow: 
                    marker = '>'
                    color = 'red'
                    y_pos = 200
                elif 'Up' == arrow: 
                    marker = '^'
                    color = 'lime'
                    y_pos = 500
                elif 'Down' == arrow: 
                    marker = 'v'
                    color = 'yellow'
                    y_pos = 100
                # Pump It Up Specific
                elif 'DownLeft' in arrow:
                    marker = 'v'
                    color = 'cyan' # Blue-ish for DL
                    y_pos = 100
                elif 'UpLeft' in arrow:
                    marker = '^'
                    color = 'cyan'
                    y_pos = 500
                elif 'Center' in arrow:
                    marker = 's' # Square
                    color = 'gold' # Yellow center
                    y_pos = 300
                elif 'UpRight' in arrow:
                    marker = '^'
                    color = 'red'
                    y_pos = 500
                elif 'DownRight' in arrow:
                    marker = 'v'
                    color = 'red'
                    y_pos = 100
                
                plt.plot(t, y_pos, marker=marker, color=color, markersize=12, markeredgecolor='black')
                plt.text(t, y_pos*1.5, arrow, color='white', fontsize=8, rotation=90, verticalalignment='bottom')

    plt.tight_layout()
    print("Opening interactive plot window...")
    plt.show()

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Analyze and visualize beatmap")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds")
    args = parser.parse_args()

    # Change directory if needed (legacy support for hardcoded path)
    if not os.path.exists('Springtime.ssc'):
        try:
             os.chdir('/Users/mukeshguntumadugu/Documents/Stepmania/stepmania/Songs/StepMania 5/Springtime')
        except:
             pass
             
    data = parse_ssc(STEPFILE)
    if data:
        notes = get_note_times(data)
        analyze_and_plot(AUDIOFILE, notes, start_arg=args.start, end_arg=args.end)
