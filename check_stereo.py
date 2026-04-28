import os
import glob
import random
import subprocess
import re

def analyze_stereo_with_ffmpeg(audio_path):
    try:
        # We use an ffmpeg filter to subtract the Right channel from the Left channel
        # 1. channelsplit -> [L], [R]
        # 2. Invert phase of [R] -> [Rn]
        # 3. Mix [L] and [Rn] -> [diff]
        # 4. volumedetect on [diff]
        
        # Command explanation:
        # asplit=2[orig][copy];[copy]pan=stereo|c0=c0|c1=-c1[inv];[orig][inv]amix=inputs=2[diff]
        
        # Actually easier: use pan filter directly to subtract R from L
        # pan=mono|c0=0.5*c0-0.5*c1
        # Then run volumedetect
        
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-t", "30", # just test first 30 seconds
            "-af", "pan=mono|c0=0.5*c0-0.5*c1,volumedetect",
            "-f", "null", "/dev/null"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stderr
        
        # Find max_volume and mean_volume
        max_match = re.search(r"max_volume: ([-.\d]+) dB", output)
        mean_match = re.search(r"mean_volume: ([-.\d]+) dB", output)
        
        if not mean_match:
            # Maybe it's native mono, let's check streams
            if "mono" in output.lower():
                return "Native Mono"
            return "Failed to parse volume"
            
        mean_vol = float(mean_match.group(1))
        max_vol = float(max_match.group(1)) if max_match else -99.0
        
        # If mean volume of (L-R) is extremely low (e.g., -60 dB to -90 dB), it's practically mono.
        # If it's around -20 dB or higher, there's significant stereo difference.
        if mean_vol < -50:
            return f"Effectively Mono (Difference mean vol: {mean_vol} dB)"
        else:
            return f"True Stereo (Difference mean vol: {mean_vol} dB, max: {max_vol} dB)"
            
    except Exception as e:
        return f"Error: {e}"

def main():
    base_dir = "/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/src/musicForBeatmap"
    ogg_files = glob.glob(os.path.join(base_dir, "**", "*.ogg"), recursive=True)
    
    if not ogg_files:
        print("No .ogg files found.")
        return
        
    sample_size = min(20, len(ogg_files))
    sampled_files = random.sample(ogg_files, sample_size)
    
    print(f"Analyzing {sample_size} random songs for stereo separation using ffmpeg (Left minus Right)...\n")
    
    for file_path in sampled_files:
        filename = os.path.basename(file_path)
        result = analyze_stereo_with_ffmpeg(file_path)
        print(f"{filename[:30]:<30} | {result}")

if __name__ == "__main__":
    main()
