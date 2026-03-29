import os
import glob
import matplotlib.pyplot as plt

BASE_DIR = "/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements"

songs = []
original_counts = []
llm_counts = []

print("Scanning directories for onset data...")
for song_dir in os.listdir(BASE_DIR):
    full_path = os.path.join(BASE_DIR, song_dir)
    if not os.path.isdir(full_path) or song_dir.startswith("_"): 
        continue
    
    # 1. Parse 'original_onsets' (Librosa extracted algorithmic ground truth)
    original_csv_files = glob.glob(os.path.join(full_path, "original_onsets_*.csv"))
    original = 0
    if original_csv_files:
        try:
            with open(original_csv_files[-1], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                original = max(0, len(lines) - 1)  # Subtract 1 for CSV header
        except Exception:
            pass
            
    # 2. Parse 'Gemini_onsets' (LLM AI detected onsets)
    llm_csv_files = glob.glob(os.path.join(full_path, "Gemini_onsets_*.csv"))
    llm = 0
    if llm_csv_files:
        try:
            with open(llm_csv_files[-1], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                llm = max(0, len(lines) - 1)
        except Exception:
            pass
            
    # Save for graphing
    songs.append(song_dir[:12] + '..') # Truncate long names
    original_counts.append(original)
    llm_counts.append(llm)

# Filter out songs with zero data
filtered_songs, filtered_original, filtered_llm = [], [], []
for s, o, l in zip(songs, original_counts, llm_counts):
    if o > 0 or l > 0:
        filtered_songs.append(s)
        filtered_original.append(o)
        filtered_llm.append(l)

print("Generating corrected matplotlib chart...")
x = range(len(filtered_songs))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
ax.bar([i - width/2 for i in x], filtered_original, width, label='Original Audio Onsets (Librosa/Algorithm)', color='#2ca02c') # green
ax.bar([i + width/2 for i in x], filtered_llm, width, label='AI Detected Onsets (LLM Output)', color='#d62728') # red

ax.set_ylabel('Total Number of Onsets Detected', fontsize=12, fontweight='bold')
ax.set_title('AI Deep Learning Onsets vs Algorithmic (Librosa) Audio Onsets', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(filtered_songs, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
output_path = '/Users/mukeshguntumadugu/.gemini/antigravity/brain/b122fa0f-01dc-4487-997a-68c11a183845/artifacts/onset_comparison.png'
plt.savefig(output_path, dpi=300)
print(f"Chart successfully saved to {output_path}")
