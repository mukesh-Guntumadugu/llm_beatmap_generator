import numpy as np
import librosa
import os

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: 'whisper' library not found. Vocal/Lyric tracking is physically disabled until run on the cluster.")
    
# Cache the Whisper model locally to avoid reloading it entirely for every song
_whisper_model = None

def parse_time_map(bpms_str, offset_str, stops_str=None):
    """
    Builds a timeline map to convert Measure/Beat indices into absolute Seconds.
    This handles dynamic BPM changes which are common in Stepmania charts.
    """
    offset = float(offset_str) if offset_str else 0.0
    
    # Parse bpms: e.g. "0.000=120.000, 32.000=150.000"
    bpm_events = []
    if bpms_str:
        pairs = bpms_str.split(',')
        for p in pairs:
            if '=' in p:
                b, bpm = p.split('=')
                bpm_events.append((float(b.strip()), float(bpm.strip())))
    else:
        bpm_events.append((0.0, 120.0))
        
    bpm_events.sort(key=lambda x: x[0])
    
    time_map = []
    # Stepmania defines Beat 0 time as -OFFSET
    current_time = -offset
    current_beat = 0.0
    
    # If the first BPM happens after beat 0, assume it was active at beat 0 anyway
    current_bpm = bpm_events[0][1]
    time_map.append((0.0, current_bpm, current_time))
    
    for beat, bpm in bpm_events:
        if beat > current_beat:
            beats_passed = beat - current_beat
            seconds_passed = beats_passed * (60.0 / current_bpm)
            current_time += seconds_passed
        
        current_beat = beat
        current_bpm = bpm
        time_map.append((current_beat, current_bpm, current_time))
        
    # Note: `#STOPS` parsing would shift time offsets here. 
    # Left as a Phase 2 addition for charts with heavy gimmicks.
    return time_map

def get_time_for_beat(target_beat, time_map):
    """ Converts a Stepmania Beat index (float) to absolute Seconds """
    active_seg = time_map[0]
    for seg in time_map:
        if seg[0] <= target_beat:
            active_seg = seg
        else:
            break
            
    seg_beat, seg_bpm, seg_time = active_seg
    beats_passed = target_beat - seg_beat
    seconds_passed = beats_passed * (60.0 / seg_bpm)
    return seg_time + seconds_passed
    
def load_audio(audio_path):
    """ Loads the audio file into memory. Returns (y, sr) or (None, None) """
    if not os.path.exists(audio_path):
        return None, None
    try:
        # Load as mono, default sample rate
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return y, sr
    except Exception as e:
        print(f"Failed to load audio {audio_path}: {e}")
        return None, None

def extract_audio_features_for_slice(y, sr, start_time, end_time):
    """
    Given a pre-loaded audio array (y, sr), slices it and extracts:
    - RMS Energy (Loudness)
    - Onset Density (Rhythmic complexity / Spectral Flux)
    - Chromagram (Average harmonic content)
    - Spectral Centroid (Brightness)
    - Spectral Bandwidth
    - Spectral Contrast
    - Spectral Flatness
    - Tempogram (Local tempo strength)
    - 13 MFCCs (Timbre / Texture)
    Returns a list of 21 features.
    """
    if y is None or sr is None:
        return [0.0] * 21

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    if start_sample < 0: start_sample = 0
    if end_sample > len(y): end_sample = len(y)
    
    if start_sample >= end_sample:
        return [0.0] * 21
        
    slice_y = y[start_sample:end_sample]
    
    try:
        # 1. Energy (Loudness map)
        rms = librosa.feature.rms(y=slice_y).mean()
        
        # 2. Rhythm & Tempo
        onset_env = librosa.onset.onset_strength(y=slice_y, sr=sr)
        onset_density = onset_env.mean() if len(onset_env)>0 else 0.0
        
        tempogram = librosa.feature.tempogram(y=slice_y, sr=sr)
        tempo_strength = tempogram.mean() if tempogram.size > 0 else 0.0
        
        # 3. Harmony & Chords
        chroma = librosa.feature.chroma_stft(y=slice_y, sr=sr)
        chroma_mean = chroma.mean() if chroma.size > 0 else 0.0
        
        # 4. Spectral Features (Brightness, Bass/Treble, Texture)
        centroid = librosa.feature.spectral_centroid(y=slice_y, sr=sr).mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=slice_y, sr=sr).mean()
        contrast = librosa.feature.spectral_contrast(y=slice_y, sr=sr).mean()
        flatness = librosa.feature.spectral_flatness(y=slice_y).mean()
        
        # 5. Timbre (Instrument texture - 13 dimensions)
        mfccs = librosa.feature.mfcc(y=slice_y, sr=sr, n_mfcc=13)
        mfcc_means = mfccs.mean(axis=1).tolist()
        
        return [
            float(rms), float(onset_density), float(tempo_strength), 
            float(chroma_mean), float(centroid), float(bandwidth), 
            float(contrast), float(flatness)
        ] + mfcc_means
        
    except Exception:
        return [0.0] * 21

def transcribe_vocals(audio_path):
    """
    Transcribes the entire audio file through OpenAI Whisper to extract word timelines.
    Only runs if Whisper is available on the HPC cluster.
    """
    global _whisper_model
    if not WHISPER_AVAILABLE:
        return []
    
    if _whisper_model is None:
        print("\n  [Speech-To-Text] Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
        
    print(f"  [Speech-To-Text] Transcribing lyrics for {os.path.basename(audio_path)}...")
    try:
        result = _whisper_model.transcribe(audio_path, word_timestamps=True)
        words = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                words.append({
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "text": word["word"].strip()
                })
        return words
    except Exception as e:
        print(f"  [Speech-To-Text] Failed to transcribe: {e}")
        return []

def get_vocal_features_for_slice(vocal_words, start_time, end_time):
    """
    Given a list of word timestamps, calculates how many words and what 
    density of vocals occur inside this measure's exact milliseconds.
    Returns: [vocal_word_count, vocal_density]
    """
    if not vocal_words:
        return [0.0, 0.0]
        
    measure_duration = end_time - start_time
    if measure_duration <= 0:
        return [0.0, 0.0]
        
    word_count = 0
    total_vocal_time = 0.0
    
    for w in vocal_words:
        overlap_start = max(start_time, w["start"])
        overlap_end = min(end_time, w["end"])
        
        if overlap_start < overlap_end:
            word_count += 1
            total_vocal_time += (overlap_end - overlap_start)
            
    vocal_density = total_vocal_time / measure_duration
    return [float(word_count), float(vocal_density)]
