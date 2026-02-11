import librosa
import librosa.display
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def extract_and_plot_features(audio_path):
    print(f"Processing: {audio_path}")
    
    try:
        y, sr = librosa.load(audio_path)
        print(f"Audio loaded. Shape: {y.shape}, Sample Rate: {sr}")
        if len(y) == 0:
            print("Error: Loaded audio is empty.")
            return
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # Dictionary to store features and their plotting functions
    features = {}

    print("Extracting features...")

    # Helper to safely extract features
    def safe_extract(name, func, **kwargs):
        try:
            return func(**kwargs)
        except Exception as e:
            print(f"Failed to extract {name}: {e}")
            return None

    # 1. Waveform
    features['Waveform'] = {'data': y, 'type': 'wave'}

    # 2. Spectrogram (dB)
    try:
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        features['Spectrogram (dB)'] = {'data': S_db, 'type': 'spec', 'y_axis': 'log'}
    except Exception as e:
         print(f"Failed to extract Spectrogram (dB): {e}")

    # 3. Mel Spectrogram
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB_mel = librosa.power_to_db(S, ref=np.max)
        features['Mel Spectrogram'] = {'data': S_dB_mel, 'type': 'spec', 'y_axis': 'mel'}
    except Exception as e:
        print(f"Failed to extract Mel Spectrogram: {e}")

    # 4. Chromagram (STFT)
    try:
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['Chromagram (STFT)'] = {'data': chroma_stft, 'type': 'spec', 'y_axis': 'chroma'}
    except Exception as e:
        print(f"Failed to extract Chromagram (STFT): {e}")

    # 5. Chromagram (CQT)
    try:
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        features['Chromagram (CQT)'] = {'data': chroma_cqt, 'type': 'spec', 'y_axis': 'chroma'}
    except Exception as e:
        print(f"Failed to extract Chromagram (CQT): {e}")

    # 6. Spectral Centroid
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['Spectral Centroid'] = {'data': cent, 'type': 'line', 'times': librosa.times_like(cent)}
    except Exception as e:
        print(f"Failed to extract Spectral Centroid: {e}")

    # 7. Spectral Bandwidth
    try:
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['Spectral Bandwidth'] = {'data': spec_bw, 'type': 'line', 'times': librosa.times_like(spec_bw)}
    except Exception as e:
        print(f"Failed to extract Spectral Bandwidth: {e}")

    # 8. Spectral Rolloff
    try:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['Spectral Rolloff'] = {'data': rolloff, 'type': 'line', 'times': librosa.times_like(rolloff)}
    except Exception as e:
        print(f"Failed to extract Spectral Rolloff: {e}")

    # 9. Zero Crossing Rate
    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        features['Zero Crossing Rate'] = {'data': zcr, 'type': 'line', 'times': librosa.times_like(zcr)}
    except Exception as e:
        print(f"Failed to extract Zero Crossing Rate: {e}")

    # 10. MFCC
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features['MFCC'] = {'data': mfcc, 'type': 'spec', 'y_axis': 'linear'} 
    except Exception as e:
        print(f"Failed to extract MFCC: {e}")

    # 11. Spectral Contrast
    try:
        S_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['Spectral Contrast'] = {'data': S_contrast, 'type': 'spec', 'y_axis': 'linear'}
    except Exception as e:
        print(f"Failed to extract Spectral Contrast: {e}")

    # 12. Tonnetz
    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features['Tonnetz'] = {'data': tonnetz, 'type': 'spec', 'y_axis': 'tonnetz'}
    except Exception as e:
        print(f"Failed to extract Tonnetz: {e}")

    # 13. Harmonic Component
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        features['Harmonic Component'] = {'data': y_harm, 'type': 'wave'}
    except Exception as e:
        print(f"Failed to extract Harmonic Component: {e}")
        y_perc = None

    # 14. Percussive Component
    if y_perc is not None:
        features['Percussive Component'] = {'data': y_perc, 'type': 'wave'}

    # 15. Tempogram
    try:
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
        features['Tempogram'] = {'data': tempogram, 'type': 'spec', 'x_axis': 'time', 'y_axis': 'tempo'}
    except Exception as e:
        print(f"Failed to extract Tempogram: {e}")
    
    # 16. RMSE (Root Mean Square Energy)
    try:
        rmse = librosa.feature.rms(y=y)
        features['RMS Energy'] = {'data': rmse, 'type': 'line', 'times': librosa.times_like(rmse)}
    except Exception as e:
        print(f"Failed to extract RMS Energy: {e}")


    # Plotting
    num_features = len(features)
    cols = 2 # Reduced columns to make plots wider
    rows = (num_features + cols - 1) // cols
    
    # Calculate figure height based on rows (make it tall)
    # 5 inches per row is a good starting point for detailed visibility
    fig_width = 15
    fig_height = rows * 5 
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    print(f"Plotting {num_features} features...")

    for i, (name, content) in enumerate(features.items()):
        ax = axes[i]
        ax.set_title(name)
        
        if content['type'] == 'wave':
            librosa.display.waveshow(content['data'], sr=sr, ax=ax)
        elif content['type'] == 'spec':
            y_axis = content.get('y_axis', None)
            x_axis = content.get('x_axis', 'time')
            librosa.display.specshow(content['data'], x_axis=x_axis, y_axis=y_axis, sr=sr, ax=ax)
            if name == 'Tempogram':
                pass 
            else:
                pass 
        elif content['type'] == 'line':
            ax.plot(content['times'], content['data'][0], label=name)
            ax.legend()
    
    # Hide empty subplots
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    print("Done. Opening scrollable window...")

    # Scrollable Window Implementation
    try:
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        root = tk.Tk()
        root.title(f"Audio Features: {os.path.basename(audio_path)}")
        
        # maximize window
        try:
             root.state('zoomed') # Windows
        except:
             try:
                root.attributes('-zoomed', True) # Linux
             except:
                root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}") # Mac/General


        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(main_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        second_frame = tk.Frame(canvas)

        canvas.create_window((0, 0), window=second_frame, anchor="nw")

        canvas_widget = FigureCanvasTkAgg(fig, master=second_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas_widget, second_frame)
        toolbar.update()
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Mousewheel scrolling
        def _on_mousewheel(event):
             canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Mac scroll handling might differ (delta usually smaller/inverted or different scale)
        # Standard binding for windows/linux is <MouseWheel>, mac can be <Button-4>/<Button-5> or just MouseWheel
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        root.mainloop()

    except ImportError:
         print("Tkinter not found. Falling back to standard plt.show() (non-scrollable).")
         plt.show()
    except Exception as e:
        print(f"Error launching scrollable window: {e}. Falling back to standard plt.show().")
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default fallback or usage
        print("Usage: python extract_features.py <path_to_audio_file>")
        # Try a default file if exists, mainly for dev/test
        default_file = "../musicForBeatmap/repro_slice.ogg" 
        if os.path.exists(default_file):
             print(f"No file provided. running on default: {default_file}")
             extract_and_plot_features(default_file)
        else:
            print("Please provide an audio file path.")
    else:
        extract_and_plot_features(sys.argv[1])
