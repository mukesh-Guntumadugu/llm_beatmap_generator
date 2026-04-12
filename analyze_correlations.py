import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', default='./pattern_finding_approach/processed_files.db')
    parser.add_argument('--output_dir', default='./pattern_finding_results/')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    
    # Get latest run_id
    run_id_df = pd.read_sql("SELECT MAX(run_id) as max_run FROM audio_features", conn)
    if run_id_df.empty or run_id_df['max_run'].iloc[0] is None:
        print("No audio features found in DB.")
        return
        
    latest_run = run_id_df['max_run'].iloc[0]
    print(f"Analyzing run: {latest_run}")
    
    # Query joined features
    query = """
    SELECT 
        a.rms_energy, a.onset_density, a.tempo_strength, a.chroma_mean, 
        a.spectral_centroid, a.spectral_bandwidth, a.spectral_contrast, a.spectral_flatness,
        s.total_active, s.jumps, s.max_dist, s.avg_dist, s.returns, s.density, s.uniq_col, s.has_hold, s.has_mine
    FROM audio_features a
    JOIN stepmania_features s 
        ON a.run_id = s.run_id 
        AND a.file_path = s.file_path 
        AND a.difficulty = s.difficulty 
        AND a.measure_idx = s.measure_idx
    WHERE a.run_id = ?
    """
    
    df = pd.read_sql(query, conn, params=(latest_run,))
    conn.close()
    
    if df.empty:
        print("No joined feature data could be found. Did you re-run pattern_finding.py to map the Stepmania features?")
        return
        
    print(f"Loaded {len(df)} aligned measures for correlation.")
    
    corr_matrix = df.corr()
    
    # We only care about the intersection: Audio Vs Stepmania
    sm_cols = ['total_active', 'jumps', 'max_dist', 'avg_dist', 'returns', 'density', 'uniq_col', 'has_hold', 'has_mine']
    audio_cols = [c for c in df.columns if c not in sm_cols]
    
    cross_corr = corr_matrix.loc[audio_cols, sm_cols]
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(cross_corr, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
    plt.title("Correlation Map: Audio Features vs Stepmania Physical Features", fontsize=16)
    plt.tight_layout()
    
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "audio_stepmania_correlation.png")
    plt.savefig(out_path, dpi=200)
    print(f"\nSUCCESS! Correlation Heatmap saved to: {out_path}")

if __name__ == "__main__":
    main()
