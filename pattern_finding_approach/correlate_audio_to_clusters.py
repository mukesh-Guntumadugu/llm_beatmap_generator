import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize the mathematical correlation between Audio Moods and Stepmania Clusters")
    parser.add_argument('--db_path', type=str, default='processed_files.db',
                        help="Path to the SQLite database synced from the HPC cluster")
    parser.add_argument('--output_dir', type=str, default='audio_cluster_correlations',
                        help="Directory to save the visualization PNGs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        conn = sqlite3.connect(args.db_path)
    except Exception as e:
        print(f"Failed to connect to database {args.db_path}: {e}")
        return
    
    print(f"Executing SQL JOIN to perfectly link Audio Features to Stepmania Clusters...")
    query = """
    SELECT 
        c.cluster_id, 
        a.rms_energy_l, a.rms_energy_r,
        a.onset_density_l, a.onset_density_r,
        a.tempo_strength_l, a.tempo_strength_r,
        a.spectral_centroid_l, a.spectral_centroid_r,
        a.vocal_density,
        a.spectral_contrast_l, a.spectral_contrast_r,
        a.spectral_flatness_l, a.spectral_flatness_r
    FROM measure_cluster_assignments c
    JOIN audio_features a 
     ON c.run_id = a.run_id 
     AND c.file_path = a.file_path 
     AND c.difficulty = a.difficulty 
     AND c.measure_idx = a.measure_idx
    WHERE c.cluster_id != -1
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        print("No correlation data found! Please make sure your current cluster job has finished completely, and then run your rsync command to pull the latest processed_files.db to your Mac.")
        return
        
    print(f"Successfully joined {len(df):,} beatmap measures across both musical audio and physical patterns!")
    
    # 1. Filter to top 20 largest clusters so the visualization is actually readable
    top_clusters = df['cluster_id'].value_counts().nlargest(20).index
    df_top = df[df['cluster_id'].isin(top_clusters)]
    
    # 2. Get the pure mathematical average audio environment for each physical pattern
    cluster_means = df_top.groupby('cluster_id').mean()
    
    # Standardize the features so they plot cleanly on a single heatmap 
    # (Because audio `energy` is a tiny decimal, but `spectral_centroid` is in thousands of Hz!)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_means)
    df_scaled = pd.DataFrame(scaled_features, index=cluster_means.index, columns=cluster_means.columns)
    
    print("\nGenerating Audio Mood Heatmap...")
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_scaled, cmap='coolwarm', center=0, annot=True, fmt=".2f")
    plt.title("Audio Mood Profiles for Top 20 Stepmania Clusters (Z-Scored)")
    plt.ylabel("HDBSCAN Cluster ID")
    plt.xlabel("Audio Characteristic")
    
    heat_path = os.path.join(args.output_dir, 'audio_mood_heatmap.png')
    plt.savefig(heat_path, dpi=300, bbox_inches='tight')
    print(f" -> Saved Heatmap to: {heat_path}")
    plt.close()
    
    print("Generating Vocal Distribution Boxplots...")
    # 4. Plot 2: Violin Plot showing whether vocals dictate specific clusters
    plt.figure(figsize=(14, 6))
    
    # Let's examine the top 8 clusters explicitly
    top_8_clusters = df['cluster_id'].value_counts().nlargest(8).index
    df_8 = df[df['cluster_id'].isin(top_8_clusters)]
    
    sns.boxplot(data=df_8, x='cluster_id', y='vocal_density', palette='plasma')
    plt.title("Vocal Presence Spread Across the Top 8 Physical Clusters")
    plt.xlabel("HDBSCAN Cluster ID")
    plt.ylabel("Vocal Density (Words sung during measure)")
    
    box_path = os.path.join(args.output_dir, 'vocal_density_boxplot.png')
    plt.savefig(box_path, dpi=300, bbox_inches='tight')
    print(f" -> Saved Boxplot to: {box_path}")
    plt.close()
    
    print("\nVisualizations natively compiled! ")
    print("💡 How to read the heatmap:")
    print("- A dark RED square means that cluster is triggered when that audio feature is extremely HIGH (e.g. intense drums).")
    print("- A dark BLUE square means that cluster happens when the audio drops LOW (e.g. a silent fadeout).")

if __name__ == '__main__':
    main()
