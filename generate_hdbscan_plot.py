import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Connect to database
conn = sqlite3.connect('./pattern_finding_approach/Process_files_Cluster.db')

# Query top 5 clusters by count (excluding noise -1)
top_clusters_query = """
SELECT cluster_id, SUM(count) as total_count 
FROM cluster_counts 
WHERE cluster_id != -1 
GROUP BY cluster_id 
ORDER BY total_count DESC 
LIMIT 5;
"""
top_clusters = pd.read_sql_query(top_clusters_query, conn)

cluster_ids = tuple(top_clusters['cluster_id'].tolist() + [-1])

# Join on run_id, file_path, difficulty, measure_idx
data_query = f"""
SELECT m.cluster_id, f.*
FROM measure_cluster_assignments m
JOIN stepmania_features f 
ON m.run_id = f.run_id 
AND m.file_path = f.file_path 
AND m.difficulty = f.difficulty 
AND m.measure_idx = f.measure_idx
WHERE m.cluster_id IN {cluster_ids}
LIMIT 10000;
"""

df = pd.read_sql_query(data_query, conn)

if not df.empty:
    # Drop non-feature columns
    feature_cols = ['total_active', 'jumps', 'max_dist', 'avg_dist', 'returns', 
                    'density', 'uniq_col', 'has_hold', 'has_mine', 'hold_duration', 
                    'symmetry_bias', 'crossover_count', 'incoming_holds', 'outgoing_holds']
    X = df[feature_cols].fillna(0)
    y = df['cluster_id']
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot Noise first
    mask_noise = y == -1
    plt.scatter(X_pca[mask_noise, 0], X_pca[mask_noise, 1], c='#A9A9A9', label='Noise (Anomalous)', alpha=0.2, s=15)
    
    # Plot top 5 clusters
    colors = ['#E63946', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
    for i, c_id in enumerate(top_clusters['cluster_id']):
        mask = y == c_id
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=f'Cluster {c_id}', alpha=0.8, s=25)
        
    plt.title('PCA Projection of Top 5 HDBSCAN Token Clusters vs Noise', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    os.makedirs('outputs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('outputs/hdbscan_pca_plot.png', dpi=300, bbox_inches='tight')

conn.close()
