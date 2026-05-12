import sqlite3
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

# Connect to database
conn = sqlite3.connect('./pattern_finding_approach/Process_files_Cluster.db')

# Join on run_id, file_path, difficulty, measure_idx
data_query = """
SELECT m.cluster_id, f.*
FROM measure_cluster_assignments m
JOIN stepmania_features f 
ON m.run_id = f.run_id 
AND m.file_path = f.file_path 
AND m.difficulty = f.difficulty 
AND m.measure_idx = f.measure_idx
WHERE m.cluster_id != -1
LIMIT 15000;
"""

df = pd.read_sql_query(data_query, conn)
conn.close()

if not df.empty:
    feature_cols = ['total_active', 'jumps', 'max_dist', 'avg_dist', 'returns', 
                    'density', 'uniq_col', 'has_hold', 'has_mine', 'hold_duration', 
                    'symmetry_bias', 'crossover_count', 'incoming_holds', 'outgoing_holds']
    X = df[feature_cols].fillna(0)
    y = df['cluster_id']
    
    # Calculate Silhouette Score
    sil_score = silhouette_score(X, y)
    
    # Calculate Davies-Bouldin Index
    db_score = davies_bouldin_score(X, y)
    
    print(f"SILHOUETTE_SCORE:{sil_score}")
    print(f"DAVIES_BOULDIN_SCORE:{db_score}")
else:
    print("NO_DATA")
