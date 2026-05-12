import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

# Connect to database
conn = sqlite3.connect('./pattern_finding_approach/Process_files_Cluster.db')

# Query the difficulty distribution for the top clusters + noise
query = """
SELECT 
    CASE 
        WHEN LOWER(difficulty) IN ('beginner', 'easy', 'medium', 'hard', 'challenge') 
        THEN LOWER(difficulty) 
        ELSE 'other' 
    END as diff_level,
    cluster_id,
    COUNT(*) as count
FROM measure_cluster_assignments
WHERE LOWER(difficulty) IN ('beginner', 'easy', 'medium', 'hard', 'challenge')
  AND cluster_id IN (4, 3981, 13, 14, 5059, -1)
GROUP BY diff_level, cluster_id;
"""
df = pd.read_sql_query(query, conn)
conn.close()

if not df.empty:
    # Pivot the data
    pivot_df = df.pivot(index='diff_level', columns='cluster_id', values='count').fillna(0)
    
    # Normalize to 100%
    pivot_df_perc = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    # Reorder index to match difficulty progression
    diff_order = ['beginner', 'easy', 'medium', 'hard', 'challenge']
    pivot_df_perc = pivot_df_perc.reindex(diff_order)
    
    # Capitalize for plotting
    pivot_df_perc.index = pivot_df_perc.index.str.capitalize()
    
    # Map cluster IDs to names for legend
    cluster_names = {
        -1: 'Noise (Anomalous)',
        4: 'C4: Low-Density Basic Steps',
        13: 'C13: Med-Density Alternating',
        14: 'C14: Med-Density Mixed',
        5059: 'C5059: Complex Streams (Jumps/Holds)',
        3981: 'C3981: High-Density Jump Streams'
    }
    pivot_df_perc.rename(columns=cluster_names, inplace=True)
    
    # Order columns by density (Noise, then C4 -> C3981)
    col_order = [
        'Noise (Anomalous)',
        'C4: Low-Density Basic Steps',
        'C13: Med-Density Alternating',
        'C14: Med-Density Mixed',
        'C5059: Complex Streams (Jumps/Holds)',
        'C3981: High-Density Jump Streams'
    ]
    # Only keep columns that exist in the dataframe
    col_order = [c for c in col_order if c in pivot_df_perc.columns]
    pivot_df_perc = pivot_df_perc[col_order]

    # Plot
    colors = ['#A9A9A9', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#E63946']
    
    ax = pivot_df_perc.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors, edgecolor='black')
    
    plt.title('Distribution of HDBSCAN Token Clusters Across Difficulties', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Beatmap Difficulty Level', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage of Measures (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    
    # Move legend outside
    plt.legend(title='Token Archetype', bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
    plt.tight_layout()
    
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/cluster_difficulty_distribution.png', dpi=300, bbox_inches='tight')
    print("Generated outputs/cluster_difficulty_distribution.png")
