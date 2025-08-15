"""
Analyze the large gap between GraphMamba validation and test accuracy
"""

import pandas as pd
import json
import numpy as np

def analyze_temporal_splits():
    """Analyze the temporal distribution and characteristics of train/val/test splits"""
    
    # Load data
    data_name = 'triadic_perfect_long_dense'
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    
    with open(f'./processed/{data_name}/ml_{data_name}_gt_fixed.json', 'r') as f:
        ground_truth = json.load(f)
    
    timestamps = sorted(g_df['ts'].unique())
    print(f"Total timestamps: {len(timestamps)}")
    print(f"Timestamp range: {timestamps[0]:.1f} - {timestamps[-1]:.1f}")
    
    # Split analysis (same as in GraphMamba)
    train_ts = int(len(timestamps) * 0.7)  # 19
    val_ts = int(len(timestamps) * 0.15)   # 4  
    test_ts = len(timestamps) - train_ts - val_ts  # 5
    
    train_timestamps = timestamps[:train_ts]
    val_timestamps = timestamps[train_ts:train_ts + val_ts]
    test_timestamps = timestamps[train_ts + val_ts:]
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_timestamps)} timestamps ({timestamps[0]:.1f} - {timestamps[train_ts-1]:.1f})")
    print(f"Val:   {len(val_timestamps)} timestamps ({timestamps[train_ts]:.1f} - {timestamps[train_ts + val_ts - 1]:.1f})")
    print(f"Test:  {len(test_timestamps)} timestamps ({timestamps[train_ts + val_ts]:.1f} - {timestamps[-1]:.1f})")
    
    # Analyze edges per timestamp
    print(f"\n{'Timestamp':<12} {'Edges':<8} {'New Edges':<12} {'Cumulative':<12} {'Split':<8}")
    print("-" * 60)
    
    cumulative_edges = 0
    prev_edge_count = 0
    
    for i, ts in enumerate(timestamps):
        edges_at_ts = len(g_df[g_df['ts'] <= ts])
        new_edges = edges_at_ts - prev_edge_count
        prev_edge_count = edges_at_ts
        
        if i < train_ts:
            split = "Train"
        elif i < train_ts + val_ts:
            split = "Val"
        else:
            split = "Test"
            
        print(f"{ts:<12.1f} {new_edges:<8} {new_edges:<12} {edges_at_ts:<12} {split:<8}")
    
    # Analyze triadic closure patterns
    print(f"\nTriadic closure analysis:")
    print(f"{'Split':<8} {'Timestamps':<12} {'Avg New Edges':<15} {'Triadic Closures':<18}")
    print("-" * 60)
    
    for split_name, split_timestamps in [("Train", train_timestamps), ("Val", val_timestamps), ("Test", test_timestamps)]:
        new_edges_per_ts = []
        triadic_closures = 0
        
        for ts in split_timestamps:
            if str(ts) in ground_truth:
                triadic_closures += len(ground_truth[str(ts)])
            
            # Count new edges at this timestamp
            new_edges = len(g_df[g_df['ts'] == ts])
            new_edges_per_ts.append(new_edges)
        
        avg_new_edges = np.mean(new_edges_per_ts) if new_edges_per_ts else 0
        print(f"{split_name:<8} {len(split_timestamps):<12} {avg_new_edges:<15.2f} {triadic_closures:<18}")
    
    # Analyze graph density evolution
    print(f"\nGraph density evolution:")
    max_node = max(g_df['u'].max(), g_df['i'].max()) + 1
    max_possible_edges = max_node * (max_node - 1) // 2
    
    print(f"Max nodes: {max_node}")
    print(f"Max possible edges: {max_possible_edges}")
    
    densities = []
    for ts in timestamps:
        edges_count = len(g_df[g_df['ts'] <= ts])
        density = edges_count / max_possible_edges
        densities.append(density)
    
    train_density = np.mean(densities[:train_ts])
    val_density = np.mean(densities[train_ts:train_ts + val_ts])
    test_density = np.mean(densities[train_ts + val_ts:])
    
    print(f"Average density - Train: {train_density:.4f}, Val: {val_density:.4f}, Test: {test_density:.4f}")
    
    # Check for temporal drift
    print(f"\nTemporal characteristics:")
    
    # Edges per timestamp in each split
    train_edge_counts = [len(g_df[g_df['ts'] == ts]) for ts in train_timestamps]
    val_edge_counts = [len(g_df[g_df['ts'] == ts]) for ts in val_timestamps]  
    test_edge_counts = [len(g_df[g_df['ts'] == ts]) for ts in test_timestamps]
    
    print(f"New edges per timestamp:")
    print(f"  Train: mean={np.mean(train_edge_counts):.2f}, std={np.std(train_edge_counts):.2f}")
    print(f"  Val:   mean={np.mean(val_edge_counts):.2f}, std={np.std(val_edge_counts):.2f}")
    print(f"  Test:  mean={np.mean(test_edge_counts):.2f}, std={np.std(test_edge_counts):.2f}")

if __name__ == "__main__":
    analyze_temporal_splits()
