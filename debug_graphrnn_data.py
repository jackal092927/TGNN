"""
Debug GraphRNN data preparation
"""

import pandas as pd
import json
import numpy as np

def debug_data_preparation():
    data_name = 'triadic_perfect_long_dense'
    
    # Load data
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    print("🔍 DEBUGGING DATA PREPARATION")
    print("=" * 50)
    
    print(f"📊 Dataset Overview:")
    print(f"   Total edges: {len(g_df)}")
    print(f"   Timestamps: {sorted(g_df['ts'].unique())}")
    print(f"   Ground truth timestamps: {sorted([int(k) for k in ground_truth.keys()])}")
    
    # Create node mapping
    all_nodes = sorted(set(g_df['u'].tolist() + g_df['i'].tolist()))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    print(f"   Nodes: {len(all_nodes)}")
    
    # Split data temporally
    timestamps = sorted(g_df['ts'].unique())
    total_ts = len(timestamps)
    train_ts = int(total_ts * 0.6)
    val_ts = int(total_ts * 0.2)
    
    train_timestamps = timestamps[:train_ts]
    val_timestamps = timestamps[train_ts:train_ts + val_ts]
    test_timestamps = timestamps[train_ts + val_ts:]
    
    print(f"\n📅 Temporal Split:")
    print(f"   Train: {train_timestamps}")
    print(f"   Val: {val_timestamps}")
    print(f"   Test: {test_timestamps}")
    
    # Check ground truth for training transitions
    print(f"\n🎯 Training Transitions:")
    training_pairs = 0
    for i in range(len(train_timestamps) - 1):
        current_ts = train_timestamps[i]
        next_ts = train_timestamps[i + 1]
        
        print(f"   {current_ts} → {next_ts}:")
        
        # Check if next_ts has ground truth
        if str(next_ts) in ground_truth:
            gt_indices = ground_truth[str(next_ts)]
            print(f"     GT indices: {gt_indices}")
            
            # Get actual edges for next timestamp
            edges_at_next_ts = g_df[g_df['ts'] == next_ts]
            print(f"     Actual edges at {next_ts}: {len(edges_at_next_ts)}")
            
            if len(edges_at_next_ts) > 0:
                edge_pairs = []
                for _, row in edges_at_next_ts.iterrows():
                    u_idx = node_to_idx[int(row.u)]
                    v_idx = node_to_idx[int(row.i)]
                    edge_pairs.append((u_idx, v_idx))
                print(f"     Edge pairs: {edge_pairs[:5]}...")
                training_pairs += len(edge_pairs)
            else:
                print(f"     ⚠️  No edges found for timestamp {next_ts}")
        else:
            print(f"     ❌ No ground truth for timestamp {next_ts}")
    
    print(f"\n📈 Training Data Summary:")
    print(f"   Total training edge pairs: {training_pairs}")
    
    # Check validation data
    print(f"\n🔍 Validation Data:")
    val_pairs = 0
    val_sequence_timestamps = timestamps[:train_ts + val_ts]
    for i in range(len(val_sequence_timestamps) - 1):
        current_ts = val_sequence_timestamps[i]
        next_ts = val_sequence_timestamps[i + 1]
        
        if next_ts in val_timestamps and str(next_ts) in ground_truth:
            edges_at_next_ts = g_df[g_df['ts'] == next_ts]
            val_pairs += len(edges_at_next_ts)
            print(f"   {current_ts} → {next_ts}: {len(edges_at_next_ts)} edges")
    
    print(f"   Total validation edge pairs: {val_pairs}")
    
    # Sample some ground truth data
    print(f"\n📋 Sample Ground Truth:")
    for ts_str, indices in list(ground_truth.items())[:5]:
        ts = int(ts_str)
        edges_at_ts = g_df[g_df['ts'] == ts]
        print(f"   Timestamp {ts}: GT indices {indices}, Actual edges: {len(edges_at_ts)}")
        if len(edges_at_ts) > 0:
            sample_edges = [(int(row.u), int(row.i)) for _, row in edges_at_ts.head(3).iterrows()]
            print(f"     Sample edges: {sample_edges}")

if __name__ == "__main__":
    debug_data_preparation()
