"""
Investigate the timestamp mismatch between data and ground truth
"""

import pandas as pd
import json
import numpy as np

def investigate_timestamp_mismatch():
    data_name = 'triadic_perfect_long_dense'
    
    print("🔍 INVESTIGATING TIMESTAMP MISMATCH")
    print("=" * 60)
    
    # Load data
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    print("📊 DATA ANALYSIS:")
    print(f"   Graph CSV columns: {list(g_df.columns)}")
    print(f"   Graph CSV shape: {g_df.shape}")
    print(f"   Data timestamps: {sorted(g_df['ts'].unique())}")
    print(f"   Data timestamp range: {g_df['ts'].min()} - {g_df['ts'].max()}")
    print(f"   Total unique data timestamps: {len(g_df['ts'].unique())}")
    
    print(f"\n🎯 GROUND TRUTH ANALYSIS:")
    gt_timestamps = [int(k) for k in ground_truth.keys()]
    print(f"   GT timestamps: {sorted(gt_timestamps)[:10]}... (showing first 10)")
    print(f"   GT timestamp range: {min(gt_timestamps)} - {max(gt_timestamps)}")
    print(f"   Total GT timestamps: {len(gt_timestamps)}")
    
    print(f"\n❓ MISMATCH INVESTIGATION:")
    print(f"   Data uses timestamps: {sorted(g_df['ts'].unique())}")
    print(f"   GT uses timestamps: {sorted(gt_timestamps)[:20]}...")
    
    # Check if GT timestamps correspond to edge indices
    print(f"\n🔍 HYPOTHESIS: GT timestamps might be edge indices")
    print(f"   Total edges in data: {len(g_df)}")
    print(f"   Max GT timestamp: {max(gt_timestamps)}")
    print(f"   Max edge index would be: {len(g_df) - 1}")
    
    # Check some specific examples
    print(f"\n📋 DETAILED EXAMPLES:")
    
    # Look at first few GT entries
    for i, (ts_str, indices) in enumerate(list(ground_truth.items())[:5]):
        ts = int(ts_str)
        print(f"\n   GT Entry {i+1}:")
        print(f"     GT timestamp: {ts}")
        print(f"     GT indices: {indices}")
        
        # Check if this timestamp exists in data
        edges_at_ts = g_df[g_df['ts'] == ts]
        print(f"     Edges in data at timestamp {ts}: {len(edges_at_ts)}")
        
        # Check if GT timestamp is an edge index
        if ts < len(g_df):
            edge_row = g_df.iloc[ts]
            print(f"     If {ts} is edge index: u={edge_row.u}, v={edge_row.i}, ts={edge_row.ts}")
        
        # Check if indices point to actual edges
        print(f"     Checking GT indices as edge pointers:")
        for idx in indices[:3]:  # Check first 3 indices
            if idx < len(g_df):
                edge_row = g_df.iloc[idx]
                print(f"       Index {idx} → u={edge_row.u}, v={edge_row.i}, ts={edge_row.ts}")
    
    print(f"\n🧩 PATTERN ANALYSIS:")
    
    # Check if there's a pattern in GT timestamps
    gt_ts_sorted = sorted(gt_timestamps)
    print(f"   First 20 GT timestamps: {gt_ts_sorted[:20]}")
    
    # Check gaps
    gaps = []
    for i in range(1, min(20, len(gt_ts_sorted))):
        gap = gt_ts_sorted[i] - gt_ts_sorted[i-1]
        gaps.append(gap)
    print(f"   Gaps between consecutive GT timestamps: {gaps[:10]}")
    
    # Check if GT timestamps align with any data pattern
    print(f"\n🔗 ALIGNMENT CHECK:")
    
    # Method 1: Check if GT timestamps are shifted data timestamps
    data_ts = sorted(g_df['ts'].unique())
    potential_offsets = [50, 100, 1000]  # Common offset values
    
    for offset in potential_offsets:
        shifted_data_ts = [ts + offset for ts in data_ts]
        overlap = set(shifted_data_ts) & set(gt_timestamps)
        print(f"   Data timestamps + {offset} overlap with GT: {len(overlap)} matches")
        if len(overlap) > 0:
            print(f"     Sample overlapping timestamps: {sorted(list(overlap))[:5]}")
    
    # Method 2: Check if GT is using different indexing
    print(f"\n📈 EDGE DISTRIBUTION BY TIMESTAMP:")
    for ts in sorted(g_df['ts'].unique())[:10]:
        count = len(g_df[g_df['ts'] == ts])
        print(f"   Timestamp {ts}: {count} edges")
    
    print(f"\n💡 HYPOTHESIS SUMMARY:")
    print(f"   1. Data timestamps: 0-27 (sequential)")
    print(f"   2. GT timestamps: 50+ (different numbering)")
    print(f"   3. Possible causes:")
    print(f"      a) GT uses edge indices instead of timestamps")
    print(f"      b) GT uses different timestamp offset/scaling")
    print(f"      c) GT was generated with different parameters")
    print(f"      d) Data preprocessing changed timestamp format")

if __name__ == "__main__":
    investigate_timestamp_mismatch()
