"""
Fix the ground truth format by converting edge indices to timestamps
"""

import pandas as pd
import json
import numpy as np

def fix_ground_truth_format():
    data_name = 'triadic_perfect_long_dense'
    
    print("🔧 FIXING GROUND TRUTH FORMAT")
    print("=" * 50)
    
    # Load data
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth_raw = json.load(f)
    
    print(f"📊 Original format:")
    print(f"   GT entries: {len(ground_truth_raw)}")
    print(f"   Sample: {list(ground_truth_raw.items())[:3]}")
    
    # Convert ground truth from edge-index format to timestamp format
    ground_truth_by_timestamp = {}
    
    for edge_idx_str, parent_indices in ground_truth_raw.items():
        edge_idx = int(edge_idx_str)
        
        # Get the timestamp when this edge was added
        if edge_idx < len(g_df):
            edge_row = g_df.iloc[edge_idx]
            timestamp = edge_row.ts
            u, v = edge_row.u, edge_row.i
            
            # Initialize timestamp entry if not exists
            if timestamp not in ground_truth_by_timestamp:
                ground_truth_by_timestamp[timestamp] = []
            
            # Add this edge to the timestamp
            ground_truth_by_timestamp[timestamp].append((int(u), int(v)))
    
    print(f"\n✅ Fixed format:")
    print(f"   GT timestamps: {sorted(ground_truth_by_timestamp.keys())}")
    print(f"   Sample entries:")
    
    for ts in sorted(ground_truth_by_timestamp.keys())[:5]:
        edges = ground_truth_by_timestamp[ts]
        print(f"     Timestamp {ts}: {len(edges)} edges - {edges[:3]}{'...' if len(edges) > 3 else ''}")
    
    # Save the fixed ground truth
    output_file = f'./processed/{data_name}/ml_{data_name}_gt_fixed.json'
    
    # Convert to serializable format
    gt_serializable = {}
    for ts, edges in ground_truth_by_timestamp.items():
        gt_serializable[str(ts)] = edges
    
    with open(output_file, 'w') as f:
        json.dump(gt_serializable, f, indent=2)
    
    print(f"\n💾 Saved fixed ground truth to: {output_file}")
    
    # Verify the fix
    print(f"\n🔍 VERIFICATION:")
    data_timestamps = sorted(g_df['ts'].unique())
    gt_timestamps = sorted(ground_truth_by_timestamp.keys())
    
    print(f"   Data timestamps: {data_timestamps}")
    print(f"   Fixed GT timestamps: {gt_timestamps}")
    print(f"   Overlap: {set(data_timestamps) & set(gt_timestamps)}")
    print(f"   Perfect match: {data_timestamps == gt_timestamps}")
    
    return ground_truth_by_timestamp

if __name__ == "__main__":
    fixed_gt = fix_ground_truth_format()
