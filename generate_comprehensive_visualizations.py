#!/usr/bin/env python3
"""
Generate comprehensive event influence visualizations across multiple timestamps and edge pairs
for the triadic_perfect_long_dense dataset.
"""

import subprocess
import os
import pandas as pd
import numpy as np

def load_triadic_data():
    """Load the triadic dataset to find interesting edge events"""
    data_path = './processed/triadic_perfect_long_dense/ml_triadic_perfect_long_dense.csv'
    g_df = pd.read_csv(data_path)
    
    # Get unique timestamps
    timestamps = sorted(g_df['ts'].unique())
    print(f"Dataset has {len(timestamps)} timestamps: {timestamps}")
    
    # Find edges that appear at each timestamp
    edge_events = {}
    for ts in timestamps:
        edges_at_ts = g_df[g_df['ts'] == ts]
        edge_events[ts] = edges_at_ts[['u', 'i']].values.tolist()
        print(f"Timestamp {ts}: {len(edges_at_ts)} edges")
    
    return g_df, timestamps, edge_events

def generate_visualizations():
    """Generate visualizations for multiple timestamps and edge pairs"""
    g_df, timestamps, edge_events = load_triadic_data()
    
    # Create output directory
    output_dir = './results_triadic_long_dense/comprehensive_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Select interesting timestamps for visualization
    # We want to cover early, middle, and late stages of the network evolution
    selected_timestamps = [5, 10, 15, 20, 25]  # Cover the range
    
    # For each timestamp, select some edge pairs to visualize
    visualization_configs = []
    
    for ts in selected_timestamps:
        if ts in timestamps:
            # Get edges that appear at this timestamp
            edges_at_ts = edge_events[ts]
            
            # Select a few representative edges for visualization
            if len(edges_at_ts) > 0:
                # Take first few edges, but ensure we have variety
                selected_edges = edges_at_ts[:min(3, len(edges_at_ts))]
                
                for u, v in selected_edges:
                    # Use timestamp index (ts-1) since we want to predict edges at ts
                    t_index = timestamps.index(ts) - 1
                    if t_index >= 0:  # Make sure we have a valid previous timestamp
                        visualization_configs.append({
                            't_index': t_index,
                            'u': int(u),
                            'v': int(v),
                            'timestamp': ts,
                            'description': f"Predicting edge ({u},{v}) at timestamp {ts}"
                        })
    
    print(f"\nGenerated {len(visualization_configs)} visualization configurations:")
    for config in visualization_configs:
        print(f"  t_index={config['t_index']}, u={config['u']}, v={config['v']}: {config['description']}")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    successful_viz = 0
    
    for i, config in enumerate(visualization_configs):
        print(f"\n[{i+1}/{len(visualization_configs)}] Generating visualization for {config['description']}")
        
        try:
            # Run the visualization script
            cmd = [
                '/common/home/cx122/miniconda3/envs/tgib/bin/python',
                'visualize_event_influence.py',
                '--data', 'triadic_perfect_long_dense',
                '--ckpt', './results_triadic_long_dense/triadic_perfect_long_dense_best_model.pth',
                '--t_index', str(config['t_index']),
                '--u', str(config['u']),
                '--v', str(config['v']),
                '--hidden_dim', '128',
                '--pos_dim', '128',
                '--mamba_state_dim', '16',
                '--gnn_layers', '2',
                '--out_dir', output_dir,
                '--topk', '15'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✅ Success: {config['description']}")
                successful_viz += 1
            else:
                print(f"  ❌ Failed: {config['description']}")
                print(f"    Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout: {config['description']}")
        except Exception as e:
            print(f"  💥 Exception: {config['description']}")
            print(f"    Error: {str(e)}")
    
    print(f"\n🎉 Visualization generation complete!")
    print(f"   Successful: {successful_viz}/{len(visualization_configs)}")
    print(f"   Output directory: {output_dir}")
    
    # List generated files
    if os.path.exists(output_dir):
        print(f"\n📁 Generated files:")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"   ✅ {file_path} - {file_size/1024/1024:.1f} MB")

if __name__ == "__main__":
    generate_visualizations()
