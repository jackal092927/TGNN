#!/usr/bin/env python3
"""
Demonstration of Adaptive Weight Calculation for TGAM Autoregressive
Shows how adaptive weights are calculated from actual data distribution
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_class_distribution():
    """Analyze the actual positive/negative class distribution in the dataset"""
    
    print("🔍 Analyzing Class Distribution in Triadic Dataset")
    print("=" * 60)
    
    # Load dataset
    data = 'triadic_fixed'
    g_df = pd.read_csv(f'./processed/{data}/ml_{data}.csv')
    
    print(f"Dataset: {len(g_df)} total edges")
    print(f"Timestamps: {g_df.ts.min():.1f} to {g_df.ts.max():.1f}")
    
    # Use training split
    val_time = np.quantile(g_df.ts, 0.7)
    train_mask = g_df.ts <= val_time
    train_data = g_df[train_mask]
    
    train_src_l = train_data.u.values
    train_dst_l = train_data.i.values 
    train_ts_l = train_data.ts.values
    
    print(f"\nTraining edges: {len(train_src_l)}")
    
    # Simulate the autoregressive analysis
    max_timestamps = 10
    max_candidates_per_timestamp = 50
    
    # Group edges by timestamp (skip timestamp 0 as it's initial state)
    timestamp_groups = defaultdict(list)
    for i in range(len(train_ts_l)):
        if train_ts_l[i] > 0:  # Only predict edges after initial state
            timestamp_groups[train_ts_l[i]].append(i)
    
    unique_timestamps = sorted(timestamp_groups.keys())
    limited_timestamps = unique_timestamps[:max_timestamps]
    
    print(f"\nAnalyzing {len(limited_timestamps)} timestamps:")
    
    total_positives = 0
    total_negatives = 0
    timestamp_details = []
    
    for current_ts in limited_timestamps:
        # Get all nodes that are active up to this timestamp
        history_mask = train_ts_l < current_ts
        if not np.any(history_mask):
            continue
            
        # Find active nodes from history
        history_src = train_src_l[history_mask]
        history_dst = train_dst_l[history_mask]
        active_nodes = set(history_src) | set(history_dst)
        active_nodes = list(active_nodes)
        
        if len(active_nodes) < 2:
            continue
        
        # Get actual edges formed at this timestamp
        current_edges = timestamp_groups[current_ts]
        actual_edge_pairs = set()
        for edge_idx in current_edges:
            if edge_idx < len(train_src_l):
                actual_edge_pairs.add((train_src_l[edge_idx], train_dst_l[edge_idx]))
        
        # Count positives
        num_positives = len(actual_edge_pairs)
        total_positives += num_positives
        
        # Count potential negatives (using same sampling strategy as training)
        max_possible_edges = len(active_nodes) * (len(active_nodes) - 1)  # All directed pairs excluding self-loops
        actual_negative_sample_size = min(num_positives * 2, max_candidates_per_timestamp - num_positives)
        
        total_negatives += actual_negative_sample_size
        
        timestamp_details.append({
            'timestamp': current_ts,
            'active_nodes': len(active_nodes),
            'positives': num_positives,
            'negatives_sampled': actual_negative_sample_size,
            'max_possible': max_possible_edges,
            'ratio': actual_negative_sample_size / max(num_positives, 1)
        })
    
    # Show detailed breakdown
    print(f"\n📊 Timestamp Breakdown:")
    print(f"{'TS':<6} {'Nodes':<7} {'Pos':<5} {'Neg':<5} {'Max':<7} {'Ratio':<8}")
    print("-" * 45)
    for detail in timestamp_details[:8]:  # Show first 8 for brevity
        print(f"{detail['timestamp']:<6.1f} {detail['active_nodes']:<7} {detail['positives']:<5} "
              f"{detail['negatives_sampled']:<5} {detail['max_possible']:<7} {detail['ratio']:<8.2f}")
    
    if len(timestamp_details) > 8:
        print(f"... and {len(timestamp_details) - 8} more timestamps")
    
    # Calculate overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"  Total Positives: {total_positives}")
    print(f"  Total Negatives (sampled): {total_negatives}")
    
    if total_positives > 0:
        overall_ratio = total_negatives / total_positives
        adaptive_pos_weight = overall_ratio
        adaptive_neg_weight = 1.0
        
        print(f"  Class Imbalance Ratio (neg/pos): {overall_ratio:.2f}")
        print(f"  📐 ADAPTIVE WEIGHTS:")
        print(f"    Positive weight: {adaptive_pos_weight:.2f}")
        print(f"    Negative weight: {adaptive_neg_weight:.2f}")
        
        # Compare with common fixed ratios
        print(f"\n⚖️  Comparison with Fixed Ratios:")
        fixed_ratios = [1.0, 2.0, 3.0, 5.0]
        for fixed_ratio in fixed_ratios:
            diff = abs(fixed_ratio - adaptive_pos_weight)
            if diff < 0.5:
                status = "✅ Close match"
            elif diff < 1.0:
                status = "⚠️  Moderate difference"
            else:
                status = "❌ Poor match"
            print(f"    Fixed 1:{fixed_ratio:.1f} ratio → Pos weight {fixed_ratio:.1f} | {status} (diff: {diff:.2f})")
        
        print(f"\n💡 Insights:")
        if adaptive_pos_weight > 3.0:
            print(f"   → High class imbalance detected! Adaptive weighting is crucial.")
        elif adaptive_pos_weight > 1.5:
            print(f"   → Moderate class imbalance. Adaptive weighting provides good balance.")
        else:
            print(f"   → Relatively balanced classes. Fixed weights might also work.")
        
        print(f"   → Adaptive approach eliminates manual tuning of weight ratios")
        print(f"   → Weights automatically adjust to dataset characteristics")
        
    else:
        print(f"  ❌ No positive examples found")

def compare_weighting_strategies():
    """Compare different weighting strategies"""
    
    print(f"\n" + "=" * 60)
    print("🧪 Comparing Weighting Strategies")
    print("=" * 60)
    
    strategies = [
        {"name": "No Weighting", "pos_weight": 1.0, "neg_weight": 1.0},
        {"name": "Fixed 2:1", "pos_weight": 2.0, "neg_weight": 1.0},
        {"name": "Fixed 3:1", "pos_weight": 3.0, "neg_weight": 1.0},
        {"name": "Fixed 5:1", "pos_weight": 5.0, "neg_weight": 1.0},
        {"name": "Adaptive (Data-driven)", "pos_weight": "AUTO", "neg_weight": 1.0}
    ]
    
    print(f"{'Strategy':<25} {'Pos Weight':<12} {'Pros':<30} {'Cons'}")
    print("-" * 90)
    
    for strategy in strategies:
        pos_w = strategy['pos_weight']
        if strategy['name'] == "No Weighting":
            pros = "Simple, no hyperparameters"
            cons = "Ignores class imbalance"
        elif strategy['name'].startswith("Fixed"):
            pros = "Manual control, predictable"
            cons = "Requires tuning, dataset-specific"
        else:  # Adaptive
            pros = "Auto-tuned, generalizable"
            cons = "Slightly more computation"
        
        print(f"{strategy['name']:<25} {str(pos_w):<12} {pros:<30} {cons}")
    
    print(f"\n🎯 Recommendation:")
    print(f"   Use ADAPTIVE weighting for:")
    print(f"   ✅ New datasets with unknown class distribution")
    print(f"   ✅ Datasets with varying imbalance across timestamps")
    print(f"   ✅ Production systems requiring minimal hyperparameter tuning")
    print(f"   \n   Use FIXED weighting for:")
    print(f"   ⚙️  Fine-tuned research experiments")
    print(f"   ⚙️  When you have domain knowledge about optimal ratios")

if __name__ == "__main__":
    analyze_class_distribution()
    compare_weighting_strategies()
    
    print(f"\n🚀 Ready to test adaptive weighting:")
    print(f"   python run_autoregressive_gpu.py --gpu 0")
    print(f"   python test_individual_proper.py --gpu 0") 