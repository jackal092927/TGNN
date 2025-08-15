#!/usr/bin/env python3
"""
Debug Candidate Generation in Autoregressive Evaluation
=======================================================

This script analyzes the exact candidate ratios and prediction patterns
to understand why the model appears to be doing trivial prediction.
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict
sys.path.append('.')

from train_tgam_proper_eval import train_tgam_proper_eval
from utils import RandEdgeSampler

def debug_candidate_generation():
    """Debug the candidate generation process"""
    
    print("🔍 Debugging Candidate Generation")
    print("=" * 50)
    
    # Load data exactly like in training
    data = 'triadic_fixed'
    g_df = pd.read_csv(f'./processed/{data}/ml_{data}.csv')
    
    # Use the corrected uniform timeline splits
    timeline_start = g_df.ts.min()
    timeline_end = g_df.ts.max()
    total_timeline = timeline_end - timeline_start
    
    train_time = timeline_start + (total_timeline * 0.6)
    val_time = timeline_start + (total_timeline * 0.8)
    
    val_mask = (g_df.ts > train_time) & (g_df.ts <= val_time)
    val_data = g_df[val_mask]
    
    print(f"Validation data: {len(val_data)} edges")
    print(f"Validation timestamps: {sorted(val_data.ts.unique())}")
    print()
    
    # Simulate candidate generation for validation
    val_src_l = val_data.u.values
    val_dst_l = val_data.i.values
    val_ts_l = val_data.ts.values
    
    # Group edges by timestamp
    timestamp_groups = defaultdict(list)
    for i in range(len(val_ts_l)):
        ts = val_ts_l[i]
        timestamp_groups[ts].append(i)
    
    # Create sampler
    sampler = RandEdgeSampler(val_src_l, val_dst_l)
    
    print("Candidate generation analysis:")
    print()
    
    for ts in sorted(timestamp_groups.keys()):
        edges_at_ts = timestamp_groups[ts]
        print(f"Timestamp {ts}:")
        print(f"  Ground truth edges: {len(edges_at_ts)}")
        
        # Simulate candidate generation
        candidates = []
        ground_truth_edges = set()
        
        # Add positive candidates
        for edge_idx in edges_at_ts:
            src_node = val_src_l[edge_idx]
            dst_node = val_dst_l[edge_idx]
            candidates.append((src_node, dst_node, 1))
            ground_truth_edges.add((src_node, dst_node))
        
        # Add negative candidates (2:1 ratio)
        num_negatives = len(edges_at_ts) * 2
        print(f"  Target negatives: {num_negatives}")
        
        negatives_added = 0
        attempts = 0
        max_attempts = num_negatives * 10  # Prevent infinite loop
        
        while negatives_added < num_negatives and attempts < max_attempts:
            try:
                u_fake, i_fake = sampler.sample(1)
                neg_src, neg_dst = u_fake[0], i_fake[0]
                if (neg_src, neg_dst) not in ground_truth_edges:
                    candidates.append((neg_src, neg_dst, 0))
                    negatives_added += 1
            except:
                pass
            attempts += 1
        
        # Analyze final candidates
        pos_count = sum(1 for _, _, label in candidates if label == 1)
        neg_count = sum(1 for _, _, label in candidates if label == 0)
        total_candidates = len(candidates)
        
        print(f"  Final candidates: {total_candidates} ({pos_count} pos, {neg_count} neg)")
        print(f"  Actual ratio: {neg_count}:{pos_count} = {neg_count/pos_count:.2f}:1")
        
        # Simulate trivial predictions
        if total_candidates > 0:
            acc_all_positive = pos_count / total_candidates
            acc_all_negative = neg_count / total_candidates
            print(f"  If predict ALL POSITIVE: accuracy = {acc_all_positive:.4f}")
            print(f"  If predict ALL NEGATIVE: accuracy = {acc_all_negative:.4f}")
        print()

if __name__ == '__main__':
    debug_candidate_generation() 