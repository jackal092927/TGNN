"""
Analyze why the rule-based triadic closure method isn't perfect

This script investigates:
1. Why rule-based predictions fail (false positives)
2. Why some triadic closures are missed (false negatives) 
3. The nature of the triadic closure dataset
"""

import pandas as pd
import numpy as np
import json
import time
from collections import defaultdict, Counter


def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    # Load ground truth
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    return g_df, e_feat, n_feat, ground_truth


def get_edges_at_timestamp(g_df, timestamp):
    """Get all edges up to and including given timestamp"""
    edges_df = g_df[g_df.ts <= timestamp]
    
    edge_list = []
    edge_indices = []
    
    for _, row in edges_df.iterrows():
        edge_list.append((row.u, row.i))
        edge_indices.append(row.idx)
    
    return edge_list, edge_indices


def find_triadic_closures_detailed(edge_list, all_nodes):
    """
    Find triadic closure candidates with detailed reasoning
    """
    # Build adjacency list
    adj = defaultdict(set)
    existing_edges = set()
    
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
        existing_edges.add((min(u, v), max(u, v)))
    
    candidates = {}
    candidate_reasons = {}  # Track why each candidate was suggested
    
    # Apply triadic closure rule
    for shared_node in all_nodes:
        neighbors = list(adj[shared_node])
        
        # For each pair of neighbors of shared_node
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                node_a, node_b = neighbors[i], neighbors[j]
                edge_key = (min(node_a, node_b), max(node_a, node_b))
                
                # If A-B edge doesn't exist, it's a triadic closure candidate
                if edge_key not in existing_edges:
                    # Calculate confidence based on number of shared neighbors
                    shared_neighbors = adj[node_a] & adj[node_b]
                    confidence = len(shared_neighbors) / max(1, len(adj[node_a]) + len(adj[node_b]) - len(shared_neighbors))
                    
                    # Track all reasons (shared neighbors) for this candidate
                    if edge_key not in candidate_reasons:
                        candidate_reasons[edge_key] = []
                    candidate_reasons[edge_key].append({
                        'shared_neighbor': shared_node,
                        'confidence': confidence,
                        'node_a_degree': len(adj[node_a]),
                        'node_b_degree': len(adj[node_b]),
                        'total_shared_neighbors': len(shared_neighbors)
                    })
                    
                    # Store the best (highest confidence) prediction for this edge
                    if edge_key not in candidates or confidence > candidates[edge_key][1]:
                        candidates[edge_key] = (shared_node, confidence)
    
    return candidates, candidate_reasons


def analyze_rule_based_detailed(data_name='triadic_medium'):
    """
    Detailed analysis of rule-based predictions
    """
    print(f"=== DETAILED ANALYSIS OF RULE-BASED METHOD ({data_name}) ===\n")
    
    # Load data
    g_df, e_feat, n_feat, ground_truth = load_triadic_data(data_name)
    print(f"Dataset: {len(g_df)} edges, {len(n_feat)} nodes")
    print(f"Ground truth triadic closures: {len(ground_truth)}")
    
    # Convert ground truth to edge mapping
    gt_edge_to_parents = {}
    for edge_idx_str, parent_edges in ground_truth.items():
        edge_idx = int(edge_idx_str)
        # Find the edge in g_df
        edge_row = g_df[g_df.idx == edge_idx]
        if not edge_row.empty:
            u, v = edge_row.iloc[0]['u'], edge_row.iloc[0]['i']
            edge_key = (min(u, v), max(u, v))
            gt_edge_to_parents[edge_key] = {
                'parent_edges': parent_edges,
                'timestamp': edge_row.iloc[0]['ts'],
                'edge_idx': edge_idx
            }
    
    print(f"Ground truth edges mapped: {len(gt_edge_to_parents)}")
    
    # Analyze test timestamps
    all_nodes = set(range(len(n_feat)))
    max_timestamp = int(g_df.ts.max())
    val_end = int(max_timestamp * 0.8)
    test_timestamps = list(range(val_end + 1, max_timestamp + 1))
    
    print(f"Analyzing test timestamps: {test_timestamps}\n")
    
    total_rule_predictions = 0
    total_correct_predictions = 0
    total_actual_edges = 0
    total_missed_triadic_closures = 0
    
    false_positives = []
    false_negatives = []
    true_positives = []
    
    for test_ts in test_timestamps:
        print(f"\n--- TIMESTAMP {test_ts} ---")
        
        # Get current edges (up to timestamp-1)
        current_edges, _ = get_edges_at_timestamp(g_df, test_ts - 1)
        print(f"Existing edges before ts {test_ts}: {len(current_edges)}")
        
        # Get actual new edges at this timestamp
        new_edges_df = g_df[g_df.ts == test_ts]
        actual_new_edges = set()
        actual_triadic_closures = set()
        
        for _, row in new_edges_df.iterrows():
            edge_key = (min(row.u, row.i), max(row.u, row.i))
            actual_new_edges.add(edge_key)
            
            # Check if this is a ground truth triadic closure
            if edge_key in gt_edge_to_parents:
                actual_triadic_closures.add(edge_key)
        
        print(f"Actual new edges at ts {test_ts}: {len(actual_new_edges)}")
        print(f"Actual triadic closures at ts {test_ts}: {len(actual_triadic_closures)}")
        
        total_actual_edges += len(actual_new_edges)
        
        # Get rule-based predictions
        candidates, candidate_reasons = find_triadic_closures_detailed(current_edges, all_nodes)
        print(f"Rule-based candidates: {len(candidates)}")
        
        total_rule_predictions += len(candidates)
        
        # Analyze predictions
        for edge_key, (shared_neighbor, confidence) in candidates.items():
            if edge_key in actual_new_edges:
                total_correct_predictions += 1
                true_positives.append({
                    'edge': edge_key,
                    'timestamp': test_ts,
                    'shared_neighbor': shared_neighbor,
                    'confidence': confidence,
                    'is_gt_triadic': edge_key in actual_triadic_closures,
                    'reasons': candidate_reasons[edge_key]
                })
            else:
                false_positives.append({
                    'edge': edge_key,
                    'timestamp': test_ts,
                    'shared_neighbor': shared_neighbor,
                    'confidence': confidence,
                    'reasons': candidate_reasons[edge_key]
                })
        
        # Find missed triadic closures
        for edge_key in actual_triadic_closures:
            if edge_key not in candidates:
                total_missed_triadic_closures += 1
                false_negatives.append({
                    'edge': edge_key,
                    'timestamp': test_ts,
                    'gt_info': gt_edge_to_parents[edge_key]
                })
        
        print(f"Correct predictions: {sum(1 for e in candidates if e in actual_new_edges)}")
        print(f"False positives: {sum(1 for e in candidates if e not in actual_new_edges)}")
        print(f"Missed triadic closures: {sum(1 for e in actual_triadic_closures if e not in candidates)}")
    
    # Summary analysis
    print(f"\n{'='*60}")
    print("OVERALL ANALYSIS")
    print(f"{'='*60}")
    print(f"Total rule predictions: {total_rule_predictions}")
    print(f"Total correct predictions: {total_correct_predictions}")
    print(f"Total actual edges: {total_actual_edges}")
    print(f"Total missed triadic closures: {total_missed_triadic_closures}")
    print(f"Rule precision: {total_correct_predictions/total_rule_predictions:.4f}")
    print(f"Rule recall (on all edges): {total_correct_predictions/total_actual_edges:.4f}")
    
    # Analyze false positives
    print(f"\n--- FALSE POSITIVES ANALYSIS ({len(false_positives)} cases) ---")
    if false_positives:
        confidence_dist = [fp['confidence'] for fp in false_positives]
        print(f"Confidence distribution: mean={np.mean(confidence_dist):.4f}, std={np.std(confidence_dist):.4f}")
        
        # Analyze reasons for false positives
        reason_counts = Counter()
        for fp in false_positives:
            for reason in fp['reasons']:
                reason_counts[f"shared_neighbor_{reason['shared_neighbor']}"] += 1
        
        print("Top shared neighbors causing false positives:")
        for neighbor, count in reason_counts.most_common(5):
            print(f"  {neighbor}: {count} times")
    
    # Analyze true positives  
    print(f"\n--- TRUE POSITIVES ANALYSIS ({len(true_positives)} cases) ---")
    if true_positives:
        gt_triadic_count = sum(1 for tp in true_positives if tp['is_gt_triadic'])
        print(f"True positives that are ground truth triadic closures: {gt_triadic_count}/{len(true_positives)}")
        
        confidence_dist = [tp['confidence'] for tp in true_positives]
        print(f"Confidence distribution: mean={np.mean(confidence_dist):.4f}, std={np.std(confidence_dist):.4f}")
    
    # Analyze false negatives (missed triadic closures)
    print(f"\n--- FALSE NEGATIVES ANALYSIS ({len(false_negatives)} cases) ---")
    if false_negatives:
        print("Why were these triadic closures missed by the rule?")
        
        for i, fn in enumerate(false_negatives[:5]):  # Show first 5 examples
            edge_key = fn['edge']
            gt_info = fn['gt_info']
            timestamp = fn['timestamp']
            
            print(f"\nExample {i+1}: Edge {edge_key} at timestamp {timestamp}")
            print(f"  Ground truth parent edges: {gt_info['parent_edges']}")
            
            # Check if parent edges existed before this timestamp
            current_edges, _ = get_edges_at_timestamp(g_df, timestamp - 1)
            current_edge_set = set((min(u, v), max(u, v)) for u, v in current_edges)
            
            # Find parent edges in the dataset
            parent_edges_found = []
            for parent_idx in gt_info['parent_edges']:
                parent_row = g_df[g_df.idx == parent_idx]
                if not parent_row.empty:
                    pu, pv = parent_row.iloc[0]['u'], parent_row.iloc[0]['i']
                    parent_key = (min(pu, pv), max(pu, pv))
                    parent_ts = parent_row.iloc[0]['ts']
                    parent_edges_found.append({
                        'edge': parent_key,
                        'timestamp': parent_ts,
                        'existed_before': parent_key in current_edge_set
                    })
            
            print(f"  Parent edges analysis:")
            for pe in parent_edges_found:
                print(f"    {pe['edge']} at ts {pe['timestamp']}: {'existed' if pe['existed_before'] else 'NOT existed'}")
    
    return {
        'total_rule_predictions': total_rule_predictions,
        'total_correct_predictions': total_correct_predictions,
        'total_actual_edges': total_actual_edges,
        'total_missed_triadic_closures': total_missed_triadic_closures,
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'true_positives': len(true_positives)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Rule-based Failures')
    parser.add_argument('--data', type=str, default='triadic_medium')
    
    args = parser.parse_args()
    
    print("INVESTIGATING: Why isn't the rule-based method perfect?")
    print("="*60)
    
    results = analyze_rule_based_detailed(data_name=args.data)
    
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}")
    print("The rule-based method isn't perfect because:")
    print("1. FALSE POSITIVES: Rule suggests edges that don't actually form")
    print("2. FALSE NEGATIVES: Some triadic closures are missed by the rule")  
    print("3. TEMPORAL CONSTRAINTS: Parent edges might not exist at prediction time")
    print("4. COMPETING PREDICTIONS: Multiple shared neighbors suggest same edge")
    print("5. DATASET NOISE: Not all triadic opportunities result in edges")
