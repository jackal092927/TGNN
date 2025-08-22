#!/usr/bin/env python3
"""
Analyze influence score coverage for positive node pairs at each timestamp.
For each timestamp t:
1. Rank all existing edges from (t-1) according to influence scores
2. Find rank threshold that covers >=95% of positive node pairs at t
3. Record coverage statistics and thresholds
"""

import os
import json
import csv
import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Dict, Set
from collections import defaultdict

# Import your model and influence computation
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graphmamba_explain import GraphMamba
from visualize_event_influence import per_event_influence_grad_x_gate

def load_triadic_data():
    """Load triadic dataset"""
    data_path = './processed/triadic_perfect_long_dense/ml_triadic_perfect_long_dense.csv'
    g_df = pd.read_csv(data_path)
    return g_df

def build_graph_sequence(g_df: pd.DataFrame, timestamps: List[float], device: torch.device):
    """Build adjacency matrices for each timestamp"""
    max_node = int(max(g_df['u'].max(), g_df['i'].max())) + 1
    seq = []
    for ts in timestamps:
        A = torch.zeros(max_node, max_node, device=device)
        cur = g_df[g_df['ts'] <= ts]
        for _, r in cur.iterrows():
            u, v = int(r['u']), int(r['i'])
            A[u, v] = 1.0
            A[v, u] = 1.0
        seq.append(A)
    return seq, max_node

def get_positive_pairs_at_timestamp(g_df: pd.DataFrame, t: float) -> Set[Tuple[int, int]]:
    """Get all positive node pairs (u,v) that appear at timestamp t"""
    edges_at_t = g_df[g_df['ts'] == t]
    positive_pairs = set()
    for _, row in edges_at_t.iterrows():
        u, v = int(row['u']), int(row['i'])
        positive_pairs.add((min(u, v), max(u, v)))  # Ensure consistent ordering
    return positive_pairs

def get_existing_edges_at_timestamp(graph_sequence: List[torch.Tensor], t: int) -> Set[Tuple[int, int]]:
    """Get all existing edges at timestamp t-1 (or t=0 if t=0)"""
    if t == 0:
        return set()  # No existing edges at t=0
    
    A_prev = graph_sequence[t-1]
    existing_edges = set()
    N = A_prev.shape[0]
    
    for i in range(N):
        for j in range(i+1, N):  # Upper triangular only
            if float(A_prev[i, j]) > 0.0:
                existing_edges.add((i, j))
    
    return existing_edges

def compute_influence_scores_for_all_edges(model: GraphMamba, 
                                         graph_sequence: List[torch.Tensor], 
                                         t: int,
                                         device: torch.device) -> List[Tuple[Tuple[int, int], float]]:
    """
    Compute influence scores for all existing edges at timestamp t-1.
    Returns list of ((u,v), influence_score) sorted by score descending.
    """
    if t == 0:
        return []  # No existing edges to rank
    
    existing_edges = get_existing_edges_at_timestamp(graph_sequence, t)
    if not existing_edges:
        return []
    
    print(f"  Computing influence scores for {len(existing_edges)} existing edges...")
    
    # For each existing edge, compute its influence on a "dummy" target
    # We'll use the first positive pair at timestamp t as the target
    positive_pairs = get_positive_pairs_at_timestamp(
        pd.read_csv('./processed/triadic_perfect_long_dense/ml_triadic_perfect_long_dense.csv'), 
        float(t)
    )
    
    if not positive_pairs:
        print(f"  Warning: No positive pairs at timestamp {t}")
        return []
    
    # Use the first positive pair as target for influence computation
    target_u, target_v = list(positive_pairs)[0]
    
    # Compute influence matrix for this target
    try:
        influence_matrix, _, _ = per_event_influence_grad_x_gate(
            model, graph_sequence, t-1, target_u, target_v, mask_to_edges=True
        )
        
        # Extract influence scores for existing edges
        edge_influences = []
        for (u, v) in existing_edges:
            score = float(influence_matrix[u, v].cpu())
            edge_influences.append(((u, v), score))
        
        # Sort by influence score (descending)
        edge_influences.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Computed influence scores for {len(edge_influences)} edges")
        return edge_influences
        
    except Exception as e:
        print(f"  Error computing influence scores: {e}")
        return []

def check_triadic_coverage(ranked_edges: List[Tuple[Tuple[int, int], float]], 
                          positive_pairs: Set[Tuple[int, int]], 
                          rank_threshold: int) -> Tuple[int, float]:
    """
    Check how many positive pairs are covered by top-k ranked edges.
    A pair (u,v) is covered if there exists w such that both (u,w) and (v,w) are in top-k.
    """
    if rank_threshold == 0:
        return 0, 0.0
    
    # Get top-k edges
    top_edges = set(ranked_edges[:rank_threshold])
    top_edge_set = set(edge for edge, _ in top_edges)
    
    # Build adjacency from top edges
    top_adj = defaultdict(set)
    for (u, v) in top_edge_set:
        top_adj[u].add(v)
        top_adj[v].add(u)
    
    # Check coverage for each positive pair
    covered_pairs = 0
    total_pairs = len(positive_pairs)
    
    for (u, v) in positive_pairs:
        # Check if there exists w such that both (u,w) and (v,w) are in top edges
        u_neighbors = top_adj[u]
        v_neighbors = top_adj[v]
        common_neighbors = u_neighbors.intersection(v_neighbors)
        
        if common_neighbors:  # At least one common neighbor
            covered_pairs += 1
    
    coverage_rate = covered_pairs / total_pairs if total_pairs > 0 else 0.0
    return covered_pairs, coverage_rate

def find_rank_threshold_for_95_percent_coverage(ranked_edges: List[Tuple[Tuple[int, int], float]], 
                                              positive_pairs: Set[Tuple[int, int]]) -> int:
    """
    Find the minimum rank threshold that achieves >=95% coverage of positive pairs.
    """
    if not ranked_edges or not positive_pairs:
        return 0
    
    # Binary search for the optimal threshold
    left, right = 0, len(ranked_edges)
    optimal_threshold = len(ranked_edges)  # Default to all edges
    
    while left <= right:
        mid = (left + right) // 2
        _, coverage_rate = check_triadic_coverage(ranked_edges, positive_pairs, mid)
        
        if coverage_rate >= 0.95:  # 95% coverage achieved
            optimal_threshold = mid
            right = mid - 1  # Try to find smaller threshold
        else:
            left = mid + 1  # Need more edges
    
    return optimal_threshold

def analyze_timestamp_coverage(model: GraphMamba, 
                             graph_sequence: List[torch.Tensor], 
                             g_df: pd.DataFrame, 
                             t: int,
                             device: torch.device) -> Dict:
    """
    Analyze coverage for a specific timestamp t.
    """
    print(f"\n=== Analyzing Timestamp {t} ===")
    
    # Get positive pairs at current timestamp
    positive_pairs = get_positive_pairs_at_timestamp(g_df, float(t))
    print(f"  Positive pairs at t={t}: {len(positive_pairs)}")
    
    # Get existing edges from previous timestamp
    existing_edges = get_existing_edges_at_timestamp(graph_sequence, t)
    print(f"  Existing edges at t-1: {len(existing_edges)}")
    
    if t == 0:
        # Special case: no existing edges at t=0
        return {
            'timestamp': t,
            'positive_pairs': len(positive_pairs),
            'existing_edges_prev': 0,
            'rank_threshold': 0,
            'coverage_rate': 0.0,
            'selected_edges_percentage': 0.0,
            'coverage_achieved': 0,
            'total_positive': len(positive_pairs)
        }
    
    # Compute influence scores for existing edges
    ranked_edges = compute_influence_scores_for_all_edges(model, graph_sequence, t, device)
    
    if not ranked_edges:
        print(f"  Warning: No influence scores computed for t={t}")
        return {
            'timestamp': t,
            'positive_pairs': len(positive_pairs),
            'existing_edges_prev': len(existing_edges),
            'rank_threshold': 0,
            'coverage_rate': 0.0,
            'selected_edges_percentage': 0.0,
            'coverage_achieved': 0,
            'total_positive': len(positive_pairs)
        }
    
    # Find rank threshold for 95% coverage
    rank_threshold = find_rank_threshold_for_95_percent_coverage(ranked_edges, positive_pairs)
    
    # Verify coverage
    covered_pairs, actual_coverage = check_triadic_coverage(ranked_edges, positive_pairs, rank_threshold)
    
    # Calculate percentage of selected edges
    selected_edges_percentage = (rank_threshold / len(existing_edges) * 100) if existing_edges else 0.0
    
    print(f"  Rank threshold for 95% coverage: {rank_threshold}")
    print(f"  Actual coverage achieved: {actual_coverage:.3f} ({covered_pairs}/{len(positive_pairs)})")
    print(f"  Selected edges: {rank_threshold}/{len(existing_edges)} ({selected_edges_percentage:.1f}%)")
    
    return {
        'timestamp': t,
        'positive_pairs': len(positive_pairs),
        'existing_edges_prev': len(existing_edges),
        'rank_threshold': rank_threshold,
        'coverage_rate': actual_coverage,
        'selected_edges_percentage': selected_edges_percentage,
        'coverage_achieved': covered_pairs,
        'total_positive': len(positive_pairs)
    }

def main():
    """Main analysis function"""
    print("=== INFLUENCE SCORE COVERAGE ANALYSIS ===\n")
    
    # Load data
    g_df = load_triadic_data()
    timestamps = sorted(g_df['ts'].unique())
    print(f"Dataset spans {len(timestamps)} timestamps: {timestamps}")
    
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build graph sequence
    graph_sequence, max_nodes = build_graph_sequence(g_df, timestamps, device)
    
    # Load the trained model
    ckpt_path = './results_triadic_long_dense/triadic_perfect_long_dense_best_model.pth'
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}")
        return
    
    model = GraphMamba(
        max_nodes=max_nodes,
        pos_dim=128,
        hidden_dim=128,
        gnn_layers=2,
        mamba_state_dim=16,
        dropout=0.1,
        use_edge_gates=True,
        gate_temperature=1.0
    ).to(device)
    
    # Load checkpoint
    sd = torch.load(ckpt_path, map_location=device)
    state_dict = sd.get("model_state", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Analyze each timestamp
    results = []
    for t in range(len(timestamps)):
        result = analyze_timestamp_coverage(model, graph_sequence, g_df, t, device)
        results.append(result)
    
    # Save results
    output_dir = './results_triadic_long_dense'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'influence_coverage_analysis.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'positive_pairs', 'existing_edges_prev', 'rank_threshold',
            'coverage_rate', 'selected_edges_percentage', 'coverage_achieved', 'total_positive'
        ])
        for result in results:
            writer.writerow([
                result['timestamp'], result['positive_pairs'], result['existing_edges_prev'],
                result['rank_threshold'], result['coverage_rate'], result['selected_edges_percentage'],
                result['coverage_achieved'], result['total_positive']
            ])
    
    # Save JSON summary
    json_path = os.path.join(output_dir, 'influence_coverage_summary.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    
    # Print key insights
    print(f"\n=== KEY INSIGHTS ===")
    total_positive = sum(r['positive_pairs'] for r in results)
    avg_coverage = np.mean([r['coverage_rate'] for r in results if r['coverage_rate'] > 0])
    avg_selection = np.mean([r['selected_edges_percentage'] for r in results if r['selected_edges_percentage'] > 0])
    
    print(f"Total positive pairs across all timestamps: {total_positive}")
    print(f"Average coverage rate: {avg_coverage:.3f}")
    print(f"Average edge selection percentage: {avg_selection:.1f}%")
    
    # Find best and worst performing timestamps
    valid_results = [r for r in results if r['coverage_rate'] > 0]
    if valid_results:
        best_ts = max(valid_results, key=lambda x: x['coverage_rate'])
        worst_ts = min(valid_results, key=lambda x: x['coverage_rate'])
        print(f"Best coverage at t={best_ts['timestamp']}: {best_ts['coverage_rate']:.3f}")
        print(f"Worst coverage at t={worst_ts['timestamp']}: {worst_ts['coverage_rate']:.3f}")

if __name__ == "__main__":
    main()
