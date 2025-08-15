"""
Visualize specific triadic closure predictions to understand why rule-based isn't perfect
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict


def load_and_visualize_timestamp(data_name='triadic_medium', target_ts=17):
    """
    Visualize what happens at a specific timestamp
    """
    print(f"=== VISUALIZING TIMESTAMP {target_ts} IN {data_name} ===\n")
    
    # Load data
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    # Get edges before target timestamp
    before_edges = g_df[g_df.ts < target_ts]
    at_timestamp = g_df[g_df.ts == target_ts]
    
    print(f"Edges before timestamp {target_ts}:")
    edge_list = []
    for _, row in before_edges.iterrows():
        print(f"  {row.u} -- {row.i} (ts={row.ts}, idx={row.idx})")
        edge_list.append((row.u, row.i))
    
    print(f"\nActual new edge at timestamp {target_ts}:")
    actual_new_edge = None
    for _, row in at_timestamp.iterrows():
        print(f"  {row.u} -- {row.i} (idx={row.idx})")
        actual_new_edge = (min(row.u, row.i), max(row.u, row.i))
        
        # Check if it's a ground truth triadic closure
        if str(row.idx) in ground_truth:
            parent_edges = ground_truth[str(row.idx)]
            print(f"    -> This IS a ground truth triadic closure!")
            print(f"    -> Parent edges: {parent_edges}")
            
            # Find parent edges
            for parent_idx in parent_edges:
                parent_row = g_df[g_df.idx == parent_idx]
                if not parent_row.empty:
                    pu, pv = parent_row.iloc[0]['u'], parent_row.iloc[0]['i']
                    print(f"       Parent {parent_idx}: {pu} -- {pv} (ts={parent_row.iloc[0]['ts']})")
        else:
            print(f"    -> This is NOT a ground truth triadic closure")
    
    # Build adjacency list
    adj = defaultdict(set)
    existing_edges = set()
    
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
        existing_edges.add((min(u, v), max(u, v)))
    
    print(f"\nCurrent graph adjacency:")
    for node in sorted(adj.keys()):
        if adj[node]:
            neighbors = sorted(list(adj[node]))
            print(f"  Node {node}: connected to {neighbors}")
    
    # Apply triadic closure rule
    print(f"\nApplying triadic closure rule:")
    print("Looking for patterns: A-B, B-C exist -> predict A-C")
    
    candidates = {}
    all_nodes = set()
    for u, v in edge_list:
        all_nodes.add(u)
        all_nodes.add(v)
    
    for shared_node in sorted(all_nodes):
        neighbors = sorted(list(adj[shared_node]))
        
        if len(neighbors) >= 2:
            print(f"\nShared node {shared_node} has neighbors: {neighbors}")
            
            # Check all pairs of neighbors
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    node_a, node_b = neighbors[i], neighbors[j]
                    edge_key = (min(node_a, node_b), max(node_a, node_b))
                    
                    # If A-B edge doesn't exist, it's a candidate
                    if edge_key not in existing_edges:
                        shared_neighbors = adj[node_a] & adj[node_b]
                        confidence = len(shared_neighbors) / max(1, len(adj[node_a]) + len(adj[node_b]) - len(shared_neighbors))
                        
                        print(f"  -> CANDIDATE: {node_a} -- {node_b} (shared via {shared_node})")
                        print(f"     Confidence: {confidence:.4f}")
                        print(f"     Node {node_a} neighbors: {sorted(list(adj[node_a]))}")
                        print(f"     Node {node_b} neighbors: {sorted(list(adj[node_b]))}")
                        print(f"     Total shared neighbors: {len(shared_neighbors)} {sorted(list(shared_neighbors))}")
                        
                        if edge_key == actual_new_edge:
                            print(f"     *** THIS IS THE ACTUAL NEW EDGE! ***")
                        else:
                            print(f"     (This edge does not actually form)")
                        
                        candidates[edge_key] = (shared_node, confidence)
    
    print(f"\nSUMMARY:")
    print(f"  Rule-based candidates: {len(candidates)}")
    print(f"  Actual new edge: {actual_new_edge}")
    print(f"  Rule correctly predicted: {actual_new_edge in candidates}")
    print(f"  False positives: {len(candidates) - (1 if actual_new_edge in candidates else 0)}")
    
    print(f"\nWHY FALSE POSITIVES OCCUR:")
    print("  - The rule identifies ALL possible triadic closures")
    print("  - But only ONE edge actually forms at this timestamp")
    print("  - The dataset has temporal/causal constraints the rule doesn't know")
    print("  - Multiple competing triadic opportunities exist simultaneously")


if __name__ == "__main__":
    load_and_visualize_timestamp('triadic_medium', 17)
    print("\n" + "="*60)
    load_and_visualize_timestamp('triadic_medium', 20)
