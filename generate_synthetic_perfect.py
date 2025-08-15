"""
Generate perfect triadic closure data without random node sampling limitations.

This version removes the artificial 20-node sampling constraint to create
a dataset where ALL structurally valid triadic closures are realized.
"""

import os
import random
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def generate_perfect_triadic_data(
    num_nodes=100,
    num_initial_edges=100,
    num_timesteps=500,
    noise_ratio=0.0,  # Default to no noise for perfect triadic closure
    node_feat_dim=100,
    edge_feat_dim=100,
    data_name="triadic_perfect"
):
    """
    Generates a perfect triadic closure dataset without random sampling limitations.
    
    Key differences from original:
    1. Checks ALL nodes for triadic opportunities (not just 20 random)
    2. Creates ALL valid triadic closures at each timestep
    3. Optional noise can be disabled for pure triadic closure behavior
    
    Args:
        num_nodes (int): Total number of nodes in the graph.
        num_initial_edges (int): Number of random seed edges to start with.
        num_timesteps (int): Number of simulation steps.
        noise_ratio (float): Proportion of timesteps that get random noise edges.
        node_feat_dim (int): Dimension of node features.
        edge_feat_dim (int): Dimension of edge features.
        data_name (str): Name of the dataset, used for directory and file naming.
    """
    print(f"Generating perfect triadic closure data: {data_name}")
    print(f"Key improvement: Checking ALL {num_nodes} nodes (not just 20 random)")
    
    # Initialize nodes and adjacency list for tracking neighbors
    nodes = set(range(num_nodes))
    adj = defaultdict(set)
    edges = []
    edge_idx_counter = 0
    ground_truth = {}  # {child_edge_idx: [parent1_edge_idx, parent2_edge_idx]}

    # === 1. Create initial seed edges ===
    print("Creating initial seed edges...")
    for _ in range(num_initial_edges):
        u, v = random.sample(nodes, 2)
        if v not in adj[u]:
            ts = 0  # Initial timestamp
            edges.append({'u': u, 'i': v, 'ts': ts, 'label': 1, 'idx': edge_idx_counter})
            adj[u].add(v)
            adj[v].add(u)
            edge_idx_counter += 1

    # === 2. Simulate perfect triadic closure evolution ===
    print("Simulating perfect triadic closure evolution...")
    for ts in tqdm(range(1, num_timesteps + 1), desc="Perfect triadic evolution"):
        new_edges_this_timestep = []
        
        # --- PERFECT Causal edge generation (ALL nodes checked) ---
        potential_closures = []
        
        # CHECK ALL NODES (not just random 20)
        for u in nodes:
            # Find neighbors of u (w)
            for w in list(adj[u]):
                # Find neighbors of w (v), which are not u or already connected to u
                for v in list(adj[w]):
                    if u != v and v not in adj[u]:
                        potential_closures.append((u, v, w))
        
        if potential_closures:
            # Create ALL possible triadic closures at this timestep
            edges_to_add = {}  # {(u,v): (w, parent1, parent2)} - deduplicate by edge
            
            for u, v, w in potential_closures:
                # Skip if this edge is already being added this timestep
                edge_key = (min(u, v), max(u, v))  # Normalize edge direction
                if edge_key in edges_to_add:
                    continue
                    
                # Find the parent edge indices
                parent1, parent2 = -1, -1
                for edge in edges:
                    if (edge['u'] == u and edge['i'] == w) or (edge['u'] == w and edge['i'] == u):
                        parent1 = edge['idx']
                    if (edge['u'] == w and edge['i'] == v) or (edge['u'] == v and edge['i'] == w):
                        parent2 = edge['idx']
                
                if parent1 != -1 and parent2 != -1:
                    edges_to_add[edge_key] = (w, parent1, parent2)
            
            # Add all unique triadic closure edges
            for (u, v), (w, parent1, parent2) in edges_to_add.items():
                new_edge = {'u': u, 'i': v, 'ts': ts, 'label': 1, 'idx': edge_idx_counter}
                new_edges_this_timestep.append(new_edge)
                ground_truth[edge_idx_counter] = [parent1, parent2]
                edge_idx_counter += 1

        # --- Optional noise edge generation ---
        if noise_ratio > 0 and (random.random() < noise_ratio or not potential_closures):
            u, v = random.sample(nodes, 2)
            if v not in adj[u]:
                new_edge = {'u': u, 'i': v, 'ts': ts, 'label': 1, 'idx': edge_idx_counter}
                new_edges_this_timestep.append(new_edge)
                edge_idx_counter += 1

        # Add the new edges to the graph
        for edge in new_edges_this_timestep:
            edges.append(edge)
            adj[edge['u']].add(edge['i'])
            adj[edge['i']].add(edge['u'])
        
        # Stop early if no new edges (graph saturated)
        if len(new_edges_this_timestep) == 0:
            print(f"Graph saturated at timestamp {ts} (no more triadic closures possible)")
            break

    # === 3. Format and save the data ===
    output_dir = os.path.join("processed", data_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving perfect triadic data to {output_dir}...")

    # Create DataFrame
    g_df = pd.DataFrame(edges)

    # Generate random features
    num_total_edges = len(g_df)
    e_feat = np.random.randn(num_total_edges + 1, edge_feat_dim)
    n_feat = np.random.randn(num_nodes + 1, node_feat_dim)

    # Save files
    csv_path = os.path.join(output_dir, f"ml_{data_name}.csv")
    gt_path = os.path.join(output_dir, f"ml_{data_name}_gt.json")
    node_feat_path = os.path.join(output_dir, f"ml_{data_name}_node.npy")
    edge_feat_path = os.path.join(output_dir, f"ml_{data_name}.npy")

    g_df.to_csv(csv_path, index=False)
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f)
    np.save(node_feat_path, n_feat)
    np.save(edge_feat_path, e_feat)

    # Print statistics
    print(f"\nPerfect triadic data generation complete!")
    print(f"Generated {len(g_df)} total edges.")
    print(f"Generated {len(ground_truth)} triadic closures ({len(ground_truth)/len(g_df)*100:.1f}%).")
    print(f"Final timestamp: {g_df.ts.max()}")
    print(f"CSV saved to: {csv_path}")
    print(f"Ground truth saved to: {gt_path}")
    print(f"Node features saved to: {node_feat_path}")
    print(f"Edge features saved to: {edge_feat_path}")
    
    # Analyze triadic closure rate by timestamp
    print(f"\nTriadic closure analysis:")
    triadic_by_ts = defaultdict(int)
    for edge_idx_str in ground_truth:
        edge_idx = int(edge_idx_str)
        edge_row = g_df[g_df.idx == edge_idx]
        if not edge_row.empty:
            ts = edge_row.iloc[0]['ts']
            triadic_by_ts[ts] += 1
    
    timestamp_stats = g_df.groupby('ts').size().reset_index(name='edge_count')
    print("Timestamp | Total Edges | Triadic Closures | Non-Triadic")
    print("-" * 55)
    for _, row in timestamp_stats.iterrows():
        ts = row['ts']
        total = row['edge_count']
        triadic = triadic_by_ts[ts]
        non_triadic = total - triadic
        print(f"{ts:9d} | {total:11d} | {triadic:16d} | {non_triadic:11d}")

    return g_df, ground_truth


def create_perfect_datasets():
    """
    Create several perfect triadic closure datasets of different sizes
    """
    datasets_to_create = [
        {
            'name': 'triadic_perfect_small',
            'params': {
                'num_nodes': 20, 
                'num_initial_edges': 10, 
                'num_timesteps': 15, 
                'noise_ratio': 0.0
            }
        },
        {
            'name': 'triadic_perfect_medium',
            'params': {
                'num_nodes': 30, 
                'num_initial_edges': 15, 
                'num_timesteps': 20, 
                'noise_ratio': 0.0
            }
        },
        {
            'name': 'triadic_perfect_large',
            'params': {
                'num_nodes': 50, 
                'num_initial_edges': 25, 
                'num_timesteps': 30, 
                'noise_ratio': 0.0
            }
        }
    ]
    
    created_datasets = []
    for dataset_config in datasets_to_create:
        name = dataset_config['name']
        params = dataset_config['params']
        
        print(f"\n{'='*60}")
        print(f"Creating {name} with params: {params}")
        print(f"{'='*60}")
        
        g_df, ground_truth = generate_perfect_triadic_data(data_name=name, **params)
        created_datasets.append(name)
    
    return created_datasets


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Perfect Triadic Closure Data')
    parser.add_argument('--create_all', action='store_true', 
                       help='Create all perfect datasets')
    parser.add_argument('--data_name', type=str, default='triadic_perfect_medium',
                       help='Name of single dataset to create')
    parser.add_argument('--num_nodes', type=int, default=30)
    parser.add_argument('--num_initial_edges', type=int, default=15)
    parser.add_argument('--num_timesteps', type=int, default=20)
    parser.add_argument('--noise_ratio', type=float, default=0.0)
    
    args = parser.parse_args()
    
    if args.create_all:
        print("Creating all perfect triadic closure datasets...")
        created = create_perfect_datasets()
        print(f"\nCreated datasets: {created}")
    else:
        print(f"Creating single perfect dataset: {args.data_name}")
        generate_perfect_triadic_data(
            num_nodes=args.num_nodes,
            num_initial_edges=args.num_initial_edges,
            num_timesteps=args.num_timesteps,
            noise_ratio=args.noise_ratio,
            data_name=args.data_name
        )
