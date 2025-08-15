import os
import random
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def generate_synthetic_data(
    num_nodes=100,
    num_initial_edges=100,
    num_timesteps=500,
    noise_ratio=0.1,
    node_feat_dim=100,
    edge_feat_dim=100,
    data_name="synthetic"
):
    """
    Generates a synthetic dynamic graph dataset based on the Triadic Closure rule.

    Args:
        num_nodes (int): Total number of nodes in the graph.
        num_initial_edges (int): Number of random seed edges to start with.
        num_timesteps (int): Number of simulation steps.
        noise_ratio (float): Proportion of new edges at each step that are random noise.
        node_feat_dim (int): Dimension of node features.
        edge_feat_dim (int): Dimension of edge features.
        data_name (str): Name of the dataset, used for directory and file naming.
    """
    print("Generating synthetic data...")
    # Initialize nodes and adjacency list for tracking neighbors
    nodes = set(range(num_nodes))
    adj = defaultdict(set)
    edges = []
    edge_idx_counter = 0
    ground_truth = {}  # {child_edge_idx: [parent1_edge_idx, parent2_edge_idx]}

    # === 1. Create initial seed edges ===
    for _ in range(num_initial_edges):
        u, v = random.sample(nodes, 2)
        if v not in adj[u]:
            ts = 0  # Initial timestamp
            edges.append({'u': u, 'i': v, 'ts': ts, 'label': 1, 'idx': edge_idx_counter})
            adj[u].add(v)
            adj[v].add(u)
            edge_idx_counter += 1

    # === 2. Simulate graph evolution over time ===
    for ts in tqdm(range(1, num_timesteps + 1), desc="Simulating graph evolution"):
        new_edges_this_timestep = []
        
        # --- Causal edge generation (Triadic Closure) ---
        potential_closures = []
        nodes_to_check = random.sample(nodes, k=min(len(nodes), 20)) # Check a subset for efficiency

        for u in nodes_to_check:
            # Find neighbors of u (w)
            for w in list(adj[u]):
                # Find neighbors of w (v), which are not u or already connected to u
                for v in list(adj[w]):
                    if u != v and v not in adj[u]:
                        potential_closures.append((u, v, w))
        
        if potential_closures:
            # Create ALL possible triadic closures at this timestep (FIXED!)
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

        # --- Noise edge generation ---
        if random.random() < noise_ratio or not potential_closures:
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

    # === 3. Format and save the data ===
    output_dir = os.path.join("processed", data_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving data to {output_dir}...")

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

    print(f"Synthetic data generation complete.")
    print(f"Generated {len(g_df)} total edges.")
    print(f"Generated {len(ground_truth)} edges with ground-truth explanations.")
    print(f"CSv saved to: {csv_path}")
    print(f"Ground truth saved to: {gt_path}")
    print(f"Node features saved to: {node_feat_path}")
    print(f"Edge features saved to: {edge_feat_path}")

if __name__ == "__main__":
    generate_synthetic_data() 