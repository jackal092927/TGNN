"""
Generate longer perfect triadic closure datasets using gradual seeding strategy.

Strategy: Start with fewer initial edges, then add new seed edges periodically 
during the first 50% of timestamps to create fresh triadic opportunities.
"""

import os
import random
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def generate_perfect_triadic_long(
    num_nodes=100,
    num_initial_edges=10,  # Start with fewer edges
    num_timesteps=30,      # Target longer timeline
    seed_interval=5,       # Add seed every N timestamps
    seed_edges_per_interval=3,  # How many seed edges to add
    noise_ratio=0.0,       # Keep perfect (no noise)
    node_feat_dim=100,
    edge_feat_dim=100,
    data_name="triadic_perfect_long"
):
    """
    Generate perfect triadic closure dataset with gradual seeding for longer timelines.
    
    Key strategy:
    1. Start with few initial edges
    2. Add new seed edges every `seed_interval` timestamps
    3. Only add seeds during first 50% of total timestamps
    4. Create ALL triadic closures at each timestamp (perfect behavior)
    
    Args:
        num_nodes: Total number of nodes
        num_initial_edges: Initial seed edges at t=0
        num_timesteps: Target number of timestamps
        seed_interval: Add seed edges every N timestamps
        seed_edges_per_interval: Number of seed edges to add each time
        noise_ratio: Ratio of noise edges (0.0 for perfect)
        data_name: Dataset name
    """
    print(f"Generating long perfect triadic closure data: {data_name}")
    print(f"Strategy: Gradual seeding with seeds only in first 50% of timeline")
    print(f"Parameters: {num_nodes} nodes, {num_timesteps} timestamps, seed every {seed_interval} steps")
    
    # Initialize
    nodes = list(range(num_nodes))  # Use list for easier sampling
    adj = defaultdict(set)
    edges = []
    edge_idx_counter = 0
    ground_truth = {}
    
    # Calculate seeding cutoff (first 50% of timeline)
    seeding_cutoff = num_timesteps // 2
    print(f"Will add seed edges until timestamp {seeding_cutoff} (50% of {num_timesteps})")

    # === 1. Create initial seed edges ===
    print(f"Creating {num_initial_edges} initial seed edges...")
    added_initial = 0
    attempts = 0
    max_attempts = num_initial_edges * 10
    
    while added_initial < num_initial_edges and attempts < max_attempts:
        u, v = random.sample(nodes, 2)
        if v not in adj[u]:  # Edge doesn't exist
            edges.append({'u': u, 'i': v, 'ts': 0, 'label': 1, 'idx': edge_idx_counter})
            adj[u].add(v)
            adj[v].add(u)
            edge_idx_counter += 1
            added_initial += 1
        attempts += 1
    
    print(f"Successfully added {added_initial} initial edges")

    # === 2. Simulate gradual seeding + perfect triadic closure evolution ===
    print("Simulating gradual seeding evolution...")
    
    for ts in tqdm(range(1, num_timesteps + 1), desc="Long triadic evolution"):
        new_edges_this_timestep = []
        
        # --- Add seed edges if within first 50% and at interval ---
        if ts <= seeding_cutoff and ts % seed_interval == 0:
            print(f"\n  Adding seed edges at timestamp {ts}...")
            seeds_added = 0
            attempts = 0
            max_attempts = seed_edges_per_interval * 20
            
            while seeds_added < seed_edges_per_interval and attempts < max_attempts:
                u, v = random.sample(nodes, 2)
                if v not in adj[u]:  # Edge doesn't exist
                    new_edge = {'u': u, 'i': v, 'ts': ts, 'label': 1, 'idx': edge_idx_counter}
                    new_edges_this_timestep.append(new_edge)
                    # Note: Don't update adj yet, will do after triadic closures
                    seeds_added += 1
                    edge_idx_counter += 1
                attempts += 1
            
            print(f"  Added {seeds_added} seed edges at timestamp {ts}")
        
        # --- Create ALL triadic closures (perfect behavior) ---
        potential_closures = []
        
        # Check ALL nodes for triadic opportunities
        for u in nodes:
            # Find neighbors of u (w)
            for w in list(adj[u]):
                # Find neighbors of w (v), which are not u or already connected to u
                for v in list(adj[w]):
                    if u != v and v not in adj[u]:
                        potential_closures.append((u, v, w))
        
        if potential_closures:
            # Create ALL possible triadic closures
            edges_to_add = {}  # {(u,v): (w, parent1, parent2)} - deduplicate
            
            for u, v, w in potential_closures:
                edge_key = (min(u, v), max(u, v))  # Normalize edge direction
                if edge_key in edges_to_add:
                    continue  # Already found this closure via different path
                    
                # Find parent edge indices
                parent1, parent2 = -1, -1
                for edge in edges:
                    if (edge['u'] == u and edge['i'] == w) or (edge['u'] == w and edge['i'] == u):
                        parent1 = edge['idx']
                    if (edge['u'] == w and edge['i'] == v) or (edge['u'] == v and edge['i'] == w):
                        parent2 = edge['idx']
                
                if parent1 != -1 and parent2 != -1:
                    edges_to_add[edge_key] = (w, parent1, parent2)
            
            # Add all triadic closure edges
            for (u, v), (w, parent1, parent2) in edges_to_add.items():
                new_edge = {'u': u, 'i': v, 'ts': ts, 'label': 1, 'idx': edge_idx_counter}
                new_edges_this_timestep.append(new_edge)
                ground_truth[edge_idx_counter] = [parent1, parent2]
                edge_idx_counter += 1
        
        # --- Optional noise edges (disabled for perfect datasets) ---
        if noise_ratio > 0 and (random.random() < noise_ratio or not potential_closures):
            u, v = random.sample(nodes, 2)
            if v not in adj[u]:
                new_edge = {'u': u, 'i': v, 'ts': ts, 'label': 1, 'idx': edge_idx_counter}
                new_edges_this_timestep.append(new_edge)
                edge_idx_counter += 1

        # --- Update graph state ---
        for edge in new_edges_this_timestep:
            edges.append(edge)
            adj[edge['u']].add(edge['i'])
            adj[edge['i']].add(edge['u'])
        
        # Check if we should continue (avoid infinite empty loops)
        if len(new_edges_this_timestep) == 0 and ts > seeding_cutoff:
            print(f"\nNo new edges possible and past seeding phase. Stopping at timestamp {ts-1}")
            break
            
        if len(new_edges_this_timestep) > 0:
            print(f"  Timestamp {ts}: {len(new_edges_this_timestep)} new edges ({len([e for e in new_edges_this_timestep if e['idx'] in ground_truth])} triadic)")

    # === 3. Save the data ===
    output_dir = os.path.join("processed", data_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving long perfect triadic data to {output_dir}...")

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

    # Print final statistics
    final_timestamp = g_df.ts.max()
    print(f"\nLong perfect triadic data generation complete!")
    print(f"Generated {len(g_df)} total edges over {int(final_timestamp + 1)} timestamps.")
    print(f"Generated {len(ground_truth)} triadic closures ({len(ground_truth)/len(g_df)*100:.1f}%).")
    print(f"Timeline: timestamps 0 to {int(final_timestamp)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Ground truth saved to: {gt_path}")
    
    # Analyze timeline
    print(f"\nTimeline analysis:")
    triadic_by_ts = defaultdict(int)
    for edge_idx_str in ground_truth:
        edge_idx = int(edge_idx_str)
        edge_row = g_df[g_df.idx == edge_idx]
        if not edge_row.empty:
            ts = edge_row.iloc[0]['ts']
            triadic_by_ts[ts] += 1
    
    timestamp_stats = g_df.groupby('ts').size().reset_index(name='edge_count')
    print("Timestamp | Total Edges | Triadic Closures | Seeds/Noise")
    print("-" * 60)
    for _, row in timestamp_stats.iterrows():
        ts = row['ts']
        total = row['edge_count']
        triadic = triadic_by_ts[ts]
        non_triadic = total - triadic
        print(f"{ts:9d} | {total:11d} | {triadic:16d} | {non_triadic:11d}")

    return g_df, ground_truth


def create_long_perfect_datasets():
    """
    Create several long perfect triadic closure datasets
    """
    datasets_to_create = [
        {
            'name': 'triadic_perfect_long_medium',
            'params': {
                'num_nodes': 80,
                'num_initial_edges': 8,
                'num_timesteps': 25,
                'seed_interval': 4,
                'seed_edges_per_interval': 2,
                'noise_ratio': 0.0
            }
        },
        {
            'name': 'triadic_perfect_long_large',
            'params': {
                'num_nodes': 120,
                'num_initial_edges': 10,
                'num_timesteps': 35,
                'seed_interval': 5,
                'seed_edges_per_interval': 3,
                'noise_ratio': 0.0
            }
        }
    ]
    
    created_datasets = []
    for dataset_config in datasets_to_create:
        name = dataset_config['name']
        params = dataset_config['params']
        
        print(f"\n{'='*80}")
        print(f"Creating {name}")
        print(f"Parameters: {params}")
        print(f"{'='*80}")
        
        g_df, ground_truth = generate_perfect_triadic_long(data_name=name, **params)
        created_datasets.append(name)
    
    return created_datasets


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Long Perfect Triadic Closure Data')
    parser.add_argument('--create_all', action='store_true', 
                       help='Create all long perfect datasets')
    parser.add_argument('--data_name', type=str, default='triadic_perfect_long_medium',
                       help='Name of single dataset to create')
    parser.add_argument('--num_nodes', type=int, default=80)
    parser.add_argument('--num_initial_edges', type=int, default=8)
    parser.add_argument('--num_timesteps', type=int, default=25)
    parser.add_argument('--seed_interval', type=int, default=4)
    parser.add_argument('--seed_edges_per_interval', type=int, default=2)
    
    args = parser.parse_args()
    
    if args.create_all:
        print("Creating all long perfect triadic closure datasets...")
        created = create_long_perfect_datasets()
        print(f"\nCreated datasets: {created}")
    else:
        print(f"Creating single long perfect dataset: {args.data_name}")
        generate_perfect_triadic_long(
            num_nodes=args.num_nodes,
            num_initial_edges=args.num_initial_edges,
            num_timesteps=args.num_timesteps,
            seed_interval=args.seed_interval,
            seed_edges_per_interval=args.seed_edges_per_interval,
            data_name=args.data_name
        )
