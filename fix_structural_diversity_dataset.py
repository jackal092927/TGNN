#!/usr/bin/env python3
"""
Fix Structural Diversity Dataset for GraphMamba

This script generates a proper structural diversity contagion dataset that:
1. Creates meaningful positive examples (label=1) for structural diversity contagion
2. Uses discrete timestamps compatible with GraphMamba
3. Generates sufficient data for training
4. Preserves the structural diversity concept
"""

import numpy as np
import pandas as pd
import networkx as nx
import random
import os
import json
from collections import defaultdict


class FixedStructuralDiversityModel:
    """Fixed Structural Diversity Model that generates proper positive examples."""
    
    def __init__(self, graph, seed_nodes, diversity_threshold=0.5, max_timesteps=20, random_seed=42):
        self.graph = graph
        self.seed_nodes = seed_nodes
        self.diversity_threshold = diversity_threshold
        self.max_timesteps = max_timesteps
        self.random_seed = random_seed
        
        # Initialize node states (0=inactive, 1=active)
        self.node_states = {node: 0 for node in graph.nodes()}
        for seed in seed_nodes:
            self.node_states[seed] = 1
            
        # Track activations and their explanations
        self.activations = []  # [(node, timestep, explanation)]
        self.timestep = 0
        
        # Set random seed
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    def get_active_neighbors(self, node):
        """Get active neighbors of a node."""
        active_neighbors = []
        for neighbor in self.graph.neighbors(node):
            if self.node_states[neighbor] == 1:
                active_neighbors.append(neighbor)
        return active_neighbors
    
    def calculate_structural_diversity(self, node, active_neighbors):
        """Calculate structural diversity of active neighbors."""
        if len(active_neighbors) < 2:
            return 0.0
            
        # Count edges between active neighbors
        edges_between_active = 0
        total_possible_edges = len(active_neighbors) * (len(active_neighbors) - 1) // 2
        
        for i, u in enumerate(active_neighbors):
            for v in active_neighbors[i+1:]:
                if self.graph.has_edge(u, v):
                    edges_between_active += 1
        
        # Diversity = 1 - (density of subgraph formed by active neighbors)
        if total_possible_edges == 0:
            return 1.0
        diversity = 1.0 - (edges_between_active / total_possible_edges)
        return diversity
    
    def simulate(self):
        """Simulate Structural Diversity Model with proper positive examples."""
        for t in range(1, self.max_timesteps + 1):
            self.timestep = t
            new_activations = []
            
            for node in self.graph.nodes():
                if self.node_states[node] == 1:  # Already active
                    continue
                    
                active_neighbors = self.get_active_neighbors(node)
                
                if len(active_neighbors) >= 2:  # Need at least 2 neighbors
                    diversity = self.calculate_structural_diversity(node, active_neighbors)
                    
                    if diversity >= self.diversity_threshold:
                        # Record the activation with explanation
                        explanation = {
                            'nodes': active_neighbors,
                            'diversity': diversity,
                            'threshold': self.diversity_threshold
                        }
                        new_activations.append((node, explanation))
            
            # Apply activations and record explanations
            for node, explanation in new_activations:
                self.node_states[node] = 1
                self.activations.append((node, t, explanation))
            
            if len(new_activations) == 0:
                break  # No new activations
                
        return self.activations
    
    def get_dataset(self):
        """Convert activations to temporal graph dataset format with proper positive examples."""
        edges = []
        edge_features = []
        explanations_dict = {}
        
        edge_idx = 0
        
        # Create POSITIVE examples (causal edges from activations)
        for activation in self.activations:
            node, timestep, explanation = activation
            
            # Create edges between explained neighbors and activated node
            for neighbor in explanation['nodes']:
                edges.append({
                    'u': neighbor,
                    'i': node, 
                    'ts': timestep,
                    'label': 1,  # POSITIVE - causal edge
                    'idx': edge_idx
                })
                # Feature: [diversity_score, threshold, neighbor_count]
                edge_features.append([
                    explanation['diversity'], 
                    explanation['threshold'],
                    len(explanation['nodes'])
                ])
                explanations_dict[edge_idx] = explanation
                edge_idx += 1
        
        # Add NEGATIVE examples (non-causal edges)
        # Sample random node pairs that don't have causal relationships
        num_negative = min(len(edges), 100)  # Balance positive/negative
        
        # Get all nodes that appear in positive examples
        positive_nodes = set()
        for edge in edges:
            positive_nodes.add(edge['u'])
            positive_nodes.add(edge['i'])
        
        # Generate negative examples
        negative_count = 0
        max_attempts = 1000
        attempts = 0
        
        while negative_count < num_negative and attempts < max_attempts:
            attempts += 1
            
            # Randomly sample two nodes
            if len(positive_nodes) >= 2:
                u, v = random.sample(list(positive_nodes), 2)
            else:
                continue
                
            # Check if this pair already exists as a positive example
            pair_exists = False
            for edge in edges:
                if (edge['u'] == u and edge['i'] == v) or (edge['u'] == v and edge['i'] == u):
                    pair_exists = True
                    break
            
            if not pair_exists:
                # Create negative example
                edges.append({
                    'u': u,
                    'i': v,
                    'ts': random.randint(1, self.max_timesteps),
                    'label': 0,  # NEGATIVE - non-causal edge
                    'idx': edge_idx
                })
                edge_features.append([0.0, 0.0, 0.0])  # No diversity features
                explanations_dict[edge_idx] = {'type': 'negative', 'reason': 'non-causal'}
                edge_idx += 1
                negative_count += 1
        
        # Add some structural edges (edges that exist in the base graph but aren't causal)
        graph_edges = list(self.graph.edges())
        structural_count = 0
        max_structural = min(50, len(graph_edges))
        
        for u, v in graph_edges:
            if structural_count >= max_structural:
                break
                
            # Check if this pair already exists
            pair_exists = False
            for edge in edges:
                if (edge['u'] == u and edge['i'] == v) or (edge['u'] == v and edge['i'] == u):
                    pair_exists = True
                    break
            
            if not pair_exists:
                edges.append({
                    'u': u,
                    'i': v,
                    'ts': random.randint(1, self.max_timesteps),
                    'label': 0,  # NEGATIVE - structural but non-causal
                    'idx': edge_idx
                })
                edge_features.append([0.0, 0.0, 0.0])
                explanations_dict[edge_idx] = {'type': 'structural', 'reason': 'base_graph_edge'}
                edge_idx += 1
                structural_count += 1
        
        return {
            'edges_df': pd.DataFrame(edges),
            'edge_features': np.array(edge_features),
            'ground_truth_explanations': explanations_dict
        }


def generate_fixed_sd_dataset(graph_type='ba', n_nodes=100, n_edges=200, 
                             seed_fraction=0.1, max_timesteps=20, diversity_threshold=0.5, 
                             random_seed=42):
    """Generate a fixed structural diversity dataset."""
    
    # Create base graph
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if graph_type == 'ba':  # Barabási-Albert preferential attachment
        m = max(2, n_edges // n_nodes)  # Number of edges to attach from a new node
        graph = nx.barabasi_albert_graph(n_nodes, m, seed=random_seed)
    elif graph_type == 'er':  # Erdős-Rényi random graph
        p = 2 * n_edges / (n_nodes * (n_nodes - 1))  # Edge probability
        graph = nx.erdos_renyi_graph(n_nodes, p, seed=random_seed)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # Relabel nodes to start from 1 (to match existing data format)
    mapping = {i: i+1 for i in range(n_nodes)}
    graph = nx.relabel_nodes(graph, mapping)
    
    # Select seed nodes
    num_seeds = max(2, int(seed_fraction * n_nodes))  # Need at least 2 seeds
    seed_nodes = random.sample(list(graph.nodes()), num_seeds)
    
    # Create and run simulation
    simulator = FixedStructuralDiversityModel(
        graph, seed_nodes, 
        diversity_threshold=diversity_threshold,
        max_timesteps=max_timesteps, 
        random_seed=random_seed
    )
    
    # Run simulation
    activations = simulator.simulate()
    dataset = simulator.get_dataset()
    
    print(f"Generated fixed SD dataset:")
    print(f"  - Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    print(f"  - Seeds: {len(seed_nodes)} nodes")
    print(f"  - Activations: {len(activations)} events")
    print(f"  - Dataset edges: {len(dataset['edges_df'])} edges")
    print(f"  - Positive examples: {len(dataset['edges_df'][dataset['edges_df']['label'] == 1])}")
    print(f"  - Negative examples: {len(dataset['edges_df'][dataset['edges_df']['label'] == 0])}")
    
    return dataset, graph, simulator


def save_fixed_dataset(dataset, data_name, output_dir='./processed'):
    """Save the fixed dataset in the format expected by GraphMamba."""
    
    # Create output directory
    dataset_dir = os.path.join(output_dir, data_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save original CSV (for process.py compatibility)
    edges_df = dataset['edges_df']
    
    # Create features column
    features_str = []
    for feat_row in dataset['edge_features']:
        features_str.append(','.join([str(f) for f in feat_row]))
    
    # Create CSV with format: u,i,ts,label,features...
    csv_data = []
    for idx, row in edges_df.iterrows():
        feat_str = ','.join([str(f) for f in dataset['edge_features'][idx]])
        csv_line = f"{row['u']},{row['i']},{row['ts']},{row['label']},{feat_str}"
        csv_data.append(csv_line)
    
    # Write original CSV
    original_csv_path = os.path.join(dataset_dir, f'{data_name}.csv')
    with open(original_csv_path, 'w') as f:
        f.write('u,i,ts,label,feat\n')  # Header
        for line in csv_data:
            f.write(line + '\n')
    
    # Save processed versions (what GraphMamba expects)
    processed_csv_path = os.path.join(dataset_dir, f'ml_{data_name}.csv')
    processed_feat_path = os.path.join(dataset_dir, f'ml_{data_name}.npy')
    processed_node_feat_path = os.path.join(dataset_dir, f'ml_{data_name}_node.npy')
    
    # Process the data
    edges_df['idx'] = range(len(edges_df))
    
    # Reindex nodes to be continuous starting from 1
    all_nodes = sorted(set(edges_df['u'].tolist() + edges_df['i'].tolist()))
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes, 1)}
    
    edges_df['u'] = edges_df['u'].map(node_mapping)
    edges_df['i'] = edges_df['i'].map(node_mapping)
    edges_df['idx'] = range(1, len(edges_df) + 1)  # 1-indexed
    
    # Save processed CSV
    edges_df.to_csv(processed_csv_path, index=False)
    
    # Save edge features
    edge_features = dataset['edge_features']
    # Add empty row at beginning (for 0-indexing compatibility)
    empty_row = np.zeros(edge_features.shape[1])[np.newaxis, :]
    edge_features_with_empty = np.vstack([empty_row, edge_features])
    np.save(processed_feat_path, edge_features_with_empty)
    
    # Save node features (random features for all nodes)
    max_node_id = max(edges_df['u'].max(), edges_df['i'].max())
    node_features = np.random.randn(max_node_id + 1, edge_features.shape[1])
    np.save(processed_node_feat_path, node_features)
    
    # Save ground truth explanations
    explanations_path = os.path.join(dataset_dir, f'{data_name}_explanations.json')
    # Convert explanations to JSON-serializable format
    explanations_serializable = {}
    for k, v in dataset['ground_truth_explanations'].items():
        if isinstance(v, dict):
            # Convert numpy types to Python types
            explanations_serializable[str(k)] = {}
            for key, val in v.items():
                if isinstance(val, (np.integer, np.floating)):
                    explanations_serializable[str(k)][key] = float(val)
                elif isinstance(val, list):
                    explanations_serializable[str(k)][key] = [int(x) if isinstance(x, np.integer) else x for x in val]
                else:
                    explanations_serializable[str(k)][key] = val
        else:
            explanations_serializable[str(k)] = v
    
    with open(explanations_path, 'w') as f:
        json.dump(explanations_serializable, f, indent=2)
    
    print(f"Saved fixed dataset to {dataset_dir}")
    print(f"Files created:")
    print(f"  - {original_csv_path}")
    print(f"  - {processed_csv_path}")
    print(f"  - {processed_feat_path}")
    print(f"  - {processed_node_feat_path}")
    print(f"  - {explanations_path}")


if __name__ == "__main__":
    # Generate a fixed structural diversity dataset
    print("Generating fixed structural diversity dataset...")
    
    # Generate with better parameters
    dataset, graph, simulator = generate_fixed_sd_dataset(
        graph_type='ba',
        n_nodes=100,  # Larger graph
        n_edges=300,  # More edges
        seed_fraction=0.15,  # More seed nodes
        max_timesteps=25,  # More timesteps
        diversity_threshold=0.4,  # Lower threshold for more activations
        random_seed=42
    )
    
    # Save the fixed dataset
    save_fixed_dataset(dataset, 'synthetic_sd_ba_fixed')
    
    print("\nDataset generation complete!")
    print("The fixed dataset now has:")
    print("1. Proper positive examples (label=1) for structural diversity contagion")
    print("2. Discrete timestamps compatible with GraphMamba")
    print("3. Sufficient data size for training")
    print("4. Meaningful structural diversity features")
