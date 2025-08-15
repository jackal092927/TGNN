import numpy as np
import pandas as pd
import networkx as nx
import random
from collections import defaultdict
import os


class ContagionSimulator:
    """Base class for contagion models with ground truth explanation tracking."""
    
    def __init__(self, graph, seed_nodes, max_timesteps=100, random_seed=42):
        self.graph = graph
        self.seed_nodes = seed_nodes
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
    
    def get_active_neighbors(self, node, timestep=None):
        """Get active neighbors of a node at given timestep."""
        if timestep is None:
            timestep = self.timestep
        active_neighbors = []
        for neighbor in self.graph.neighbors(node):
            if self.node_states[neighbor] == 1:
                active_neighbors.append(neighbor)
        return active_neighbors
    
    def simulate(self):
        """Run the simulation and return activation log with ground truth."""
        raise NotImplementedError("Subclasses must implement simulate method")
    
    def get_dataset(self):
        """Convert activations to temporal graph dataset format."""
        edges = []
        edge_features = []
        explanations_dict = {}
        
        edge_idx = 0
        for activation in self.activations:
            node, timestep, explanation = activation
            
            # Create edges between explained neighbors and activated node
            if isinstance(explanation, list):  # Multiple neighbors (LTM, Complex Contagion)
                for neighbor in explanation:
                    edges.append({
                        'u': neighbor,
                        'i': node, 
                        'ts': timestep,
                        'label': 1,  # Causal edge
                        'idx': edge_idx
                    })
                    edge_features.append([1.0])  # Simple feature indicating causal edge
                    explanations_dict[edge_idx] = explanation
                    edge_idx += 1
                    
            elif isinstance(explanation, int):  # Single neighbor (ICM)
                neighbor = explanation
                edges.append({
                    'u': neighbor,
                    'i': node,
                    'ts': timestep, 
                    'label': 1,  # Causal edge
                    'idx': edge_idx
                })
                edge_features.append([1.0])
                explanations_dict[edge_idx] = [neighbor]
                edge_idx += 1
        
        # Add some non-causal edges for negative examples
        non_causal_edges = self._generate_non_causal_edges(edge_idx)
        edges.extend(non_causal_edges['edges'])
        edge_features.extend(non_causal_edges['features'])
        explanations_dict.update(non_causal_edges['explanations'])
        
        return {
            'edges_df': pd.DataFrame(edges),
            'edge_features': np.array(edge_features),
            'ground_truth_explanations': explanations_dict
        }
    
    def _generate_non_causal_edges(self, start_idx):
        """Generate non-causal edges for negative examples."""
        edges = []
        features = []
        explanations = {}
        
        edge_idx = start_idx
        num_non_causal = len(self.activations) // 2  # Add 50% non-causal edges
        
        for _ in range(num_non_causal):
            # Pick random edge from graph that wasn't causal
            u, v = random.choice(list(self.graph.edges()))
            ts = random.uniform(1, self.max_timesteps)
            
            edges.append({
                'u': u,
                'i': v,
                'ts': ts,
                'label': 0,  # Non-causal edge
                'idx': edge_idx
            })
            features.append([0.0])  # Feature indicating non-causal
            explanations[edge_idx] = []  # Empty explanation
            edge_idx += 1
            
        return {'edges': edges, 'features': features, 'explanations': explanations}


class LinearThresholdModel(ContagionSimulator):
    """Linear Threshold Model: Node activates when sum of neighbor weights > threshold."""
    
    def __init__(self, graph, seed_nodes, thresholds=None, weights=None, **kwargs):
        super().__init__(graph, seed_nodes, **kwargs)
        
        # Initialize thresholds 
        if thresholds is None:
            self.thresholds = {node: np.random.uniform(0.3, 0.7) for node in graph.nodes()}
        else:
            self.thresholds = thresholds
            
        # Initialize edge weights
        if weights is None:
            self.weights = {}
            for u, v in graph.edges():
                weight = np.random.uniform(0.1, 0.5)
                self.weights[(u, v)] = weight
                self.weights[(v, u)] = weight  # Symmetric
        else:
            self.weights = weights
    
    def simulate(self):
        """Simulate Linear Threshold Model."""
        for t in range(1, self.max_timesteps + 1):
            self.timestep = t
            new_activations = []
            
            for node in self.graph.nodes():
                if self.node_states[node] == 1:  # Already active
                    continue
                    
                # Calculate influence from active neighbors
                influence = 0
                active_neighbors = self.get_active_neighbors(node)
                
                for neighbor in active_neighbors:
                    edge_key = (neighbor, node)
                    if edge_key in self.weights:
                        influence += self.weights[edge_key]
                    elif (node, neighbor) in self.weights:
                        influence += self.weights[(node, neighbor)]
                
                # Check if threshold is exceeded
                if influence > self.thresholds[node]:
                    new_activations.append((node, active_neighbors))
            
            # Apply activations and record explanations
            for node, explanation in new_activations:
                self.node_states[node] = 1
                self.activations.append((node, t, explanation))
            
            if len(new_activations) == 0:
                break  # No new activations
                
        return self.activations


class IndependentCascadeModel(ContagionSimulator):
    """Independent Cascade Model: Each newly active neighbor gets one chance to activate node."""
    
    def __init__(self, graph, seed_nodes, activation_probs=None, **kwargs):
        super().__init__(graph, seed_nodes, **kwargs)
        
        # Initialize activation probabilities
        if activation_probs is None:
            self.activation_probs = {}
            for u, v in graph.edges():
                prob = np.random.uniform(0.1, 0.4)
                self.activation_probs[(u, v)] = prob
                self.activation_probs[(v, u)] = prob  # Symmetric
        else:
            self.activation_probs = activation_probs
            
        # Track which nodes were newly activated each timestep
        self.newly_active = {0: set(seed_nodes)}
    
    def simulate(self):
        """Simulate Independent Cascade Model."""
        for t in range(1, self.max_timesteps + 1):
            self.timestep = t
            current_newly_active = set()
            
            # Only newly active nodes from previous timestep can activate others
            if t-1 not in self.newly_active:
                break
                
            for active_node in self.newly_active[t-1]:
                for neighbor in self.graph.neighbors(active_node):
                    if self.node_states[neighbor] == 1:  # Already active
                        continue
                        
                    # Try to activate with probability
                    edge_key = (active_node, neighbor)
                    if edge_key in self.activation_probs:
                        prob = self.activation_probs[edge_key]
                    elif (neighbor, active_node) in self.activation_probs:
                        prob = self.activation_probs[(neighbor, active_node)]
                    else:
                        prob = 0.2  # Default probability
                    
                    if np.random.random() < prob:
                        self.node_states[neighbor] = 1
                        current_newly_active.add(neighbor)
                        # The explanation is the single successful activator
                        self.activations.append((neighbor, t, active_node))
            
            self.newly_active[t] = current_newly_active
            
            if len(current_newly_active) == 0:
                break  # No new activations
                
        return self.activations


class ComplexContagionModel(ContagionSimulator):
    """Complex Contagion: Node needs k active neighbors to activate."""
    
    def __init__(self, graph, seed_nodes, k=2, **kwargs):
        super().__init__(graph, seed_nodes, **kwargs)
        self.k = k  # Minimum number of active neighbors needed
    
    def simulate(self):
        """Simulate Complex Contagion Model."""
        for t in range(1, self.max_timesteps + 1):
            self.timestep = t
            new_activations = []
            
            for node in self.graph.nodes():
                if self.node_states[node] == 1:  # Already active
                    continue
                    
                active_neighbors = self.get_active_neighbors(node)
                
                # Check if we have enough active neighbors
                if len(active_neighbors) >= self.k:
                    new_activations.append((node, active_neighbors))
            
            # Apply activations and record explanations
            for node, explanation in new_activations:
                self.node_states[node] = 1
                self.activations.append((node, t, explanation))
            
            if len(new_activations) == 0:
                break  # No new activations
                
        return self.activations


class StructuralDiversityModel(ContagionSimulator):
    """Structural Diversity: Activation depends on configuration of active neighbors."""
    
    def __init__(self, graph, seed_nodes, diversity_threshold=0.5, **kwargs):
        super().__init__(graph, seed_nodes, **kwargs)
        self.diversity_threshold = diversity_threshold
    
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
        """Simulate Structural Diversity Model."""
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
                        # Include the subgraph structure in explanation
                        subgraph_edges = []
                        for i, u in enumerate(active_neighbors):
                            for v in active_neighbors[i+1:]:
                                if self.graph.has_edge(u, v):
                                    subgraph_edges.append((u, v))
                        
                        explanation = {
                            'nodes': active_neighbors,
                            'edges': subgraph_edges,
                            'diversity': diversity
                        }
                        new_activations.append((node, explanation))
            
            # Apply activations and record explanations
            for node, explanation in new_activations:
                self.node_states[node] = 1
                self.activations.append((node, t, explanation))
            
            if len(new_activations) == 0:
                break  # No new activations
                
        return self.activations


def generate_synthetic_dataset(model_type, graph_type='ba', n_nodes=1000, n_edges=2000, 
                             seed_fraction=0.05, max_timesteps=50, random_seed=42):
    """Generate synthetic contagion dataset with ground truth explanations."""
    
    # Create base graph
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if graph_type == 'ba':  # Barabási-Albert preferential attachment
        m = n_edges // n_nodes  # Number of edges to attach from a new node
        graph = nx.barabasi_albert_graph(n_nodes, m, seed=random_seed)
    elif graph_type == 'er':  # Erdős-Rényi random graph
        p = 2 * n_edges / (n_nodes * (n_nodes - 1))  # Edge probability
        graph = nx.erdos_renyi_graph(n_nodes, p, seed=random_seed)
    elif graph_type == 'ws':  # Watts-Strogatz small-world
        k = max(4, n_edges // (n_nodes // 2))  # Each node connected to k nearest neighbors
        p = 0.3  # Rewiring probability
        graph = nx.watts_strogatz_graph(n_nodes, k, p, seed=random_seed)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # Relabel nodes to start from 1 (to match existing data format)
    mapping = {i: i+1 for i in range(n_nodes)}
    graph = nx.relabel_nodes(graph, mapping)
    
    # Select seed nodes
    num_seeds = max(1, int(seed_fraction * n_nodes))
    seed_nodes = random.sample(list(graph.nodes()), num_seeds)
    
    # Create and run simulation
    if model_type == 'ltm':
        simulator = LinearThresholdModel(graph, seed_nodes, max_timesteps=max_timesteps, 
                                       random_seed=random_seed)
    elif model_type == 'icm':
        simulator = IndependentCascadeModel(graph, seed_nodes, max_timesteps=max_timesteps,
                                          random_seed=random_seed)
    elif model_type == 'cc':
        k = random.randint(2, 4)  # Random threshold between 2-4
        simulator = ComplexContagionModel(graph, seed_nodes, k=k, max_timesteps=max_timesteps,
                                        random_seed=random_seed)
    elif model_type == 'sd':
        threshold = random.uniform(0.3, 0.7)  # Random diversity threshold
        simulator = StructuralDiversityModel(graph, seed_nodes, diversity_threshold=threshold,
                                           max_timesteps=max_timesteps, random_seed=random_seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Run simulation
    activations = simulator.simulate()
    dataset = simulator.get_dataset()
    
    print(f"Generated {model_type} dataset:")
    print(f"  - Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    print(f"  - Seeds: {len(seed_nodes)} nodes")
    print(f"  - Activations: {len(activations)} events")
    print(f"  - Dataset edges: {len(dataset['edges_df'])} edges")
    
    return dataset, graph, simulator


def save_dataset(dataset, data_name, output_dir='./processed'):
    """Save dataset in the format expected by the main training pipeline."""
    
    # Create output directory
    dataset_dir = os.path.join(output_dir, data_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save original CSV (for process.py compatibility)
    edges_df = dataset['edges_df']
    
    # Check if we have any edges
    if len(edges_df) == 0:
        print(f"Warning: No edges generated for {data_name}. Creating minimal dataset.")
        # Create a minimal dataset with at least one edge to avoid errors
        edges_df = pd.DataFrame([{
            'u': 1,
            'i': 2,
            'ts': 1.0,
            'label': 0,
            'idx': 0
        }])
        dataset['edge_features'] = np.array([[0.0]])
        dataset['ground_truth_explanations'] = {0: []}
    
    # Create features column (concatenate all edge feature dimensions)
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
    
    # Save processed versions (what the training script expects)
    processed_csv_path = os.path.join(dataset_dir, f'ml_{data_name}.csv')
    processed_feat_path = os.path.join(dataset_dir, f'ml_{data_name}.npy')
    processed_node_feat_path = os.path.join(dataset_dir, f'ml_{data_name}_node.npy')
    
    # Process the data using the existing pipeline logic
    # Add index column
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
    import json
    # Convert explanations to JSON-serializable format
    explanations_serializable = {}
    for k, v in dataset['ground_truth_explanations'].items():
        if isinstance(v, dict) and 'nodes' in v:  # Structural diversity model
            explanations_serializable[str(k)] = {
                'nodes': [node_mapping[n] for n in v['nodes']], 
                'edges': [(node_mapping[u], node_mapping[v_]) for u, v_ in v['edges']],
                'diversity': v['diversity']
            }
        elif isinstance(v, list):  # List of nodes
            explanations_serializable[str(k)] = [node_mapping[n] for n in v if n in node_mapping]
        else:  # Single node or empty
            explanations_serializable[str(k)] = v
    
    with open(explanations_path, 'w') as f:
        json.dump(explanations_serializable, f, indent=2)
    
    print(f"Saved dataset to {dataset_dir}:")
    print(f"  - Processed CSV: {processed_csv_path}")
    print(f"  - Edge features: {processed_feat_path}")
    print(f"  - Node features: {processed_node_feat_path}")
    print(f"  - Ground truth explanations: {explanations_path}")
    
    return dataset_dir


if __name__ == "__main__":
    """Generate example datasets for all models."""
    
    models = ['ltm', 'icm', 'cc', 'sd']
    graph_types = ['ba', 'er', 'ws']
    
    for model in models:
        for graph_type in graph_types:
            data_name = f"synthetic_{model}_{graph_type}"
            print(f"\nGenerating {data_name}...")
            
            dataset, graph, simulator = generate_synthetic_dataset(
                model_type=model,
                graph_type=graph_type,
                n_nodes=500,  # Smaller for faster generation
                n_edges=1000,
                seed_fraction=0.05,
                max_timesteps=20,
                random_seed=42
            )
            
            save_dataset(dataset, data_name)
    
    print("\nAll synthetic datasets generated successfully!") 