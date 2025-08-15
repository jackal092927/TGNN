#!/usr/bin/env python3
"""
Script to generate synthetic contagion datasets with ground truth explanations.
"""

from synthetic_contagion import generate_synthetic_dataset, save_dataset
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic contagion datasets')
    parser.add_argument('--model', type=str, choices=['ltm', 'icm', 'cc', 'sd'], 
                       default='ltm', help='Contagion model type')
    parser.add_argument('--graph', type=str, choices=['ba', 'er', 'ws'], 
                       default='ba', help='Graph topology type')
    parser.add_argument('--nodes', type=int, default=1000, help='Number of nodes')
    parser.add_argument('--edges', type=int, default=2000, help='Number of edges')
    parser.add_argument('--seeds', type=float, default=0.05, help='Fraction of seed nodes')
    parser.add_argument('--timesteps', type=int, default=50, help='Maximum timesteps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./processed', help='Output directory')
    
    args = parser.parse_args()
    
    # Generate dataset name
    data_name = f"synthetic_{args.model}_{args.graph}"
    
    print(f"Generating {data_name} dataset...")
    print(f"Parameters:")
    print(f"  - Model: {args.model}")
    print(f"  - Graph: {args.graph}")
    print(f"  - Nodes: {args.nodes}")
    print(f"  - Edges: {args.edges}")
    print(f"  - Seed fraction: {args.seeds}")
    print(f"  - Max timesteps: {args.timesteps}")
    print(f"  - Random seed: {args.seed}")
    
    # Generate the dataset
    dataset, graph, simulator = generate_synthetic_dataset(
        model_type=args.model,
        graph_type=args.graph,
        n_nodes=args.nodes,
        n_edges=args.edges,
        seed_fraction=args.seeds,
        max_timesteps=args.timesteps,
        random_seed=args.seed
    )
    
    # Save the dataset
    dataset_dir = save_dataset(dataset, data_name, args.output_dir)
    
    print(f"\nDataset generated successfully!")
    print(f"Dataset directory: {dataset_dir}")
    print(f"To train/evaluate on this dataset, use:")
    print(f"  python learn_edge.py -d {data_name}")


if __name__ == "__main__":
    main() 