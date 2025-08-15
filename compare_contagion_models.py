#!/usr/bin/env python3
"""
Compare different contagion models side-by-side.
"""

import argparse
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_analyze_dataset(dataset_name):
    """Load dataset and return final activation state."""
    base_path = Path(f"processed/{dataset_name}")
    
    if not base_path.exists():
        return None, None, None, None
    
    # Load the edge data
    df = pd.read_csv(base_path / f"ml_{dataset_name}.csv")
    
    # Load ground truth explanations
    gt_file = base_path / f"{dataset_name}_explanations.json"
    ground_truth = {}
    if gt_file.exists():
        with open(gt_file) as f:
            ground_truth = json.load(f)
    
    # Get all unique nodes
    all_nodes = set(df['u'].unique()) | set(df['i'].unique())
    
    # Create base graph from all edges
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['u'], row['i'])
    
    # Extract activation events from causal edges
    activated_nodes = set()
    causal_df = df[df['label'] == 1].copy()
    
    for _, row in causal_df.iterrows():
        activated_nodes.add(row['i'])
    
    return G, activated_nodes, len(causal_df), len(df)

def create_comparison_plot(datasets, output_file=None):
    """Create side-by-side comparison of contagion models."""
    
    valid_datasets = []
    graph_data = []
    
    # Load all datasets
    for dataset_name in datasets:
        G, activated_nodes, num_activations, total_edges = load_and_analyze_dataset(dataset_name)
        if G is not None:
            model_type = dataset_name.split('_')[1] if '_' in dataset_name else dataset_name
            valid_datasets.append({
                'name': dataset_name,
                'model': model_type.upper(),
                'graph': G,
                'activated': activated_nodes,
                'num_activations': num_activations,
                'total_edges': total_edges
            })
    
    if not valid_datasets:
        print("No valid datasets found!")
        return
    
    # Create subplots
    n_datasets = len(valid_datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 6))
    if n_datasets == 1:
        axes = [axes]
    
    for i, data in enumerate(valid_datasets):
        ax = axes[i]
        G = data['graph']
        activated_nodes = data['activated']
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Determine node colors
        node_colors = []
        for node in G.nodes():
            if node in activated_nodes:
                node_colors.append('red')
            else:
                node_colors.append('lightgray')
        
        # Draw network
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', alpha=0.5)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax, 
                             node_size=200, alpha=0.8)
        
        # Add node labels for smaller networks
        if G.number_of_nodes() <= 50:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
        
        # Title with statistics
        activation_rate = len(activated_nodes) / G.number_of_nodes() * 100
        ax.set_title(f'{data["model"]} Model\n'
                    f'{len(activated_nodes)}/{G.number_of_nodes()} nodes activated ({activation_rate:.1f}%)\n'
                    f'{data["num_activations"]} events, {data["total_edges"]} total edges',
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Overall title
    fig.suptitle('Contagion Model Comparison', fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Circle((0, 0), 0.1, color='lightgray', label='Inactive'),
        plt.Circle((0, 0), 0.1, color='red', label='Activated')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if output_file is None:
        output_file = "contagion_models_comparison.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {output_file}")
    
    # Print statistics
    print("\nModel Statistics:")
    print("=" * 50)
    for data in valid_datasets:
        activation_rate = len(data['activated']) / data['graph'].number_of_nodes() * 100
        print(f"{data['model']:3s} | {len(data['activated']):3d}/{data['graph'].number_of_nodes():3d} nodes ({activation_rate:5.1f}%) | "
              f"{data['num_activations']:3d} events | {data['total_edges']:3d} edges")

def main():
    parser = argparse.ArgumentParser(description='Compare contagion models')
    parser.add_argument('datasets', nargs='+', help='Dataset names to compare')
    parser.add_argument('--output', '-o', help='Output file name')
    
    args = parser.parse_args()
    
    create_comparison_plot(args.datasets, args.output)

if __name__ == "__main__":
    main() 