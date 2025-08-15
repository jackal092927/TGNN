#!/usr/bin/env python3
"""
Compare different triadic closure patterns side by side.
Shows how different initialization and parameters affect triangle formation.
"""

import argparse
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from visualize_triadic_closure import load_or_generate_triadic_data, reconstruct_triadic_timeline

def analyze_triadic_pattern(dataset_name):
    """Analyze triadic closure pattern for a dataset."""
    try:
        df, ground_truth = load_or_generate_triadic_data(dataset_name)
        timeline = reconstruct_triadic_timeline(df, ground_truth)
        
        if len(timeline) <= 1:
            return None
        
        # Calculate statistics
        total_edges = timeline[-1]['total_edges']
        total_triadic = sum(len([e for e in t['new_edges'] if e['is_triadic']]) 
                           for t in timeline)
        triadic_rate = (total_triadic / total_edges * 100) if total_edges > 0 else 0
        
        # Triangle count in final graph
        final_graph = timeline[-1]['graph']
        triangle_count = sum(nx.triangles(final_graph).values()) // 3
        
        # Growth patterns
        timesteps = [t['timestamp'] for t in timeline]
        edge_counts = [t['total_edges'] for t in timeline]
        
        triadic_counts = []
        running_triadic = 0
        for t in timeline:
            running_triadic += len([e for e in t['new_edges'] if e['is_triadic']])
            triadic_counts.append(running_triadic)
        
        return {
            'dataset_name': dataset_name,
            'timeline': timeline,
            'total_edges': total_edges,
            'total_triadic': total_triadic,
            'triadic_rate': triadic_rate,
            'triangle_count': triangle_count,
            'timesteps': timesteps,
            'edge_counts': edge_counts,
            'triadic_counts': triadic_counts,
            'initial_edges': timeline[0]['total_edges'],
            'final_nodes': timeline[-1]['total_nodes']
        }
    except Exception as e:
        print(f"Error analyzing {dataset_name}: {e}")
        return None

def create_triadic_comparison(datasets, output_file=None):
    """Create side-by-side comparison of triadic closure patterns."""
    
    print(f"Analyzing {len(datasets)} triadic closure datasets...")
    
    # Analyze all datasets
    analyses = []
    for dataset in datasets:
        analysis = analyze_triadic_pattern(dataset)
        if analysis:
            analyses.append(analysis)
    
    if len(analyses) == 0:
        print("No valid datasets found!")
        return
    
    print(f"Creating comparison visualization for {len(analyses)} datasets...")
    
    # Set up subplot grid
    n_datasets = len(analyses)
    cols = min(3, n_datasets)
    rows = (n_datasets + cols - 1) // cols
    
    fig = plt.figure(figsize=(6*cols, 8*rows))
    
    # Create subplots for each dataset
    for i, analysis in enumerate(analyses):
        # Graph visualization subplot
        ax_graph = plt.subplot2grid((rows*2, cols), (i//cols*2, i%cols))
        
        # Statistics subplot  
        ax_stats = plt.subplot2grid((rows*2, cols), (i//cols*2+1, i%cols))
        
        # Get final graph and layout
        final_graph = analysis['timeline'][-1]['graph']
        pos = nx.spring_layout(final_graph, k=2, iterations=50, seed=42)
        
        # === Graph Visualization ===
        
        # Draw base graph
        if final_graph.number_of_edges() > 0:
            nx.draw_networkx_edges(final_graph, pos, ax=ax_graph, 
                                 edge_color='gray', alpha=0.3, width=1)
        
        # Highlight triangle edges
        triangles_highlighted = set()
        for node in final_graph.nodes():
            for neighbor1 in final_graph.neighbors(node):
                for neighbor2 in final_graph.neighbors(node):
                    if (neighbor1 < neighbor2 and 
                        final_graph.has_edge(neighbor1, neighbor2)):
                        # This is a triangle
                        triangle_edges = [(node, neighbor1), 
                                        (node, neighbor2), 
                                        (neighbor1, neighbor2)]
                        
                        # Only highlight if not already done
                        triangle_key = tuple(sorted([node, neighbor1, neighbor2]))
                        if triangle_key not in triangles_highlighted:
                            nx.draw_networkx_edges(final_graph, pos, 
                                                 edgelist=triangle_edges,
                                                 ax=ax_graph, edge_color='red', 
                                                 width=2, alpha=0.8)
                            triangles_highlighted.add(triangle_key)
        
        # Draw nodes
        if final_graph.number_of_nodes() > 0:
            # Color nodes by triangle participation
            triangle_nodes = set()
            for triangle in triangles_highlighted:
                triangle_nodes.update(triangle)
            
            node_colors = ['orange' if node in triangle_nodes else 'lightblue' 
                          for node in final_graph.nodes()]
            
            nx.draw_networkx_nodes(final_graph, pos, node_color=node_colors, 
                                 ax=ax_graph, node_size=200, alpha=0.8)
            
            # Add node labels
            nx.draw_networkx_labels(final_graph, pos, ax=ax_graph, 
                                  font_size=6, font_weight='bold')
        
        ax_graph.set_title(f'{analysis["dataset_name"]}\n'
                          f'Nodes: {analysis["final_nodes"]}, '
                          f'Edges: {analysis["total_edges"]}\n'
                          f'Triangles: {analysis["triangle_count"]}, '
                          f'Triadic Rate: {analysis["triadic_rate"]:.1f}%',
                          fontsize=10, fontweight='bold')
        ax_graph.axis('off')
        
        # === Statistics Visualization ===
        
        ax_stats.plot(analysis['timesteps'], analysis['edge_counts'], 
                     'b-o', label='Total Edges', linewidth=2, markersize=3)
        ax_stats.plot(analysis['timesteps'], analysis['triadic_counts'], 
                     'r-s', label='Triadic Closures', linewidth=2, markersize=3)
        
        ax_stats.set_xlabel('Time', fontsize=9)
        ax_stats.set_ylabel('Count', fontsize=9)
        ax_stats.set_title('Growth Pattern', fontsize=10, fontweight='bold')
        ax_stats.legend(fontsize=8)
        ax_stats.grid(True, alpha=0.3)
        ax_stats.tick_params(labelsize=8)
    
    # Overall title
    fig.suptitle('Triadic Closure Pattern Comparison\n'
                f'Comparing {len(analyses)} different initialization/parameter settings',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file is None:
        output_file = f"triadic_patterns_comparison_{len(analyses)}_datasets.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved to: {output_file}")
    
    # Print summary statistics
    print("\n=== Triadic Closure Comparison Summary ===")
    for analysis in analyses:
        print(f"{analysis['dataset_name']:20}: "
              f"{analysis['triadic_rate']:5.1f}% triadic rate "
              f"({analysis['total_triadic']:2d}/{analysis['total_edges']:2d} edges), "
              f"{analysis['triangle_count']:2d} triangles")
    
    return output_file

def generate_diverse_triadic_datasets():
    """Generate several triadic closure datasets with different characteristics."""
    from generate_synthetic import generate_synthetic_data
    
    datasets_to_create = [
        {
            'name': 'triadic_sparse',
            'params': {'num_nodes': 25, 'num_initial_edges': 8, 'num_timesteps': 15, 'noise_ratio': 0.1}
        },
        {
            'name': 'triadic_medium', 
            'params': {'num_nodes': 30, 'num_initial_edges': 15, 'num_timesteps': 20, 'noise_ratio': 0.05}
        },
        {
            'name': 'triadic_dense',
            'params': {'num_nodes': 35, 'num_initial_edges': 25, 'num_timesteps': 25, 'noise_ratio': 0.02}
        }
    ]
    
    created_datasets = []
    for dataset_config in datasets_to_create:
        name = dataset_config['name']
        params = dataset_config['params']
        
        data_path = Path(f"processed/{name}")
        if not data_path.exists():
            print(f"Generating {name} with params: {params}")
            generate_synthetic_data(data_name=name, **params)
        else:
            print(f"Dataset {name} already exists, skipping generation")
        
        created_datasets.append(name)
    
    return created_datasets

def main():
    parser = argparse.ArgumentParser(description='Compare triadic closure patterns')
    parser.add_argument('datasets', nargs='*', 
                       help='Dataset names to compare (default: generate diverse set)')
    parser.add_argument('--output', '-o', help='Output file name')
    parser.add_argument('--generate', action='store_true',
                       help='Generate diverse datasets for comparison')
    
    args = parser.parse_args()
    
    if args.generate or len(args.datasets) == 0:
        print("Generating diverse triadic closure datasets...")
        datasets = generate_diverse_triadic_datasets()
    else:
        datasets = args.datasets
    
    try:
        create_triadic_comparison(datasets, args.output)
    except Exception as e:
        print(f"Error creating comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 