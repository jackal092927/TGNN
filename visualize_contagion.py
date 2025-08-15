#!/usr/bin/env python3
"""
Visualize synthetic contagion datasets as animated GIFs.
Shows the temporal evolution of node activations in the network.
"""

import argparse
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import os
from pathlib import Path

def load_dataset_info(dataset_name):
    """Load dataset and ground truth information."""
    base_path = Path(f"processed/{dataset_name}")
    
    # Load the edge data
    df = pd.read_csv(base_path / f"ml_{dataset_name}.csv")
    
    # Load ground truth explanations
    gt_file = base_path / f"{dataset_name}_explanations.json"
    ground_truth = {}
    if gt_file.exists():
        with open(gt_file) as f:
            ground_truth = json.load(f)
    
    return df, ground_truth

def reconstruct_network_timeline(df, ground_truth):
    """Reconstruct the temporal network and activation timeline."""
    
    # Get all unique nodes
    all_nodes = set(df['u'].unique()) | set(df['i'].unique())
    node_list = sorted(list(all_nodes))
    
    # Create base graph from all edges
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['u'], row['i'])
    
    # Extract activation events from causal edges (label=1)
    activations = []
    causal_df = df[df['label'] == 1].copy()
    
    for _, row in causal_df.iterrows():
        edge_idx = str(int(row['idx']))
        explanation = ground_truth.get(edge_idx, [])
        
        # The activated node is the target (i), and explanation contains the activating neighbors
        activations.append({
            'time': row['ts'],
            'activated_node': row['i'],
            'activating_nodes': explanation,
            'edge_idx': edge_idx
        })
    
    # Sort by time
    activations.sort(key=lambda x: x['time'])
    
    # Create timeline
    timeline = []
    activated_nodes = set()
    
    # Add initial state
    timeline.append({
        'time': 0.0,
        'activated': set(),
        'newly_activated': set(),
        'activating_edges': set()
    })
    
    for activation in activations:
        activated_nodes.add(activation['activated_node'])
        
        timeline.append({
            'time': activation['time'],
            'activated': activated_nodes.copy(),
            'newly_activated': {activation['activated_node']},
            'activating_edges': {(src, activation['activated_node']) for src in activation['activating_nodes']}
        })
    
    return G, timeline, node_list

def create_contagion_animation(dataset_name, output_file=None, figsize=(12, 8), interval=1000):
    """Create animated GIF of contagion process."""
    
    print(f"Loading dataset: {dataset_name}")
    df, ground_truth = load_dataset_info(dataset_name)
    
    print("Reconstructing network timeline...")
    G, timeline, node_list = reconstruct_network_timeline(df, ground_truth)
    
    if len(timeline) <= 1:
        print("No activation events found in dataset!")
        return
    
    print(f"Found {len(timeline)-1} activation events")
    
    # Set up the plot
    fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=figsize, 
                                           gridspec_kw={'width_ratios': [4, 1]})
    
    # Create layout - use spring layout for better visualization
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Color scheme
    colors = {
        'inactive': 'lightgray',
        'active': 'red',
        'newly_active': 'orange',
        'activating_edge': 'red'
    }
    
    def animate(frame):
        ax_main.clear()
        ax_legend.clear()
        
        if frame >= len(timeline):
            frame = len(timeline) - 1
            
        state = timeline[frame]
        
        # Determine node colors
        node_colors = []
        for node in G.nodes():
            if node in state['newly_activated']:
                node_colors.append(colors['newly_active'])
            elif node in state['activated']:
                node_colors.append(colors['active'])
            else:
                node_colors.append(colors['inactive'])
        
        # Draw network
        nx.draw_networkx_edges(G, pos, ax=ax_main, edge_color='lightgray', alpha=0.5)
        
        # Highlight activating edges
        if state['activating_edges']:
            edge_list = [(u, v) for u, v in state['activating_edges'] if G.has_edge(u, v)]
            if edge_list:
                nx.draw_networkx_edges(G, pos, edgelist=edge_list, ax=ax_main, 
                                     edge_color=colors['activating_edge'], width=3, alpha=0.8)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax_main, 
                             node_size=300, alpha=0.8)
        
        # Add node labels
        nx.draw_networkx_labels(G, pos, ax=ax_main, font_size=8)
        
        # Title and info
        model_type = dataset_name.split('_')[1] if '_' in dataset_name else 'unknown'
        ax_main.set_title(f'{model_type.upper()} Contagion - Time: {state["time"]:.2f}\n'
                         f'Active: {len(state["activated"])} nodes', 
                         fontsize=14, fontweight='bold')
        ax_main.axis('off')
        
        # Legend
        legend_elements = [
            plt.Circle((0, 0), 0.1, color=colors['inactive'], label='Inactive'),
            plt.Circle((0, 0), 0.1, color=colors['active'], label='Active'),
            plt.Circle((0, 0), 0.1, color=colors['newly_active'], label='Newly Active'),
            plt.Line2D([0], [0], color=colors['activating_edge'], linewidth=3, label='Activating Edge')
        ]
        
        ax_legend.legend(handles=legend_elements, loc='center', fontsize=10)
        ax_legend.axis('off')
        
        # Add stats
        stats_text = f"Frame: {frame+1}/{len(timeline)}\n"
        stats_text += f"Total Nodes: {len(node_list)}\n"
        stats_text += f"Total Edges: {G.number_of_edges()}\n"
        stats_text += f"Activations: {len([t for t in timeline if t['newly_activated']])}"
        
        ax_legend.text(0.1, 0.3, stats_text, transform=ax_legend.transAxes, 
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(timeline), 
                                 interval=interval, repeat=True, blit=False)
    
    # Save as GIF
    if output_file is None:
        output_file = f"{dataset_name}_contagion.gif"
    
    print(f"Saving animation to: {output_file}")
    fps = max(1, 1000//interval)  # Ensure fps is at least 1
    anim.save(output_file, writer='pillow', fps=fps)
    plt.close()
    
    print(f"Animation saved successfully!")
    return output_file

def create_static_summary(dataset_name, output_file=None):
    """Create a static summary plot showing before/after states."""
    
    print(f"Creating static summary for: {dataset_name}")
    df, ground_truth = load_dataset_info(dataset_name)
    G, timeline, node_list = reconstruct_network_timeline(df, ground_truth)
    
    if len(timeline) <= 1:
        print("No activation events found!")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Initial state
    initial_state = timeline[0]
    node_colors_initial = ['lightgray' for _ in node_list]
    
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='lightgray', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_initial, ax=ax1, node_size=300)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8)
    ax1.set_title(f'Initial State (t=0)\nAll nodes inactive', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Final state
    final_state = timeline[-1]
    node_colors_final = []
    for node in node_list:
        if node in final_state['activated']:
            node_colors_final.append('red')
        else:
            node_colors_final.append('lightgray')
    
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='lightgray', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_final, ax=ax2, node_size=300)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=8)
    ax2.set_title(f'Final State (t={final_state["time"]:.2f})\n{len(final_state["activated"])} nodes activated', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Overall title
    model_type = dataset_name.split('_')[1] if '_' in dataset_name else 'unknown'
    fig.suptitle(f'{model_type.upper()} Contagion Process Summary', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file is None:
        output_file = f"{dataset_name}_summary.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Static summary saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Visualize synthetic contagion datasets')
    parser.add_argument('dataset', help='Dataset name (e.g., synthetic_ltm_ba)')
    parser.add_argument('--output', '-o', help='Output file name (default: auto-generated)')
    parser.add_argument('--static', action='store_true', help='Create static summary instead of animation')
    parser.add_argument('--interval', type=int, default=1500, help='Animation interval in ms (default: 1500)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], help='Figure size (default: 12 8)')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    dataset_path = Path(f"processed/{args.dataset}")
    if not dataset_path.exists():
        print(f"Error: Dataset {args.dataset} not found at {dataset_path}")
        return
    
    try:
        if args.static:
            create_static_summary(args.dataset, args.output)
        else:
            create_contagion_animation(args.dataset, args.output, 
                                     tuple(args.figsize), args.interval)
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 