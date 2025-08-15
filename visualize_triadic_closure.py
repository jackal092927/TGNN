#!/usr/bin/env python3
"""
Visualize Triadic Closure rule as animated GIFs.
Shows how triangles form over time in dynamic graphs.
"""

import argparse
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import os
from pathlib import Path
from collections import defaultdict

def load_or_generate_triadic_data(dataset_name="triadic_closure"):
    """Load triadic closure dataset or generate if it doesn't exist."""
    data_path = Path(f"processed/{dataset_name}")
    
    if not data_path.exists():
        print(f"Generating triadic closure dataset: {dataset_name}")
        from generate_synthetic import generate_synthetic_data
        generate_synthetic_data(
            num_nodes=30,
            num_initial_edges=15,
            num_timesteps=20,
            noise_ratio=0.05,
            data_name=dataset_name
        )
    
    # Load data
    df = pd.read_csv(data_path / f"ml_{dataset_name}.csv")
    
    # Load ground truth
    gt_file = data_path / f"ml_{dataset_name}_gt.json"
    ground_truth = {}
    if gt_file.exists():
        with open(gt_file) as f:
            ground_truth = json.load(f)
    
    return df, ground_truth

def reconstruct_triadic_timeline(df, ground_truth):
    """Reconstruct how the graph evolved through triadic closures."""
    
    # Get unique timestamps
    timestamps = sorted(df['ts'].unique())
    
    # Initialize graph timeline
    timeline = []
    G = nx.Graph()
    
    for ts in timestamps:
        # Get edges at this timestamp
        edges_at_time = df[df['ts'] == ts].sort_values('idx')
        
        # Track new edges and their causality
        new_edges = []
        new_triangles = []
        
        for _, edge_row in edges_at_time.iterrows():
            u, v = edge_row['u'], edge_row['i']
            edge_idx = str(edge_row['idx'])
            
            # Check if this edge creates a triadic closure
            is_triadic = edge_idx in ground_truth and ground_truth[edge_idx]
            
            # Add edge to graph
            G.add_edge(u, v)
            
            new_edge_info = {
                'u': u,
                'v': v,
                'idx': edge_row['idx'],
                'is_triadic': is_triadic,
                'parent_edges': ground_truth.get(edge_idx, [])
            }
            
            # If triadic closure, find the triangle
            if is_triadic and len(ground_truth[edge_idx]) == 2:
                parent_edges = ground_truth[edge_idx]
                # Find parent edge nodes to identify the bridge
                parent_nodes = set()
                for parent_idx in parent_edges:
                    parent_edge = df[df['idx'] == parent_idx]
                    if len(parent_edge) > 0:
                        parent_edge = parent_edge.iloc[0]
                        parent_nodes.add(parent_edge['u'])
                        parent_nodes.add(parent_edge['i'])
                
                # The triangle is formed by the new edge (u,v) and the parent nodes
                triangle_nodes = list(parent_nodes | {u, v})
                if len(triangle_nodes) == 3:
                    new_edge_info['triangle'] = triangle_nodes
                    new_triangles.append(triangle_nodes)
            
            new_edges.append(new_edge_info)
        
        timeline.append({
            'timestamp': ts,
            'graph': G.copy(),
            'new_edges': new_edges,
            'new_triangles': new_triangles,
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges()
        })
    
    return timeline

def create_triadic_closure_animation(dataset_name, output_file=None, figsize=(14, 10), interval=2000):
    """Create animated GIF showing triadic closure evolution."""
    
    print(f"Loading triadic closure dataset: {dataset_name}")
    df, ground_truth = load_or_generate_triadic_data(dataset_name)
    
    print("Reconstructing triadic closure timeline...")
    timeline = reconstruct_triadic_timeline(df, ground_truth)
    
    if len(timeline) <= 1:
        print("No temporal evolution found!")
        return
    
    print(f"Found {len(timeline)} timesteps with {len([e for t in timeline for e in t['new_edges'] if e['is_triadic']])} triadic closures")
    
    # Set up the plot
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[2, 2, 1])
    
    ax_main = fig.add_subplot(gs[0, :2])  # Main graph
    ax_info = fig.add_subplot(gs[0, 2])   # Info panel
    ax_stats = fig.add_subplot(gs[1, :])  # Statistics
    
    # Get all nodes for consistent layout
    final_graph = timeline[-1]['graph']
    pos = nx.spring_layout(final_graph, k=2, iterations=50, seed=42)
    
    # Color scheme
    colors = {
        'existing_node': 'lightblue',
        'existing_edge': 'gray',
        'new_edge': 'red',
        'triadic_edge': 'darkred', 
        'triangle_highlight': 'yellow',
        'bridge_node': 'orange'
    }
    
    def animate(frame):
        # Clear all axes
        ax_main.clear()
        ax_info.clear()
        ax_stats.clear()
        
        if frame >= len(timeline):
            frame = len(timeline) - 1
        
        state = timeline[frame]
        G = state['graph']
        timestamp = state['timestamp']
        new_edges = state['new_edges']
        new_triangles = state['new_triangles']
        
        # === Main Graph Visualization ===
        
        # Draw existing edges (only if edges exist)
        if G.number_of_edges() > 0:
            nx.draw_networkx_edges(G, pos, ax=ax_main, edge_color=colors['existing_edge'], 
                                 alpha=0.3, width=1)
        
        # Highlight new edges
        for edge_info in new_edges:
            u, v = edge_info['u'], edge_info['v']
            if G.has_edge(u, v):
                edge_color = colors['triadic_edge'] if edge_info['is_triadic'] else colors['new_edge']
                edge_width = 4 if edge_info['is_triadic'] else 2
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax_main,
                                     edge_color=edge_color, width=edge_width, alpha=0.8)
        
        # Highlight triangles
        for triangle in new_triangles:
            if len(triangle) == 3:
                triangle_edges = [(triangle[i], triangle[j]) 
                                for i in range(3) for j in range(i+1, 3)
                                if G.has_edge(triangle[i], triangle[j])]
                
                # Draw triangle background
                triangle_pos = [pos[node] for node in triangle if node in pos]
                if len(triangle_pos) == 3:
                    triangle_patch = plt.Polygon(triangle_pos, alpha=0.2, 
                                               facecolor=colors['triangle_highlight'], 
                                               edgecolor='none')
                    ax_main.add_patch(triangle_patch)
        
        # Draw nodes (only if nodes exist)
        if G.number_of_nodes() > 0:
            node_colors = []
            for node in G.nodes():
                # Check if node is a bridge in any new triadic closure
                is_bridge = False
                for edge_info in new_edges:
                    if edge_info['is_triadic'] and 'triangle' in edge_info:
                        triangle = edge_info['triangle']
                        if node in triangle and node not in [edge_info['u'], edge_info['v']]:
                            is_bridge = True
                            break
                
                if is_bridge:
                    node_colors.append(colors['bridge_node'])
                else:
                    node_colors.append(colors['existing_node'])
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax_main, 
                                 node_size=400, alpha=0.8)
            
            # Add node labels
            nx.draw_networkx_labels(G, pos, ax=ax_main, font_size=8, font_weight='bold')
        
        ax_main.set_title(f'Triadic Closure Evolution - Time {timestamp:.1f}\n'
                         f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}',
                         fontsize=14, fontweight='bold')
        ax_main.axis('off')
        
        # === Info Panel ===
        info_text = f"Timestep: {timestamp:.1f}\n\n"
        
        if new_edges:
            info_text += "New Edges:\n"
            for i, edge_info in enumerate(new_edges[:3]):  # Show max 3
                edge_type = "🔺 Triadic" if edge_info['is_triadic'] else "➕ Regular"
                info_text += f"{edge_type}: {edge_info['u']}→{edge_info['v']}\n"
            
            if len(new_edges) > 3:
                info_text += f"... +{len(new_edges)-3} more\n"
        
        if new_triangles:
            info_text += f"\n🔺 New Triangles: {len(new_triangles)}\n"
            for triangle in new_triangles[:2]:
                info_text += f"△ {'-'.join(map(str, triangle))}\n"
        
        # Count triadic closures so far
        total_triadic = sum(len([e for e in t['new_edges'] if e['is_triadic']]) 
                           for t in timeline[:frame+1])
        info_text += f"\nTotal Triadic Closures: {total_triadic}"
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        ax_info.axis('off')
        
        # === Statistics Panel ===
        
        # Prepare data for stats
        timesteps = [t['timestamp'] for t in timeline[:frame+1]]
        node_counts = [t['total_nodes'] for t in timeline[:frame+1]]
        edge_counts = [t['total_edges'] for t in timeline[:frame+1]]
        triadic_counts = []
        
        running_triadic = 0
        for t in timeline[:frame+1]:
            running_triadic += len([e for e in t['new_edges'] if e['is_triadic']])
            triadic_counts.append(running_triadic)
        
        # Plot statistics
        ax_stats.clear()
        ax_stats.plot(timesteps, edge_counts, 'b-o', label='Total Edges', markersize=4)
        ax_stats.plot(timesteps, triadic_counts, 'r-s', label='Triadic Closures', markersize=4)
        
        ax_stats.set_xlabel('Time')
        ax_stats.set_ylabel('Count')
        ax_stats.set_title('Graph Growth Statistics', fontsize=12, fontweight='bold')
        ax_stats.legend(loc='upper left')
        ax_stats.grid(True, alpha=0.3)
        
        # Add current values as text
        if frame < len(timeline):
            current_edges = edge_counts[-1] if edge_counts else 0
            current_triadic = triadic_counts[-1] if triadic_counts else 0
            triadic_rate = (current_triadic / current_edges * 100) if current_edges > 0 else 0
            
            stats_text = f"Triadic Rate: {triadic_rate:.1f}%"
            ax_stats.text(0.98, 0.95, stats_text, transform=ax_stats.transAxes,
                         horizontalalignment='right', verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(timeline), 
                                 interval=interval, repeat=True, blit=False)
    
    # Save as GIF
    if output_file is None:
        output_file = f"{dataset_name}_triadic_closure.gif"
    
    print(f"Saving animation to: {output_file}")
    fps = max(1, 1000//interval)
    anim.save(output_file, writer='pillow', fps=fps)
    plt.close()
    
    print(f"Animation saved successfully!")
    return output_file

def create_triadic_static_summary(dataset_name, output_file=None):
    """Create static summary showing before/after and key patterns."""
    
    print(f"Creating static summary for: {dataset_name}")
    df, ground_truth = load_or_generate_triadic_data(dataset_name)
    timeline = reconstruct_triadic_timeline(df, ground_truth)
    
    if len(timeline) <= 1:
        print("No temporal evolution found!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Initial and final states
    initial_state = timeline[0]
    final_state = timeline[-1]
    
    # Consistent layout
    pos = nx.spring_layout(final_state['graph'], k=2, iterations=50, seed=42)
    
    # Initial state
    ax = axes[0, 0]
    G_initial = initial_state['graph']
    nx.draw_networkx_edges(G_initial, pos, ax=ax, edge_color='gray', alpha=0.5)
    nx.draw_networkx_nodes(G_initial, pos, ax=ax, node_color='lightblue', node_size=300)
    nx.draw_networkx_labels(G_initial, pos, ax=ax, font_size=8)
    ax.set_title(f'Initial State (t={initial_state["timestamp"]:.1f})\n'
                f'{G_initial.number_of_nodes()} nodes, {G_initial.number_of_edges()} edges',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Final state with triangles highlighted
    ax = axes[0, 1]
    G_final = final_state['graph']
    
    # Find all triangles in final graph
    triangles = list(nx.triangles(G_final).items())
    triangle_nodes = set()
    for node, triangle_count in triangles:
        if triangle_count > 0:
            triangle_nodes.add(node)
    
    # Draw graph
    nx.draw_networkx_edges(G_final, pos, ax=ax, edge_color='gray', alpha=0.5)
    
    # Highlight triangle edges
    for node in G_final.nodes():
        for neighbor1 in G_final.neighbors(node):
            for neighbor2 in G_final.neighbors(node):
                if neighbor1 < neighbor2 and G_final.has_edge(neighbor1, neighbor2):
                    # This is a triangle: node-neighbor1-neighbor2
                    triangle_edges = [(node, neighbor1), (node, neighbor2), (neighbor1, neighbor2)]
                    nx.draw_networkx_edges(G_final, pos, edgelist=triangle_edges, 
                                         ax=ax, edge_color='red', width=2, alpha=0.8)
    
    # Color nodes by triangle participation
    node_colors = ['orange' if node in triangle_nodes else 'lightblue' 
                   for node in G_final.nodes()]
    nx.draw_networkx_nodes(G_final, pos, node_color=node_colors, ax=ax, node_size=300)
    nx.draw_networkx_labels(G_final, pos, ax=ax, font_size=8)
    
    ax.set_title(f'Final State (t={final_state["timestamp"]:.1f})\n'
                f'{G_final.number_of_nodes()} nodes, {G_final.number_of_edges()} edges',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Growth statistics
    ax = axes[1, 0]
    timesteps = [t['timestamp'] for t in timeline]
    edge_counts = [t['total_edges'] for t in timeline]
    
    triadic_counts = []
    running_triadic = 0
    for t in timeline:
        running_triadic += len([e for e in t['new_edges'] if e['is_triadic']])
        triadic_counts.append(running_triadic)
    
    ax.plot(timesteps, edge_counts, 'b-o', label='Total Edges', linewidth=2)
    ax.plot(timesteps, triadic_counts, 'r-s', label='Triadic Closures', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.set_title('Graph Growth Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Triadic closure examples
    ax = axes[1, 1]
    ax.text(0.05, 0.95, "Triadic Closure Process:", fontsize=14, fontweight='bold',
            transform=ax.transAxes, verticalalignment='top')
    
    example_text = """
1. Initial edges: A—B, B—C
2. Triadic opportunity: A and C not connected
3. New edge formed: A—C  
4. Triangle completed: A—B—C—A

Ground Truth: Edge A→C caused by edges A→B and B→C

Key insight: Existing structure creates 
opportunities for new connections!
    """
    
    ax.text(0.05, 0.85, example_text, fontsize=11, transform=ax.transAxes,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')
    
    # Overall title and statistics
    total_triadic = triadic_counts[-1] if triadic_counts else 0
    total_edges = edge_counts[-1] if edge_counts else 0
    triadic_rate = (total_triadic / total_edges * 100) if total_edges > 0 else 0
    
    fig.suptitle(f'Triadic Closure Analysis: {dataset_name}\n'
                f'Triadic Closure Rate: {triadic_rate:.1f}% ({total_triadic}/{total_edges} edges)',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file is None:
        output_file = f"{dataset_name}_triadic_summary.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Static summary saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Visualize Triadic Closure evolution')
    parser.add_argument('dataset', nargs='?', default='triadic_closure', 
                       help='Dataset name (default: triadic_closure)')
    parser.add_argument('--output', '-o', help='Output file name')
    parser.add_argument('--static', action='store_true', 
                       help='Create static summary instead of animation')
    parser.add_argument('--interval', type=int, default=2500, 
                       help='Animation interval in ms (default: 2500)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[14, 10], 
                       help='Figure size (default: 14 10)')
    
    args = parser.parse_args()
    
    try:
        if args.static:
            create_triadic_static_summary(args.dataset, args.output)
        else:
            create_triadic_closure_animation(args.dataset, args.output, 
                                           tuple(args.figsize), args.interval)
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 