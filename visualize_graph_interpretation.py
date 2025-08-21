#!/usr/bin/env python3
"""
Graph-level interpretation visualization for GraphMamba models.
Shows edge mask attention and node importance for link prediction.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path

def load_interpretation_data(results_file):
    """Load interpretation data from results file"""
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    details = results.get('details', {})
    if not details:
        print("❌ No interpretation details found in results")
        return None
    
    print(f"✅ Loaded interpretation data:")
    print(f"   - {len(details['predictions'])} predictions")
    print(f"   - {len(details['gates'])} gate sets")
    print(f"   - {len(details['embeddings'])} embedding sets")
    
    return details

def create_graph_attention_visualization(details, output_dir, max_examples=5):
    """Create visualization of edge attention masks for link prediction"""
    print(f"\nCreating graph attention visualizations...")
    
    # Get unique timestamps
    unique_timestamps = sorted(set(details['timestamps']))
    print(f"   - {len(unique_timestamps)} unique timestamps: {unique_timestamps}")
    
    # Create output directory
    attention_dir = os.path.join(output_dir, 'attention_visualizations')
    os.makedirs(attention_dir, exist_ok=True)
    
    # Process each timestamp
    for ts_idx, timestamp in enumerate(unique_timestamps[:max_examples]):
        print(f"   Processing timestamp {timestamp}...")
        
        # Get data for this timestamp
        ts_mask = [i for i, ts in enumerate(details['timestamps']) if ts == timestamp]
        
        if len(ts_mask) == 0:
            continue
        
        # Get gates and pairs for this timestamp
        gates = details['gates'][ts_mask[0]]  # Use first occurrence
        pairs = [details['pairs'][i] for i in ts_mask]
        predictions = [details['predictions'][i] for i in ts_mask]
        labels = [details['labels'][i] for i in ts_mask]
        
        # Convert gates back to numpy array if it's a list
        if isinstance(gates, list):
            gates = np.array(gates)
        
        # Create graph from gates
        N = gates.shape[0]
        G = nx.Graph()
        
        # Add nodes
        for i in range(N):
            G.add_node(i)
        
        # Add edges with attention weights
        edge_weights = []
        for i in range(N):
            for j in range(i+1, N):
                weight = float(gates[i, j])
                if weight > 0.01:  # Only show edges with significant attention
                    G.add_edge(i, j, weight=weight)
                    edge_weights.append(weight)
        
        if len(edge_weights) == 0:
            print(f"     No significant edges found for timestamp {timestamp}")
            continue
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Graph with edge attention
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Node colors based on degree (importance)
        node_degrees = dict(G.degree())
        max_degree = max(node_degrees.values()) if node_degrees else 1
        node_colors = [node_degrees.get(node, 0) / max_degree for node in G.nodes()]
        
        # Edge colors based on attention weights
        edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=300, cmap='viridis', ax=ax1)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                              edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=max(edge_weights),
                              width=3, alpha=0.7, ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
        
        ax1.set_title(f'Edge Attention at Timestamp {timestamp}\nNode size = degree, Edge color = attention', 
                     fontweight='bold', fontsize=12)
        ax1.axis('off')
        
        # Plot 2: Attention weight distribution
        ax2.hist(edge_weights, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Edge Attention Weights', fontweight='bold')
        ax2.set_xlabel('Attention Weight')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {np.mean(edge_weights):.3f}\nStd: {np.std(edge_weights):.3f}\nMax: {np.max(edge_weights):.3f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(attention_dir, f'timestamp_{timestamp}_attention.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Saved: {output_file}")
    
    return attention_dir

def create_comprehensive_edge_visualization(details, output_dir):
    """Create comprehensive visualization showing all edges with confidence-based coloring"""
    print(f"\nCreating comprehensive edge confidence visualization...")
    
    # Create output directory
    edge_dir = os.path.join(output_dir, 'edge_confidence_visualization')
    os.makedirs(edge_dir, exist_ok=True)
    
    # Get all predictions and labels
    predictions = np.array(details['predictions'])
    labels = np.array(details['labels'])
    pairs = details['pairs']
    timestamps = details['timestamps']
    
    print(f"   - Total predictions: {len(predictions)}")
    print(f"   - Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")
    print(f"   - Average prediction: {predictions.mean():.3f}")
    
    # Create a comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Prediction distribution histogram
    ax1.hist(predictions, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('Distribution of Model Predictions', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Frequency')
    ax1.axvline(predictions.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {predictions.mean():.3f}')
    ax1.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Labels
    ax2.scatter(labels, predictions, alpha=0.6, c=predictions, cmap='viridis')
    ax2.set_title('Predictions vs True Labels', fontweight='bold', fontsize=14)
    ax2.set_xlabel('True Label (0/1)')
    ax2.set_ylabel('Model Prediction')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Negative (0)', 'Positive (1)'])
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confidence heatmap for all edges
    # Create a matrix showing prediction confidence for each edge
    unique_timestamps = sorted(set(timestamps))
    ts_idx = 0  # Use first timestamp for visualization
    
    # Get gates for this timestamp
    ts_mask = [i for i, ts in enumerate(timestamps) if ts == unique_timestamps[ts_idx]]
    if ts_mask:
        gates = details['gates'][ts_mask[0]]
        if isinstance(gates, list):
            gates = np.array(gates)
        
        N = gates.shape[0]
        
        # Create edge confidence matrix
        edge_confidence = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    edge_confidence[i, j] = gates[i, j]
        
        # Plot heatmap
        im = ax3.imshow(edge_confidence, cmap='viridis', aspect='auto')
        ax3.set_title(f'Edge Attention Weights at Timestamp {unique_timestamps[ts_idx]:.2f}', 
                     fontweight='bold', fontsize=14)
        ax3.set_xlabel('Target Node')
        ax3.set_ylabel('Source Node')
        ax3.set_xticks(range(0, N, 5))
        ax3.set_yticks(range(0, N, 5))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Plot 4: Top attention edges
    # Get all edge attention weights and sort them
    all_edge_weights = []
    all_edge_pairs = []
    
    for ts_idx, ts in enumerate(unique_timestamps[:5]):  # Use first 5 timestamps
        ts_mask = [i for i, t in enumerate(timestamps) if t == ts]
        if ts_mask:
            gates = details['gates'][ts_mask[0]]
            if isinstance(gates, list):
                gates = np.array(gates)
            
            N = gates.shape[0]
            for i in range(N):
                for j in range(i+1, N):
                    weight = gates[i, j]
                    if weight > 0.01:  # Only significant edges
                        all_edge_weights.append(weight)
                        all_edge_pairs.append(f"({i},{j})")
    
    if all_edge_weights:
        # Sort by weight and take top 20
        sorted_indices = np.argsort(all_edge_weights)[::-1]
        top_weights = [all_edge_weights[i] for i in sorted_indices[:20]]
        top_pairs = [all_edge_pairs[i] for i in sorted_indices[:20]]
        
        bars = ax4.bar(range(len(top_weights)), top_weights, color='lightcoral', alpha=0.7)
        ax4.set_title('Top 20 Edge Attention Weights (All Timestamps)', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Edge (u,v)')
        ax4.set_xlabel('Edge (u,v)')
        ax4.set_ylabel('Attention Weight')
        ax4.set_xticks(range(len(top_weights)))
        ax4.set_xticklabels(top_pairs, rotation=45, ha='right')
        
        # Add value labels
        for j, weight in enumerate(top_weights):
            ax4.text(j, weight + 0.01, f'{weight:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No significant edges found', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('No Significant Edges', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(edge_dir, 'comprehensive_edge_confidence.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {output_file}")
    
    return edge_dir


def create_top_influential_edges_visualization(details, output_dir, top_k=5):
    """Create visualization showing top K most influential edges for each link prediction"""
    print(f"\nCreating top {top_k} influential edges visualization for each prediction...")
    
    # Create output directory
    influential_dir = os.path.join(output_dir, 'top_influential_edges')
    os.makedirs(influential_dir, exist_ok=True)
    
    # Get all predictions and labels
    predictions = np.array(details['predictions'])
    labels = np.array(details['labels'])
    pairs = details['pairs']
    timestamps = details['timestamps']
    
    print(f"   - Total predictions: {len(predictions)}")
    print(f"   - Creating individual plots for each prediction...")
    
    # Process each prediction to find top influential edges
    for i in range(len(predictions)):
        prediction = predictions[i]
        label = labels[i]
        pair = pairs[i]
        gates = details['gates'][i]
        timestamp = timestamps[i]
        
        # Convert gates back to numpy array if it's a list
        if isinstance(gates, list):
            gates = np.array(gates)
        
        N = gates.shape[0]
        
        # Find top K edges with highest attention weights
        edge_weights = []
        edge_pairs = []
        
        for u in range(N):
            for v in range(u+1, N):
                weight = gates[u, v]
                if weight > 0.001:  # Only edges with some attention
                    edge_weights.append(weight)
                    edge_pairs.append((u, v))
        
        if len(edge_weights) == 0:
            continue
        
        # Sort by weight and take top K
        sorted_indices = np.argsort(edge_weights)[::-1]
        top_weights = [edge_weights[j] for j in sorted_indices[:top_k]]
        top_pairs = [edge_pairs[j] for j in sorted_indices[:top_k]]
        
        # Create visualization for this prediction
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Top K influential edges bar chart
        bars = ax1.bar(range(len(top_weights)), top_weights, 
                       color=['red' if (u in pair or v in pair) else 'skyblue' 
                              for (u, v) in top_pairs], alpha=0.7)
        
        ax1.set_title(f'Top {top_k} Influential Edges for Link Prediction\n'
                     f'Predicted Pair: {pair}, Prediction: {prediction:.3f}, True: {label}\n'
                     f'Timestamp: {timestamp:.2f}', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Edge Rank')
        ax1.set_ylabel('Attention Weight')
        ax1.set_xticks(range(len(top_weights)))
        ax1.set_xticklabels([f'#{j+1}' for j in range(len(top_weights))])
        
        # Add value labels on bars
        for j, weight in enumerate(top_weights):
            ax1.text(j, weight + 0.001, f'{weight:.3f}', 
                    ha='center', va='bottom', fontsize=10)
        
        # Highlight edges connected to predicted pair
        for j, (u, v) in enumerate(top_pairs):
            if u in pair or v in pair:
                bars[j].set_color('red')
                bars[j].set_alpha(0.9)
                bars[j].set_edgecolor('darkred')
                bars[j].set_linewidth(2)
        
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Edge attention heatmap focused on top edges
        # Create a focused heatmap showing only the top influential edges
        focused_heatmap = np.zeros((N, N))
        
        # Add the top edges
        for (u, v), weight in zip(top_pairs, top_weights):
            focused_heatmap[u, v] = weight
            focused_heatmap[v, u] = weight  # Make symmetric
        
        # Highlight the predicted pair
        focused_heatmap[pair[0], pair[1]] = 2.0  # Special value for predicted pair
        focused_heatmap[pair[1], pair[0]] = 2.0
        
        im = ax2.imshow(focused_heatmap, cmap='viridis', aspect='auto')
        ax2.set_title(f'Top {top_k} Influential Edges Heatmap\n'
                     f'Red bars = edges connected to predicted pair', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Target Node')
        ax2.set_ylabel('Source Node')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # Add text annotations for top edges
        for j, ((u, v), weight) in enumerate(zip(top_pairs, top_weights)):
            ax2.text(v, u, f'{weight:.3f}', ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
        
        # Highlight predicted pair
        ax2.text(pair[1], pair[0], f'PREDICTED\n{pair}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(influential_dir, f'prediction_{i+1:03d}_pair_{pair[0]}_{pair[1]}_top{top_k}_edges.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print progress every 10 predictions
        if (i + 1) % 10 == 0:
            print(f"     Processed {i + 1}/{len(predictions)} predictions...")
    
    print(f"   ✅ Created {len(predictions)} individual plots")
    
    # Create a summary plot showing distribution of top edge attention weights
    print(f"   Creating summary statistics...")
    
    # Collect all top edge weights across all predictions
    all_top_weights = []
    all_top_ranks = []
    
    for i in range(len(predictions)):
        gates = details['gates'][i]
        if isinstance(gates, list):
            gates = np.array(gates)
        
        N = gates.shape[0]
        edge_weights = []
        
        for u in range(N):
            for v in range(u+1, N):
                weight = gates[u, v]
                if weight > 0.001:
                    edge_weights.append(weight)
        
        if len(edge_weights) > 0:
            sorted_weights = sorted(edge_weights, reverse=True)
            for rank in range(min(top_k, len(sorted_weights))):
                all_top_weights.append(sorted_weights[rank])
                all_top_ranks.append(rank + 1)
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Distribution of top edge attention weights
    ax1.hist(all_top_weights, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
    ax1.set_title(f'Distribution of Top {top_k} Edge Attention Weights\nAcross All Predictions', 
                  fontweight='bold', fontsize=14)
    ax1.set_xlabel('Attention Weight')
    ax1.set_ylabel('Frequency')
    ax1.axvline(np.mean(all_top_weights), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(all_top_weights):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average attention weight by rank
    rank_means = []
    rank_stds = []
    for rank in range(1, top_k + 1):
        rank_weights = [w for w, r in zip(all_top_weights, all_top_ranks) if r == rank]
        if rank_weights:
            rank_means.append(np.mean(rank_weights))
            rank_stds.append(np.std(rank_weights))
        else:
            rank_means.append(0)
            rank_stds.append(0)
    
    x_pos = range(1, top_k + 1)
    bars = ax2.bar(x_pos, rank_means, yerr=rank_stds, capsize=5, 
                   color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_title(f'Average Attention Weight by Rank\nAcross All Predictions', 
                  fontweight='bold', fontsize=14)
    ax2.set_xlabel('Edge Rank')
    ax2.set_ylabel('Average Attention Weight')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'#{r}' for r in x_pos])
    
    # Add value labels on bars
    for j, mean_val in enumerate(rank_means):
        if mean_val > 0:
            ax2.text(j + 1, mean_val + 0.001, f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontsize=10)
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_file = os.path.join(influential_dir, f'summary_top{top_k}_edges_statistics.png')
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved summary statistics: {summary_file}")
    
    return influential_dir


def create_comprehensive_graph_visualization(details, output_dir, top_k=5):
    """Create comprehensive graph visualization showing all elements on one drawing"""
    print(f"\nCreating comprehensive graph visualization for each prediction...")
    
    # Create output directory
    graph_dir = os.path.join(output_dir, 'comprehensive_graph_visualizations')
    os.makedirs(graph_dir, exist_ok=True)
    
    # Get all predictions and labels
    predictions = np.array(details['predictions'])
    labels = np.array(details['labels'])
    pairs = details['pairs']
    timestamps = details['timestamps']
    
    print(f"   - Total predictions: {len(predictions)}")
    print(f"   - Creating comprehensive graph plots for each prediction...")
    
    # Process each prediction to create comprehensive graph visualization
    for i in range(len(predictions)):
        prediction = predictions[i]
        label = labels[i]
        pair = pairs[i]
        gates = details['gates'][i]
        timestamp = timestamps[i]
        
        # Convert gates back to numpy array if it's a list
        if isinstance(gates, list):
            gates = np.array(gates)
        
        N = gates.shape[0]
        
        # Find top K edges with highest attention weights
        edge_weights = []
        edge_pairs = []
        
        for u in range(N):
            for v in range(u+1, N):
                weight = gates[u, v]
                if weight > 0.001:  # Only edges with some attention
                    edge_weights.append(weight)
                    edge_pairs.append((u, v))
        
        if len(edge_weights) == 0:
            continue
        
        # Sort by weight and take top K
        sorted_indices = np.argsort(edge_weights)[::-1]
        top_weights = [edge_weights[j] for j in sorted_indices[:top_k]]
        top_pairs = [edge_pairs[j] for j in sorted_indices[:top_k]]
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add all nodes
        for node in range(N):
            G.add_node(node)
        
        # Add all existing edges (light grey)
        all_edges = []
        for u in range(N):
            for v in range(u+1, N):
                if gates[u, v] > 0.001:  # Any edge with attention
                    all_edges.append((u, v))
                    G.add_edge(u, v, weight=gates[u, v], color='lightgrey', width=1)
        
        # Create the comprehensive visualization
        plt.figure(figsize=(16, 12))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw all existing edges (light grey)
        nx.draw_networkx_edges(G, pos, 
                              edge_color='lightgrey', 
                              width=1, 
                              alpha=0.3,
                              style='solid')
        
        # Draw top K influential edges with attention-based coloring
        top_edge_colors = []
        top_edge_widths = []
        
        for weight in top_weights:
            # Normalize weight to [0.5, 3] for width
            width = 0.5 + (weight / max(top_weights)) * 2.5
            top_edge_widths.append(width)
            
            # Color based on attention weight (red for high, orange for medium, yellow for low)
            if weight >= 0.8 * max(top_weights):
                color = 'red'
            elif weight >= 0.6 * max(top_weights):
                color = 'orange'
            elif weight >= 0.4 * max(top_weights):
                color = 'gold'
            else:
                color = 'yellow'
            top_edge_colors.append(color)
        
        # Draw top edges with attention-based styling
        for (u, v), color, width in zip(top_pairs, top_edge_colors, top_edge_widths):
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=[(u, v)], 
                                  edge_color=color, 
                                  width=width, 
                                  alpha=0.8,
                                  style='solid')
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for node in range(N):
            if node in pair:
                # Predicted pair nodes: larger and colored
                if label == 1:  # Positive
                    node_colors.append('green')
                    node_sizes.append(800)
                else:  # Negative
                    node_colors.append('red')
                    node_sizes.append(800)
            else:
                # Other nodes: smaller and grey
                node_colors.append('lightblue')
                node_sizes.append(400)
        
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors, 
                              node_size=node_sizes, 
                              alpha=0.8)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Add edge labels for top K edges
        edge_labels = {}
        for (u, v), weight in zip(top_pairs, top_weights):
            edge_labels[(u, v)] = f'{weight:.3f}'
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, font_weight='bold')
        
        # Create comprehensive title
        title = f'Comprehensive Graph Visualization\n'
        title += f'Predicted Pair: {pair} | Prediction: {prediction:.3f} | True Label: {label} | Timestamp: {timestamp:.2f}\n'
        title += f'Top {top_k} Influential Edges Highlighted (Red=High, Orange=Medium, Yellow=Low Attention)'
        
        plt.title(title, fontweight='bold', fontsize=14, pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='lightgrey', lw=2, label='All Existing Edges'),
            plt.Line2D([0], [0], color='red', lw=3, label='Top Edge (High Attention)'),
            plt.Line2D([0], [0], color='orange', lw=2.5, label='Top Edge (Medium Attention)'),
            plt.Line2D([0], [0], color='gold', lw=2, label='Top Edge (Low Attention)'),
            plt.scatter([], [], c='green', s=200, label='Predicted Pair (Positive)'),
            plt.scatter([], [], c='red', s=200, label='Predicted Pair (Negative)'),
            plt.scatter([], [], c='lightblue', s=100, label='Other Nodes')
        ]
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        # Add attention weight summary
        attention_text = f'Top {top_k} Edge Attention Weights:\n'
        for j, ((u, v), weight) in enumerate(zip(top_pairs, top_weights)):
            attention_text += f'#{j+1}: ({u},{v}) = {weight:.3f}\n'
        
        plt.figtext(0.02, 0.02, attention_text, fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(graph_dir, f'comprehensive_prediction_{i+1:03d}_pair_{pair[0]}_{pair[1]}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print progress every 10 predictions
        if (i + 1) % 10 == 0:
            print(f"     Processed {i + 1}/{len(predictions)} predictions...")
    
    print(f"   ✅ Created {len(predictions)} comprehensive graph visualizations")
    
    return graph_dir


def create_temporal_attention_evolution(details, output_dir):
    """Create visualization showing how attention patterns evolve over time"""
    print(f"\nCreating temporal attention evolution visualization...")
    
    # Create output directory
    temporal_dir = os.path.join(output_dir, 'temporal_evolution')
    os.makedirs(temporal_dir, exist_ok=True)
    
    # Get unique timestamps
    unique_timestamps = sorted(set(details['timestamps']))
    
    if len(unique_timestamps) < 2:
        print("   Need at least 2 timestamps for temporal analysis")
        return temporal_dir
    
    # Calculate average attention per timestamp
    timestamp_attention = {}
    for ts in unique_timestamps:
        ts_mask = [i for i, t in enumerate(details['timestamps']) if t == ts]
        if ts_mask:
            gates_for_ts = [details['gates'][i] for i in ts_mask]
            # Convert lists to numpy arrays if needed
            gates_for_ts = [np.array(g) if isinstance(g, list) else g for g in gates_for_ts]
            avg_gates = np.mean(gates_for_ts, axis=0)
            timestamp_attention[ts] = avg_gates
    
    # Create temporal evolution plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Average attention over time
    timestamps_sorted = sorted(timestamp_attention.keys())
    avg_attention_values = [np.mean(timestamp_attention[ts]) for ts in timestamps_sorted]
    
    ax1.plot(timestamps_sorted, avg_attention_values, 'b-o', linewidth=2, markersize=6)
    ax1.set_title('Evolution of Average Edge Attention Over Time', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Average Attention Weight')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Attention heatmap over time
    attention_matrix = np.array([timestamp_attention[ts].flatten() for ts in timestamps_sorted])
    
    im = ax2.imshow(attention_matrix, cmap='viridis', aspect='auto', 
                    extent=[0, attention_matrix.shape[1], timestamps_sorted[0], timestamps_sorted[-1]])
    ax2.set_title('Edge Attention Heatmap Over Time', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Edge Index (flattened)')
    ax2.set_ylabel('Timestamp')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Attention Weight')
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(temporal_dir, 'temporal_attention_evolution.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {output_file}")
    
    return temporal_dir

def main():
    """Main function to create all interpretation visualizations"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize GraphMamba interpretation results')
    parser.add_argument('--results_file', required=True, help='Path to results JSON file')
    parser.add_argument('--output_dir', help='Output directory for visualizations (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    print("🧪 GRAPH-LEVEL INTERPRETATION VISUALIZATION")
    print("="*60)
    
    # Check for results file
    results_file = args.results_file
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("Please run training first to generate interpretation data.")
        return
    
    # Load interpretation data
    details = load_interpretation_data(results_file)
    if not details:
        return
    
    # Auto-generate output directory based on experiment timestamp
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Extract experiment timestamp from results file path
        results_path = Path(results_file)
        experiment_dir = results_path.parent.name
        
        # Check if it contains a timestamp pattern (e.g., icm_viz_enhanced_20240816_1430)
        if '_' in experiment_dir and any(char.isdigit() for char in experiment_dir):
            # Extract the timestamp part more robustly
            parts = experiment_dir.split('_')
            # Look for the last two parts that form a timestamp (YYYYMMDD_HHMMSS)
            for i in range(len(parts) - 1):
                if (len(parts[i]) == 8 and parts[i].isdigit() and 
                    len(parts[i+1]) == 6 and parts[i+1].isdigit()):
                    timestamp_part = f"{parts[i]}_{parts[i+1]}"
                    break
            else:
                # Fallback: use last two parts
                timestamp_part = f"{parts[-2]}_{parts[-1]}"
            
            output_dir = f"./experiments/{experiment_dir}/interpretation_visualizations_{timestamp_part}"
        else:
            # Fallback to default naming
            output_dir = f"./experiments/{experiment_dir}/interpretation_visualizations"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Results file: {results_file}")
    
    # Create different types of visualizations
    try:
        # 1. Edge attention visualizations
        attention_dir = create_graph_attention_visualization(details, output_dir, max_examples=3)
        
        # 2. Comprehensive edge confidence visualization
        edge_dir = create_comprehensive_edge_visualization(details, output_dir)
        
        # 3. Top influential edges for each prediction
        influential_dir = create_top_influential_edges_visualization(details, output_dir, top_k=5)
        
        # 4. Comprehensive graph visualization (all elements on one drawing)
        graph_dir = create_comprehensive_graph_visualization(details, output_dir, top_k=5)
        
        # 5. Temporal attention evolution
        temporal_dir = create_temporal_attention_evolution(details, output_dir)
        
        print(f"\n🎉 ALL INTERPRETATION VISUALIZATIONS COMPLETED!")
        print(f"   📊 Edge attention: {attention_dir}")
        print(f"   🔗 Edge confidence: {edge_dir}")
        print(f"   🎯 Top influential edges: {influential_dir}")
        print(f"   🕸️  Comprehensive graph: {graph_dir}")
        print(f"   ⏰ Temporal evolution: {temporal_dir}")
        
        # List all generated files
        print(f"\n📁 Generated files:")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"   ✅ {file_path} - {file_size} bytes")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
