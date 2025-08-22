#!/usr/bin/env python3
"""
Analyze triadic network evolution statistics.
Count new edges added at each timestamp and provide insights into network growth patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_and_analyze_triadic_data():
    """Load triadic dataset and analyze network evolution"""
    data_path = './processed/triadic_perfect_long_dense/ml_triadic_perfect_long_dense.csv'
    g_df = pd.read_csv(data_path)
    
    print("=== TRIADIC NETWORK EVOLUTION ANALYSIS ===\n")
    
    # Get unique timestamps
    timestamps = sorted(g_df['ts'].unique())
    print(f"Dataset spans {len(timestamps)} timestamps: {timestamps}")
    
    # Count edges at each timestamp
    edge_counts = {}
    cumulative_edges = {}
    new_edges_per_timestamp = {}
    total_edges_so_far = 0
    
    print(f"\n{'Timestamp':<10} {'New Edges':<12} {'Total Edges':<12} {'Growth Rate':<12}")
    print("-" * 50)
    
    for ts in timestamps:
        edges_at_ts = g_df[g_df['ts'] == ts]
        new_edges = len(edges_at_ts)
        total_edges_so_far += new_edges
        
        edge_counts[ts] = new_edges
        cumulative_edges[ts] = total_edges_so_far
        new_edges_per_timestamp[ts] = new_edges
        
        # Calculate growth rate (percentage increase)
        if ts == 0:
            growth_rate = 0.0
        else:
            prev_total = cumulative_edges[ts-1]
            growth_rate = (new_edges / prev_total * 100) if prev_total > 0 else 0.0
        
        print(f"{ts:<10} {new_edges:<12} {total_edges_so_far:<12} {growth_rate:>8.1f}%")
    
    return g_df, timestamps, edge_counts, cumulative_edges, new_edges_per_timestamp

def analyze_network_phases(timestamps, edge_counts, cumulative_edges):
    """Analyze different phases of network evolution"""
    print(f"\n=== NETWORK EVOLUTION PHASES ===\n")
    
    # Find peak activity periods
    peak_timestamps = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Peak Activity Timestamps (most new edges):")
    for ts, count in peak_timestamps:
        print(f"  t={ts}: {count} new edges")
    
    # Analyze early, middle, and late phases
    total_timestamps = len(timestamps)
    early_cutoff = total_timestamps // 3
    middle_cutoff = 2 * total_timestamps // 3
    
    early_phase = timestamps[:early_cutoff]
    middle_phase = timestamps[early_cutoff:middle_cutoff]
    late_phase = timestamps[middle_cutoff:]
    
    print(f"\nPhase Analysis:")
    print(f"  Early Phase (t={early_phase[0]}-{early_phase[-1]}): {len(early_phase)} timestamps")
    print(f"  Middle Phase (t={middle_phase[0]}-{middle_phase[-1]}): {len(middle_phase)} timestamps")
    print(f"  Late Phase (t={late_phase[0]}-{late_phase[-1]}): {len(late_phase)} timestamps")
    
    # Calculate edge counts for each phase
    early_edges = sum(edge_counts[ts] for ts in early_phase)
    middle_edges = sum(edge_counts[ts] for ts in middle_phase)
    late_edges = sum(edge_counts[ts] for ts in late_phase)
    
    print(f"\nEdge Distribution by Phase:")
    print(f"  Early Phase: {early_edges} edges ({early_edges/sum(edge_counts.values())*100:.1f}%)")
    print(f"  Middle Phase: {middle_edges} edges ({middle_edges/sum(edge_counts.values())*100:.1f}%)")
    print(f"  Late Phase: {late_edges} edges ({late_edges/sum(edge_counts.values())*100:.1f}%)")

def create_visualization(timestamps, edge_counts, cumulative_edges):
    """Create visualization of network evolution"""
    print(f"\n=== CREATING NETWORK EVOLUTION VISUALIZATION ===\n")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: New edges per timestamp
    ax1.bar(timestamps, [edge_counts[ts] for ts in timestamps], 
            color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('New Edges Added')
    ax1.set_title('Network Evolution: New Edges per Timestamp')
    ax1.grid(True, alpha=0.3)
    
    # Highlight peak timestamps
    peak_timestamps = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    for ts, count in peak_timestamps:
        ax1.bar(ts, count, color='red', alpha=0.8, label=f'Peak: t={ts}' if ts == peak_timestamps[0][0] else "")
    
    ax1.legend()
    
    # Plot 2: Cumulative edges over time
    ax2.plot(timestamps, [cumulative_edges[ts] for ts in timestamps], 
             marker='o', linewidth=2, markersize=6, color='darkgreen')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Total Edges (Cumulative)')
    ax2.set_title('Network Growth: Cumulative Edges Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Add growth rate annotations
    for i, ts in enumerate(timestamps[::3]):  # Every 3rd timestamp
        if ts > 0:
            growth_rate = (edge_counts[ts] / cumulative_edges[ts-1] * 100) if cumulative_edges[ts-1] > 0 else 0
            ax2.annotate(f'{growth_rate:.1f}%', 
                        xy=(ts, cumulative_edges[ts]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = './results_triadic_long_dense/network_evolution_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    return output_path

def generate_summary_statistics(timestamps, edge_counts, cumulative_edges):
    """Generate comprehensive summary statistics"""
    print(f"\n=== SUMMARY STATISTICS ===\n")
    
    total_edges = sum(edge_counts.values())
    avg_edges_per_timestamp = total_edges / len(timestamps)
    
    # Calculate variance and standard deviation
    edge_values = list(edge_counts.values())
    variance = np.var(edge_values)
    std_dev = np.std(edge_values)
    
    print(f"Total Edges: {total_edges}")
    print(f"Total Timestamps: {len(timestamps)}")
    print(f"Average Edges per Timestamp: {avg_edges_per_timestamp:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Variance: {variance:.2f}")
    
    # Find timestamps with no new edges
    zero_edge_timestamps = [ts for ts, count in edge_counts.items() if count == 0]
    if zero_edge_timestamps:
        print(f"Timestamps with No New Edges: {zero_edge_timestamps}")
    else:
        print("All timestamps have new edges")
    
    # Find the most active timestamp
    most_active_ts = max(edge_counts.items(), key=lambda x: x[1])
    print(f"Most Active Timestamp: t={most_active_ts[0]} with {most_active_ts[1]} new edges")
    
    # Calculate growth acceleration
    print(f"\nGrowth Pattern Analysis:")
    for i in range(1, len(timestamps)):
        ts = timestamps[i]
        prev_ts = timestamps[i-1]
        current_growth = edge_counts[ts]
        prev_growth = edge_counts[prev_ts]
        
        if prev_growth > 0:
            acceleration = current_growth - prev_growth
            if abs(acceleration) > 10:  # Significant change
                direction = "↑" if acceleration > 0 else "↓"
                print(f"  t={ts}: {direction} {abs(acceleration)} edges change from t={prev_ts}")

def main():
    """Main analysis function"""
    # Load and analyze data
    g_df, timestamps, edge_counts, cumulative_edges, new_edges_per_timestamp = load_and_analyze_triadic_data()
    
    # Analyze network phases
    analyze_network_phases(timestamps, edge_counts, cumulative_edges)
    
    # Generate summary statistics
    generate_summary_statistics(timestamps, edge_counts, cumulative_edges)
    
    # Create visualization
    viz_path = create_visualization(timestamps, edge_counts, cumulative_edges)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Network evolution analysis saved to: {viz_path}")
    print(f"Key insights:")
    print(f"  - Dataset spans {len(timestamps)} timestamps")
    print(f"  - Total {sum(edge_counts.values())} edges added")
    print(f"  - Peak activity identified and highlighted")
    print(f"  - Growth patterns analyzed across phases")

if __name__ == "__main__":
    main()
