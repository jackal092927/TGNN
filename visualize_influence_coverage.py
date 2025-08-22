#!/usr/bin/env python3
"""
Visualize the influence coverage analysis results.
Create plots showing coverage rates, edge selection efficiency, and patterns across timestamps.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_coverage_results():
    """Load the coverage analysis results"""
    json_path = './results_triadic_long_dense/influence_coverage_summary.json'
    if not os.path.exists(json_path):
        print(f"Error: Results file not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    return results

def create_coverage_visualization(results):
    """Create comprehensive visualization of coverage results"""
    print("Creating influence coverage visualization...")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Coverage Rate vs Timestamp
    ax1.plot(df['timestamp'], df['coverage_rate'], 'o-', linewidth=2, markersize=8, color='darkblue')
    ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Target')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Coverage Rate')
    ax1.set_title('Influence Score Coverage Rate Across Timestamps')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # Highlight problematic timestamps
    low_coverage = df[df['coverage_rate'] < 0.5]
    if not low_coverage.empty:
        ax1.scatter(low_coverage['timestamp'], low_coverage['coverage_rate'], 
                   color='red', s=100, zorder=5, label='Low Coverage (<50%)')
        ax1.legend()
    
    # Plot 2: Edge Selection Percentage vs Timestamp
    ax2.plot(df['timestamp'], df['selected_edges_percentage'], 'o-', linewidth=2, markersize=8, color='darkgreen')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Edge Selection Percentage (%)')
    ax2.set_title('Percentage of Existing Edges Needed for 95% Coverage')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Highlight efficient selections
    efficient = df[df['selected_edges_percentage'] < 50]
    if not efficient.empty:
        ax2.scatter(efficient['timestamp'], efficient['selected_edges_percentage'], 
                   color='green', s=100, zorder=5, label='Efficient Selection (<50%)')
        ax2.legend()
    
    # Plot 3: Coverage vs Selection Efficiency Scatter
    ax3.scatter(df['selected_edges_percentage'], df['coverage_rate'], 
               c=df['timestamp'], cmap='viridis', s=100, alpha=0.7)
    ax3.set_xlabel('Edge Selection Percentage (%)')
    ax3.set_ylabel('Coverage Rate')
    ax3.set_title('Coverage Rate vs Edge Selection Efficiency')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    scatter = ax3.scatter(df['selected_edges_percentage'], df['coverage_rate'], 
                         c=df['timestamp'], cmap='viridis', s=100, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Timestamp')
    
    # Plot 4: Positive Pairs and Coverage by Phase
    # Define phases
    early_phase = df[df['timestamp'] <= 8]
    middle_phase = df[(df['timestamp'] > 8) & (df['timestamp'] <= 17)]
    late_phase = df[df['timestamp'] > 17]
    
    phases = [early_phase, middle_phase, late_phase]
    phase_names = ['Early (t=0-8)', 'Middle (t=9-17)', 'Late (t=18-27)']
    colors = ['skyblue', 'orange', 'red']
    
    for i, (phase, name, color) in enumerate(zip(phases, phase_names, colors)):
        if not phase.empty:
            avg_coverage = phase['coverage_rate'].mean()
            avg_selection = phase['selected_edges_percentage'].mean()
            total_pairs = phase['positive_pairs'].sum()
            
            ax4.bar(f'{name}\nCoverage', avg_coverage, color=color, alpha=0.7, 
                   label=f'{name}: {avg_coverage:.3f}')
            ax4.bar(f'{name}\nSelection', avg_selection/100, color=color, alpha=0.4)
    
    ax4.set_ylabel('Rate/Percentage')
    ax4.set_title('Average Performance by Network Evolution Phase')
    ax4.set_ylim(0, 1.05)
    ax4.legend()
    
    # Add text annotations for phase statistics
    for i, (phase, name) in enumerate(zip(phases, phase_names)):
        if not phase.empty:
            total_pairs = phase['positive_pairs'].sum()
            ax4.text(i*2 + 0.5, 0.9, f'Total Pairs: {total_pairs}', 
                    ha='center', va='top', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = './results_triadic_long_dense/influence_coverage_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    return output_path

def create_detailed_analysis_plot(results):
    """Create a detailed analysis plot focusing on key insights"""
    print("Creating detailed analysis plot...")
    
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Coverage and Selection Overlay
    x = df['timestamp']
    y1 = df['coverage_rate']
    y2 = df['selected_edges_percentage'] / 100  # Convert to 0-1 scale
    
    ax1.plot(x, y1, 'o-', linewidth=2, markersize=8, color='darkblue', label='Coverage Rate')
    ax1.plot(x, y2, 's-', linewidth=2, markersize=6, color='darkgreen', label='Edge Selection %')
    ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Coverage Target')
    
    # Highlight exceptional cases
    perfect_coverage = df[df['coverage_rate'] == 1.0]
    efficient_selection = df[df['selected_edges_percentage'] < 50]
    
    if not perfect_coverage.empty:
        ax1.scatter(perfect_coverage['timestamp'], perfect_coverage['coverage_rate'], 
                   color='gold', s=150, zorder=5, marker='*', label='Perfect Coverage (100%)')
    
    if not efficient_selection.empty:
        ax1.scatter(efficient_selection['timestamp'], efficient_selection['selected_edges_percentage']/100, 
                   color='lime', s=150, zorder=5, marker='^', label='Efficient Selection (<50%)')
    
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Rate/Percentage')
    ax1.set_title('Coverage Rate vs Edge Selection Efficiency Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Positive Pairs and Network Growth
    ax2_twin = ax2.twinx()
    
    # Bar plot for positive pairs
    bars = ax2.bar(x, df['positive_pairs'], alpha=0.6, color='lightblue', label='Positive Pairs')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Number of Positive Pairs', color='lightblue')
    ax2.tick_params(axis='y', labelcolor='lightblue')
    
    # Line plot for cumulative edges
    cumulative_edges = df['existing_edges_prev'].cumsum()
    line = ax2_twin.plot(x, cumulative_edges, 'o-', linewidth=2, markersize=6, 
                         color='darkred', label='Cumulative Existing Edges')
    ax2_twin.set_ylabel('Cumulative Existing Edges', color='darkred')
    ax2_twin.tick_params(axis='y', labelcolor='darkred')
    
    # Highlight massive spike at t=23
    if 23 in df['timestamp'].values:
        t23_idx = df[df['timestamp'] == 23].index[0]
        ax2.bar(23, df.loc[t23_idx, 'positive_pairs'], color='red', alpha=0.8, label='Massive Spike (t=23)')
    
    ax2.set_title('Positive Pairs vs Network Growth Over Time')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save the detailed plot
    output_path = './results_triadic_long_dense/influence_coverage_detailed_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Detailed analysis plot saved to: {output_path}")
    
    return output_path

def print_key_insights(results):
    """Print key insights from the analysis"""
    print("\n=== KEY INSIGHTS FROM INFLUENCE COVERAGE ANALYSIS ===\n")
    
    df = pd.DataFrame(results)
    
    # Overall statistics
    total_positive = df['positive_pairs'].sum()
    avg_coverage = df['coverage_rate'].mean()
    avg_selection = df['selected_edges_percentage'].mean()
    
    print(f"📊 OVERALL PERFORMANCE:")
    print(f"   Total Positive Pairs: {total_positive}")
    print(f"   Average Coverage Rate: {avg_coverage:.3f} ({avg_coverage*100:.1f}%)")
    print(f"   Average Edge Selection: {avg_selection:.1f}%")
    
    # Best and worst performers
    best_coverage = df.loc[df['coverage_rate'].idxmax()]
    worst_coverage = df.loc[df['coverage_rate'].idxmin()]
    
    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"   Timestamp {best_coverage['timestamp']}: {best_coverage['coverage_rate']:.3f} coverage")
    print(f"   Positive Pairs: {best_coverage['positive_pairs']}")
    print(f"   Edge Selection: {best_coverage['selected_edges_percentage']:.1f}%")
    
    print(f"\n⚠️  WORST PERFORMANCE:")
    print(f"   Timestamp {worst_coverage['timestamp']}: {worst_coverage['coverage_rate']:.3f} coverage")
    print(f"   Positive Pairs: {worst_coverage['positive_pairs']}")
    print(f"   Edge Selection: {worst_coverage['selected_edges_percentage']:.1f}%")
    
    # Most efficient selections
    efficient_selections = df[df['selected_edges_percentage'] < 50].sort_values('selected_edges_percentage')
    if not efficient_selections.empty:
        print(f"\n✨ MOST EFFICIENT EDGE SELECTIONS:")
        for _, row in efficient_selections.head(3).iterrows():
            print(f"   t={row['timestamp']}: {row['coverage_rate']:.3f} coverage with {row['selected_edges_percentage']:.1f}% selection")
    
    # Perfect coverage cases
    perfect_coverage = df[df['coverage_rate'] == 1.0]
    if not perfect_coverage.empty:
        print(f"\n🎯 PERFECT COVERAGE ACHIEVED AT:")
        for _, row in perfect_coverage.iterrows():
            print(f"   t={row['timestamp']}: {row['positive_pairs']} pairs, {row['selected_edges_percentage']:.1f}% selection")
    
    # Phase analysis
    early_phase = df[df['timestamp'] <= 8]
    middle_phase = df[(df['timestamp'] > 8) & (df['timestamp'] <= 17)]
    late_phase = df[df['timestamp'] > 17]
    
    print(f"\n📈 PHASE ANALYSIS:")
    for phase_name, phase_data in [("Early (t=0-8)", early_phase), 
                                  ("Middle (t=9-17)", middle_phase), 
                                  ("Late (t=18-27)", late_phase)]:
        if not phase_data.empty:
            phase_coverage = phase_data['coverage_rate'].mean()
            phase_selection = phase_data['selected_edges_percentage'].mean()
            phase_pairs = phase_data['positive_pairs'].sum()
            print(f"   {phase_name}: {phase_coverage:.3f} coverage, {phase_selection:.1f}% selection, {phase_pairs} pairs")

def main():
    """Main function to create visualizations and analysis"""
    print("=== INFLUENCE COVERAGE VISUALIZATION ===\n")
    
    # Load results
    results = load_coverage_results()
    if results is None:
        return
    
    # Create visualizations
    viz1_path = create_coverage_visualization(results)
    viz2_path = create_detailed_analysis_plot(results)
    
    # Print key insights
    print_key_insights(results)
    
    print(f"\n=== VISUALIZATION COMPLETE ===")
    print(f"Files generated:")
    print(f"  1. Comprehensive Coverage Analysis: {viz1_path}")
    print(f"  2. Detailed Analysis Plot: {viz2_path}")
    print(f"  3. Summary Document: influence_coverage_analysis_summary.md")

if __name__ == "__main__":
    main()
