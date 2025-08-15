#!/usr/bin/env python3
"""
Demonstration script to explain how ground truth explanations are constructed.
"""

import pandas as pd
import json
from collections import defaultdict

def analyze_ground_truth_explanations(dataset_name):
    """Analyze and explain the ground truth explanations for a dataset."""
    
    print(f"🔍 Analyzing Ground Truth Explanations for: {dataset_name}")
    print("=" * 60)
    
    # Load dataset and explanations
    df = pd.read_csv(f'processed/{dataset_name}/ml_{dataset_name}.csv')
    
    with open(f'processed/{dataset_name}/{dataset_name}_explanations.json') as f:
        explanations = json.load(f)
    
    print(f"Dataset Overview:")
    print(f"  📊 Total edges: {len(df)}")
    print(f"  ✅ Causal edges (label=1): {len(df[df.label == 1])}")
    print(f"  ❌ Non-causal edges (label=0): {len(df[df.label == 0])}")
    print(f"  📝 Explanations provided: {len([e for e in explanations.values() if e])}")
    print()
    
    # Group edges by explanation (to show collective activation events)
    explanation_groups = defaultdict(list)
    for idx, explanation in explanations.items():
        if explanation:  # Skip empty explanations
            explanation_key = tuple(sorted(explanation))
            explanation_groups[explanation_key].append(int(idx))
    
    print(f"🎯 Activation Events Analysis:")
    print(f"  🔄 Unique activation events: {len(explanation_groups)}")
    print()
    
    # Show detailed examples
    print("📋 Detailed Examples:")
    print("-" * 40)
    
    # Show first few activation events
    for i, (explanation_nodes, edge_indices) in enumerate(list(explanation_groups.items())[:5]):
        print(f"\n🌟 Activation Event #{i+1}:")
        print(f"   Explaining nodes: {list(explanation_nodes)}")
        print(f"   Number of edges created: {len(edge_indices)}")
        print(f"   Edge indices: {edge_indices}")
        
        # Show the actual edges
        event_edges = df[df.idx.isin(edge_indices)].sort_values('idx')
        print(f"   📍 Edges in dataset:")
        for _, edge in event_edges.iterrows():
            print(f"      Edge {edge.idx}: {edge.u} → {edge.i} at time {edge.ts}")
        
        # Determine activated node(s)
        activated_nodes = set(event_edges['i'].unique())
        print(f"   🎯 Activated node(s): {activated_nodes}")
        print(f"   💡 Interpretation: Nodes {list(explanation_nodes)} collectively")
        print(f"      activated node(s) {activated_nodes} at time {event_edges.ts.iloc[0]}")
    
    print("\n" + "=" * 60)
    
    # Show example of non-causal edges
    non_causal_edges = df[df.label == 0].head(3)
    if len(non_causal_edges) > 0:
        print("❌ Non-Causal Edge Examples:")
        print("-" * 30)
        for _, edge in non_causal_edges.iterrows():
            explanation = explanations.get(str(edge.idx), [])
            print(f"   Edge {edge.idx}: {edge.u} → {edge.i} at time {edge.ts}")
            print(f"   Explanation: {explanation} (empty = non-causal)")
        print()
    
    # Time analysis
    causal_df = df[df.label == 1]
    if len(causal_df) > 0:
        print(f"⏰ Temporal Analysis:")
        print(f"   First activation: time {causal_df.ts.min():.2f}")
        print(f"   Last activation: time {causal_df.ts.max():.2f}")
        print(f"   Most active time: {causal_df.ts.mode().iloc[0]:.2f}")
        
        # Count activations per timestep
        activations_per_time = causal_df.groupby('ts').size()
        print(f"   Peak activations: {activations_per_time.max()} edges at time {activations_per_time.idxmax():.2f}")
    
    print("\n" + "=" * 60)

def compare_explanation_formats(dataset_name):
    """Show how the same activation event appears in different formats."""
    
    print(f"🔄 Explanation Format Comparison for: {dataset_name}")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(f'processed/{dataset_name}/ml_{dataset_name}.csv')
    with open(f'processed/{dataset_name}/{dataset_name}_explanations.json') as f:
        explanations = json.load(f)
    
    # Find a good example (multiple edges with same explanation)
    explanation_counts = defaultdict(int)
    for explanation in explanations.values():
        if explanation:
            explanation_counts[tuple(sorted(explanation))] += 1
    
    # Get the most common explanation
    most_common_explanation = max(explanation_counts.items(), key=lambda x: x[1])
    example_nodes = list(most_common_explanation[0])
    
    print(f"📝 Example: Nodes {example_nodes} activate a target node")
    print()
    
    # Find all edges with this explanation
    example_edge_indices = []
    for idx, explanation in explanations.items():
        if explanation and tuple(sorted(explanation)) == tuple(sorted(example_nodes)):
            example_edge_indices.append(int(idx))
    
    example_edges = df[df.idx.isin(example_edge_indices[:4])].sort_values('idx')  # Just show first 4
    
    print("🎯 How this appears in different formats:")
    print()
    
    print("1️⃣ Simulation Format (during contagion):")
    if len(example_edges) > 0:
        target_node = example_edges.iloc[0]['i']
        timestamp = example_edges.iloc[0]['ts']
        print(f"   activation = (node={target_node}, timestep={timestamp}, explanation={example_nodes})")
    print()
    
    print("2️⃣ Dataset Format (ml_*.csv):")
    for _, edge in example_edges.iterrows():
        print(f"   {edge.u},{edge.i},{edge.ts},{edge.label},{edge.idx}")
    print()
    
    print("3️⃣ Ground Truth Format (*_explanations.json):")
    for _, edge in example_edges.iterrows():
        print(f"   \"{edge.idx}\": {example_nodes}")
    print()
    
    print("4️⃣ Evaluation Format (what TGNN sees):")
    for _, edge in example_edges.iterrows():
        print(f"   For edge {edge.idx} ({edge.u}→{edge.i}): Should highlight nodes {example_nodes}")
    print()

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python explain_ground_truth.py <dataset_name>")
        print("Example: python explain_ground_truth.py synthetic_ltm_ba")
        return
    
    dataset_name = sys.argv[1]
    
    try:
        analyze_ground_truth_explanations(dataset_name)
        print()
        compare_explanation_formats(dataset_name)
        
        print("\n🎓 Key Takeaways:")
        print("=" * 40)
        print("✅ Each edge index maps to nodes that caused its formation")
        print("✅ Multiple edges can share the same explanation (collective activation)")
        print("✅ Empty explanations [] indicate non-causal edges")
        print("✅ Ground truth reflects activation-level, not edge-level causality")
        print("✅ TGNN explanations should highlight the ground truth nodes")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Dataset '{dataset_name}' not found.")
        print(f"Available datasets: synthetic_ltm_ba, synthetic_icm_ba, synthetic_cc_ws, synthetic_sd_ba")
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")

if __name__ == "__main__":
    main() 