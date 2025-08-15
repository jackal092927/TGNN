#!/usr/bin/env python3
"""
Test TGNN models on Triadic Closure rule datasets.
Evaluates how well the model can identify which existing edges caused new edges to form.
"""

import argparse
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_triadic_closure_dataset(num_nodes=100, num_initial_edges=50, num_timesteps=100, noise_ratio=0.1):
    """Generate a triadic closure dataset using the existing function."""
    import sys
    sys.path.append('.')
    from generate_synthetic import generate_synthetic_data
    
    generate_synthetic_data(
        num_nodes=num_nodes,
        num_initial_edges=num_initial_edges, 
        num_timesteps=num_timesteps,
        noise_ratio=noise_ratio,
        data_name="triadic_closure"
    )
    print("✅ Triadic closure dataset generated!")

def load_triadic_closure_data():
    """Load triadic closure dataset and ground truth."""
    try:
        df = pd.read_csv('processed/triadic_closure/ml_triadic_closure.csv')
        with open('processed/triadic_closure/ml_triadic_closure_gt.json') as f:
            ground_truth = json.load(f)
        return df, ground_truth
    except FileNotFoundError:
        print("❌ Triadic closure dataset not found. Generating...")
        generate_triadic_closure_dataset()
        return load_triadic_closure_data()

def analyze_triadic_patterns(df, ground_truth):
    """Analyze the triadic closure patterns in the dataset."""
    print("📊 Triadic Closure Dataset Analysis:")
    print("=" * 50)
    
    print(f"📈 Dataset Overview:")
    print(f"  Total edges: {len(df)}")
    print(f"  Causal edges (with parents): {len(ground_truth)}")
    print(f"  Noise edges: {len(df) - len(ground_truth)}")
    print(f"  Triadic closure rate: {len(ground_truth)/len(df)*100:.1f}%")
    
    # Temporal analysis
    causal_edge_indices = [int(idx) for idx in ground_truth.keys()]
    causal_df = df[df.idx.isin(causal_edge_indices)]
    
    print(f"\n⏰ Temporal Pattern:")
    print(f"  First triadic closure: time {causal_df.ts.min():.2f}")
    print(f"  Last triadic closure: time {causal_df.ts.max():.2f}")
    print(f"  Peak time: {causal_df.ts.mode().iloc[0]:.2f}")
    
    # Closure dependency analysis
    parent_frequencies = defaultdict(int)
    for parents in ground_truth.values():
        for parent in parents:
            parent_frequencies[parent] += 1
    
    most_influential_edges = sorted(parent_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n🌟 Most Influential Edges (caused most closures):")
    for edge_idx, count in most_influential_edges:
        edge_info = df[df.idx == edge_idx].iloc[0]
        print(f"  Edge {edge_idx} ({edge_info.u}→{edge_info.i}): enabled {count} closures")
    
    return causal_df

def evaluate_triadic_explanation_accuracy(model, test_loader, num_neighbors, device, ground_truth):
    """
    Evaluate how well TGNN explanations match triadic closure ground truth.
    
    Ground truth format: {edge_idx: [parent_edge1, parent_edge2]}
    We need to check if model explanations highlight the nodes involved in parent edges.
    """
    print("\n🎯 Evaluating Triadic Closure Explanation Accuracy:")
    print("-" * 50)
    
    model.eval()
    all_precisions = []
    all_recalls = []
    all_mrrs = []
    evaluated_count = 0
    
    with torch.no_grad():
        for batch_idx, (src_l, dst_l, ts_l, e_idx_l, label_l) in enumerate(tqdm(test_loader, desc="Evaluating")):
            
            # Get model explanations (attention weights)
            _, attention_weights, neighbor_info = model.tem_conv_with_attn(
                src_l, ts_l, model.num_layers, num_neighbors
            )
            
            # Process each edge in the batch
            for i, edge_idx in enumerate(e_idx_l):
                edge_idx_str = str(int(edge_idx))
                
                if edge_idx_str not in ground_truth:
                    continue  # Skip non-causal edges
                
                parent_edges = ground_truth[edge_idx_str]
                if not parent_edges:
                    continue  # Skip edges without ground truth
                
                # Convert parent edges to parent nodes
                # For triadic closure: edge u→v is caused by edges that connect u,v via intermediate nodes
                ground_truth_nodes = set()
                
                # Load dataset to find parent edge nodes
                df = pd.read_csv('processed/triadic_closure/ml_triadic_closure.csv')
                for parent_edge_idx in parent_edges:
                    parent_edge = df[df.idx == parent_edge_idx]
                    if len(parent_edge) > 0:
                        parent_edge = parent_edge.iloc[0]
                        ground_truth_nodes.add(parent_edge.u)
                        ground_truth_nodes.add(parent_edge.i)
                
                if len(ground_truth_nodes) == 0:
                    continue
                
                # Get model's attention-based explanation
                if len(attention_weights) > 0 and len(neighbor_info) > 0:
                    attn = attention_weights[-1][i].cpu().numpy()  # Last layer attention
                    neighbors = neighbor_info[-1]['nodes'][i]
                    
                    # Filter out padding (node 0)
                    valid_mask = neighbors != 0
                    if valid_mask.sum() == 0:
                        continue
                    
                    valid_neighbors = neighbors[valid_mask]
                    valid_attention = attn[valid_mask]
                    
                    # Sort by attention weight (descending)
                    sorted_indices = np.argsort(valid_attention)[::-1]
                    sorted_neighbors = valid_neighbors[sorted_indices]
                    
                    # Calculate explanation metrics
                    ground_truth_list = list(ground_truth_nodes)
                    
                    # Precision@k and Recall@k
                    for k in [1, 2, 5]:
                        if len(sorted_neighbors) >= k:
                            top_k = set(sorted_neighbors[:k])
                            intersection = top_k.intersection(ground_truth_nodes)
                            precision_k = len(intersection) / k
                            recall_k = len(intersection) / len(ground_truth_nodes)
                            
                            if k == 2:  # Store P@2 and R@2 for averaging
                                all_precisions.append(precision_k)
                                all_recalls.append(recall_k)
                    
                    # Mean Reciprocal Rank (MRR)
                    mrr = 0
                    for gt_node in ground_truth_nodes:
                        if gt_node in sorted_neighbors:
                            rank = np.where(sorted_neighbors == gt_node)[0][0] + 1
                            mrr += 1 / rank
                    mrr /= len(ground_truth_nodes)
                    all_mrrs.append(mrr)
                    
                    evaluated_count += 1
    
    # Calculate final metrics
    if evaluated_count > 0:
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_mrr = np.mean(all_mrrs)
        
        print(f"📊 Triadic Closure Explanation Results:")
        print(f"  Evaluated edges: {evaluated_count}")
        print(f"  Average Precision@2: {avg_precision:.4f}")
        print(f"  Average Recall@2: {avg_recall:.4f}")
        print(f"  Average MRR: {avg_mrr:.4f}")
        
        return {
            'precision_at_2': avg_precision,
            'recall_at_2': avg_recall,
            'mrr': avg_mrr,
            'num_evaluated': evaluated_count
        }
    else:
        print("❌ No edges with ground truth found for evaluation")
        return None

def visualize_triadic_closure_example(df, ground_truth, num_examples=3):
    """Visualize examples of triadic closures."""
    print(f"\n🎨 Triadic Closure Examples:")
    print("-" * 30)
    
    example_count = 0
    for edge_idx_str, parent_edges in list(ground_truth.items())[:num_examples]:
        if not parent_edges:
            continue
            
        print(f"\n📍 Example {example_count + 1}:")
        
        # Get the child edge
        child_edge = df[df.idx == int(edge_idx_str)].iloc[0]
        print(f"  New edge: {child_edge.u} → {child_edge.i} (time {child_edge.ts:.2f})")
        
        # Get parent edges
        print(f"  Caused by parent edges:")
        for parent_idx in parent_edges:
            parent_edge = df[df.idx == parent_idx]
            if len(parent_edge) > 0:
                parent_edge = parent_edge.iloc[0]
                print(f"    Edge {parent_idx}: {parent_edge.u} → {parent_edge.i} (time {parent_edge.ts:.2f})")
        
        # Show the triadic pattern
        if len(parent_edges) == 2:
            parent1 = df[df.idx == parent_edges[0]].iloc[0]
            parent2 = df[df.idx == parent_edges[1]].iloc[0]
            
            # Find the common node (bridge)
            nodes1 = {parent1.u, parent1.i}
            nodes2 = {parent2.u, parent2.i}
            bridge_node = nodes1.intersection(nodes2)
            
            if bridge_node:
                bridge = list(bridge_node)[0]
                print(f"  💡 Triadic pattern: Bridge node {bridge} connected {child_edge.u} and {child_edge.i}")
        
        example_count += 1

def main():
    parser = argparse.ArgumentParser(description='Test TGNN on Triadic Closure datasets')
    parser.add_argument('--generate', action='store_true', help='Generate new triadic closure dataset')
    parser.add_argument('--analyze', action='store_true', help='Analyze triadic closure patterns')
    parser.add_argument('--visualize', action='store_true', help='Visualize triadic closure examples')
    parser.add_argument('--nodes', type=int, default=100, help='Number of nodes')
    parser.add_argument('--initial_edges', type=int, default=50, help='Initial edges')
    parser.add_argument('--timesteps', type=int, default=100, help='Number of timesteps')
    parser.add_argument('--noise_ratio', type=float, default=0.1, help='Noise ratio')
    
    args = parser.parse_args()
    
    if args.generate:
        print("🔄 Generating Triadic Closure Dataset...")
        generate_triadic_closure_dataset(
            num_nodes=args.nodes,
            num_initial_edges=args.initial_edges,
            num_timesteps=args.timesteps,
            noise_ratio=args.noise_ratio
        )
    
    if args.analyze or args.visualize:
        print("📊 Loading Triadic Closure Dataset...")
        df, ground_truth = load_triadic_closure_data()
        
        if args.analyze:
            analyze_triadic_patterns(df, ground_truth)
        
        if args.visualize:
            visualize_triadic_closure_example(df, ground_truth)
    
    if not any([args.generate, args.analyze, args.visualize]):
        print("Usage: python test_triadic_closure.py [--generate] [--analyze] [--visualize]")
        print("Example: python test_triadic_closure.py --generate --analyze --visualize")

if __name__ == "__main__":
    main() 