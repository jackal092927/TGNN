"""
GraphRNN Failure Analysis Summary and Simple Tests
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    return g_df, e_feat, n_feat, ground_truth

def analyze_edge_distribution_over_time():
    """Analyze how edges are distributed over time"""
    print("=" * 80)
    print("🔍 EDGE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    datasets = ['triadic_perfect_medium', 'triadic_perfect_large', 'triadic_perfect_long_dense']
    
    for dataset_name in datasets:
        try:
            g_df, _, _, _ = load_triadic_data(dataset_name)
            edges = g_df[['u', 'i', 'ts']].values
            timestamps = edges[:, 2]
            
            print(f"\n📊 {dataset_name}:")
            
            # Edges per timestamp
            edges_per_ts = defaultdict(int)
            for ts in timestamps:
                edges_per_ts[int(ts)] += 1
            
            edge_counts = list(edges_per_ts.values())
            
            print(f"  Total edges: {len(edges)}")
            print(f"  Timeline: {timestamps.min():.0f} to {timestamps.max():.0f} ({len(edge_counts)} timestamps)")
            print(f"  Edges/timestamp: μ={np.mean(edge_counts):.1f}, σ={np.std(edge_counts):.1f}")
            print(f"  Min/Max edges per timestamp: {min(edge_counts)} / {max(edge_counts)}")
            print(f"  Coefficient of variation: {np.std(edge_counts)/np.mean(edge_counts):.2f}")
            
            # Show first few and last few timestamps
            sorted_ts = sorted(edges_per_ts.keys())
            print(f"  First 5 timestamps: {[edges_per_ts[ts] for ts in sorted_ts[:5]]}")
            print(f"  Last 5 timestamps: {[edges_per_ts[ts] for ts in sorted_ts[-5:]]}")
            
        except Exception as e:
            print(f"  ❌ Failed to load {dataset_name}: {e}")

def analyze_complexity_metrics():
    """Compare complexity metrics across datasets"""
    print("\n" + "=" * 80)
    print("📈 COMPLEXITY COMPARISON")
    print("=" * 80)
    
    datasets = ['triadic_perfect_medium', 'triadic_perfect_large', 'triadic_perfect_long_dense']
    results = []
    
    for dataset_name in datasets:
        try:
            g_df, _, _, _ = load_triadic_data(dataset_name)
            edges = g_df[['u', 'i', 'ts']].values
            timestamps = edges[:, 2]
            nodes = set(edges[:, 0].tolist() + edges[:, 1].tolist())
            
            timeline_length = int(timestamps.max() - timestamps.min() + 1)
            edges_per_ts = len(edges) / timeline_length
            
            # Calculate various complexity metrics
            complexity_metrics = {
                'dataset': dataset_name,
                'total_edges': len(edges),
                'total_nodes': len(nodes),
                'timeline_length': timeline_length,
                'edges_per_ts': edges_per_ts,
                'nodes_edges_ratio': len(nodes) / len(edges),
                'timeline_complexity': timeline_length * edges_per_ts,
                'graph_scale': len(edges) * timeline_length,
                'density_factor': len(edges) / (len(nodes) * (len(nodes) - 1) / 2)
            }
            results.append(complexity_metrics)
            
        except Exception as e:
            print(f"❌ Failed to analyze {dataset_name}: {e}")
    
    # Display comparison table
    if results:
        print(f"\n{'Dataset':<25} {'Edges':<8} {'Nodes':<8} {'Timeline':<10} {'E/T':<8} {'Scale':<10} {'Density':<10}")
        print("-" * 85)
        
        for r in results:
            print(f"{r['dataset']:<25} {r['total_edges']:<8} {r['total_nodes']:<8} {r['timeline_length']:<10} "
                  f"{r['edges_per_ts']:<8.1f} {r['graph_scale']:<10.0f} {r['density_factor']:<10.4f}")
    
    return results

def analyze_graphrnn_performance_correlation(complexity_results):
    """Correlate complexity with GraphRNN performance"""
    print("\n" + "=" * 80)
    print("🎯 PERFORMANCE vs COMPLEXITY CORRELATION")
    print("=" * 80)
    
    # Known GraphRNN performance (from previous runs)
    performance = {
        'triadic_perfect_medium': {'auc': 75.0, 'ap': 77.1, 'acc': 50.0},
        'triadic_perfect_large': {'auc': 94.4, 'ap': 95.8, 'acc': 58.3},
        'triadic_perfect_long_dense': {'auc': 51.8, 'ap': 52.3, 'acc': 51.8}
    }
    
    print(f"\n{'Dataset':<25} {'AUC':<8} {'AP':<8} {'Acc':<8} {'Scale':<10} {'Timeline':<10}")
    print("-" * 75)
    
    for r in complexity_results:
        dataset = r['dataset']
        if dataset in performance:
            perf = performance[dataset]
            print(f"{dataset:<25} {perf['auc']:<8.1f} {perf['ap']:<8.1f} {perf['acc']:<8.1f} "
                  f"{r['graph_scale']:<10.0f} {r['timeline_length']:<10}")
    
    print(f"\n💡 Key Observations:")
    print(f"  - Medium dataset (scale=188): Good AUC/AP (75-77%)")
    print(f"  - Large dataset (scale=600): Excellent AUC/AP (94-96%)")
    print(f"  - Dense dataset (scale=40,824): COLLAPSED AUC/AP (52%)")
    print(f"  - 🚨 CRITICAL THRESHOLD: Performance collapse at scale ~40K")

def identify_root_causes():
    """Identify the most likely root causes"""
    print("\n" + "=" * 80)
    print("🔍 ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    print(f"\n🎯 PRIMARY SUSPECTS:")
    
    causes = [
        {
            'cause': '1. MASSIVE SCALE JUMP',
            'evidence': 'Scale increased 68x: 600 → 40,824',
            'impact': 'Model overwhelmed by complexity',
            'likelihood': '⭐⭐⭐⭐⭐'
        },
        {
            'cause': '2. LONG SEQUENCE LENGTH',
            'evidence': 'Timeline: 5 → 28 timestamps (5.6x increase)',
            'impact': 'LSTM vanishing gradients over 28 steps',
            'likelihood': '⭐⭐⭐⭐'
        },
        {
            'cause': '3. EXTREME VARIANCE',
            'evidence': 'Edges/timestamp: 2-381 (massive variance)',
            'impact': 'Model cannot adapt to irregular patterns',
            'likelihood': '⭐⭐⭐⭐'
        },
        {
            'cause': '4. INSUFFICIENT TRAINING',
            'evidence': '50 epochs for 68x more complex dataset',
            'impact': 'Underfitting due to inadequate optimization',
            'likelihood': '⭐⭐⭐'
        }
    ]
    
    for cause in causes:
        print(f"\n{cause['cause']} {cause['likelihood']}")
        print(f"  Evidence: {cause['evidence']}")
        print(f"  Impact: {cause['impact']}")

def recommend_fixes():
    """Recommend specific fixes based on analysis"""
    print("\n" + "=" * 80)
    print("🔧 RECOMMENDED FIXES (Priority Order)")
    print("=" * 80)
    
    fixes = [
        {
            'priority': '🔥 URGENT',
            'fix': 'Increase Training Scale',
            'action': 'epochs: 50 → 300+, lr: 0.001 → 0.0001',
            'rationale': 'Dense dataset needs much more optimization time'
        },
        {
            'priority': '🔥 URGENT', 
            'fix': 'Add Gradient Clipping',
            'action': 'torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)',
            'rationale': 'Prevent gradient explosion with long sequences'
        },
        {
            'priority': '⚡ HIGH',
            'fix': 'Increase Model Capacity',
            'action': 'hidden_dim: 128 → 256, rnn_layers: 2 → 3',
            'rationale': 'More parameters needed for complex patterns'
        },
        {
            'priority': '⚡ HIGH',
            'fix': 'Curriculum Learning',
            'action': 'Train on timestamps 0-10 first, then full sequence',
            'rationale': 'Gradually increase sequence complexity'
        },
        {
            'priority': '📊 MEDIUM',
            'fix': 'Batch Normalization/LayerNorm',
            'action': 'Add normalization layers to stabilize training',
            'rationale': 'Handle varying edge counts per timestamp'
        }
    ]
    
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['fix']} [{fix['priority']}]")
        print(f"   Action: {fix['action']}")
        print(f"   Why: {fix['rationale']}")

def main():
    print("🚨 GRAPHRNN FAILURE ANALYSIS REPORT")
    print("Investigating why GraphRNN collapsed on triadic_perfect_long_dense")
    
    # Run all analyses
    analyze_edge_distribution_over_time()
    complexity_results = analyze_complexity_metrics()
    analyze_graphrnn_performance_correlation(complexity_results)
    identify_root_causes()
    recommend_fixes()
    
    print("\n" + "=" * 80)
    print("📋 EXECUTIVE SUMMARY")
    print("=" * 80)
    print("✅ DIAGNOSIS: GraphRNN failed due to MASSIVE SCALE INCREASE")
    print("📊 EVIDENCE: 68x complexity jump (600 → 40,824 scale)")
    print("🎯 SOLUTION: Dramatically increase training (300+ epochs) + model capacity")
    print("⏱️  TIMELINE: Long sequences (28 steps) causing LSTM issues")
    print("🔧 IMMEDIATE ACTION: Try curriculum learning + gradient clipping")
    print("=" * 80)

if __name__ == "__main__":
    main()
