"""
Debug why the simple fix is giving 0% accuracy
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def debug_simple_fix():
    """Debug the simple confidence fix"""
    
    print("🔍" + "="*60 + "🔍")
    print("DEBUGGING SIMPLE FIX: 0% ACCURACY ISSUE")
    print("🔍" + "="*60 + "🔍")
    
    # Load data
    g_df = pd.read_csv('./processed/triadic_perfect_long_dense/ml_triadic_perfect_long_dense.csv')
    
    # Test one specific timestamp
    test_ts = 22
    
    print(f"\n🧪 TESTING TIMESTAMP {test_ts}:")
    
    # Get edges before timestamp
    edges_before = g_df[g_df['ts'] < test_ts]
    edges_up_to_t = [(int(row.u), int(row.i)) for _, row in edges_before.iterrows()]
    
    print(f"  Edges before ts {test_ts}: {len(edges_before)}")
    
    # Get actual edges at timestamp
    edges_at_ts = g_df[g_df['ts'] == test_ts]
    actual_edges = set()
    for _, row in edges_at_ts.iterrows():
        actual_edges.add((int(row.u), int(row.i)))
        actual_edges.add((int(row.i), int(row.u)))  # Both directions
    
    print(f"  Actual edges at ts {test_ts}: {len(actual_edges)//2}")
    
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges_up_to_t:
        adj[u].add(v)
        adj[v].add(u)
    
    # Get all nodes
    all_nodes = sorted(set(g_df['u'].tolist() + g_df['i'].tolist()))
    
    # Make predictions with FIXED formula
    predictions = {}
    
    for u in all_nodes:
        for v in all_nodes:
            if u >= v:  # Skip duplicates
                continue
            
            if v in adj[u]:  # Edge already exists
                continue
            
            # Find common neighbors
            common_neighbors = adj[u].intersection(adj[v])
            
            if len(common_neighbors) > 0:
                # FIXED FORMULA
                confidence = min(1.0, len(common_neighbors) / 1.0)
                predictions[(u, v)] = confidence
    
    print(f"  Total predictions: {len(predictions)}")
    
    # Analyze predictions
    if predictions:
        confidences = list(predictions.values())
        print(f"  Confidence stats:")
        print(f"    Mean: {np.mean(confidences):.3f}")
        print(f"    Min: {np.min(confidences):.3f}")
        print(f"    Max: {np.max(confidences):.3f}")
        print(f"    > 0.5: {sum(1 for c in confidences if c > 0.5)}/{len(confidences)} ({100*sum(1 for c in confidences if c > 0.5)/len(confidences):.1f}%)")
        
        # Check accuracy manually
        correct = 0
        for (u, v), confidence in predictions.items():
            if (u, v) in actual_edges or (v, u) in actual_edges:
                correct += 1
        
        accuracy = correct / len(predictions) if predictions else 0
        print(f"  Manual accuracy check: {correct}/{len(predictions)} = {accuracy:.3f}")
        
        # Show a few examples
        print(f"\n  Example predictions:")
        count = 0
        for (u, v), confidence in list(predictions.items())[:5]:
            is_correct = (u, v) in actual_edges or (v, u) in actual_edges
            print(f"    ({u}, {v}): confidence={confidence:.3f}, correct={is_correct}")
            count += 1
    
    # Debug the evaluation logic
    print(f"\n🔍 DEBUGGING EVALUATION LOGIC:")
    
    # Separate positive and negative predictions
    pos_predictions = []
    neg_predictions = []
    
    for (u, v), confidence in predictions.items():
        if (u, v) in actual_edges or (v, u) in actual_edges:
            pos_predictions.append(confidence)
        else:
            neg_predictions.append(confidence)
    
    print(f"  Positive predictions: {len(pos_predictions)}")
    print(f"  Negative predictions: {len(neg_predictions)}")
    
    if pos_predictions:
        print(f"  Positive confidences: {pos_predictions[:5]}")
    if neg_predictions:
        print(f"  Negative confidences: {neg_predictions[:5]}")
    
    # Check threshold behavior
    if pos_predictions and neg_predictions:
        pos_above_05 = sum(1 for c in pos_predictions if c > 0.5)
        neg_above_05 = sum(1 for c in neg_predictions if c > 0.5)
        
        print(f"  Positives > 0.5: {pos_above_05}/{len(pos_predictions)}")
        print(f"  Negatives > 0.5: {neg_above_05}/{len(neg_predictions)}")
        
        # Expected accuracy with 0.5 threshold
        total_correct = pos_above_05 + (len(neg_predictions) - neg_above_05)
        total_samples = len(pos_predictions) + len(neg_predictions)
        expected_acc = total_correct / total_samples if total_samples > 0 else 0
        
        print(f"  Expected accuracy with 0.5 threshold: {expected_acc:.3f}")

def test_original_vs_fixed_formula():
    """Compare original vs fixed confidence formulas"""
    
    print(f"\n📊 FORMULA COMPARISON:")
    
    common_neighbor_counts = [1, 2, 3, 4, 5, 10, 15]
    
    print(f"  {'Common Neighbors':<15} {'Original (/10)':<15} {'Fixed (/1)':<15}")
    print(f"  {'-'*45}")
    
    for count in common_neighbor_counts:
        original = min(1.0, count / 10.0)
        fixed = min(1.0, count / 1.0)
        print(f"  {count:<15} {original:<15.3f} {fixed:<15.3f}")

if __name__ == "__main__":
    debug_simple_fix()
    test_original_vs_fixed_formula()
