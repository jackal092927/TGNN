"""
Debug why rule-based method only gets 60% accuracy on perfect triadic dataset
Expected: ~100% accuracy since dataset is deterministic
Actual: 60% accuracy - something is wrong!
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict

def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    return g_df, ground_truth

def find_triadic_closures_rule_based(edges_up_to_t, all_nodes):
    """
    Find all possible triadic closures at timestamp t
    Returns dict of {(u,v): confidence_score}
    """
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges_up_to_t:
        adj[u].add(v)
        adj[v].add(u)
    
    triadic_predictions = {}
    
    # For each pair of nodes
    for u in all_nodes:
        for v in all_nodes:
            if u >= v:  # Avoid duplicates and self-loops
                continue
            
            if v in adj[u]:  # Edge already exists
                continue
            
            # Find common neighbors
            common_neighbors = adj[u].intersection(adj[v])
            
            if len(common_neighbors) > 0:
                # Confidence based on number of common neighbors
                # More common neighbors = higher confidence
                confidence = min(1.0, len(common_neighbors) / 10.0)  # Cap at 1.0
                triadic_predictions[(u, v)] = confidence
    
    return triadic_predictions

def analyze_rule_based_accuracy_issue():
    """Analyze why rule-based accuracy is only 60%"""
    
    print("🔍" + "="*70 + "🔍")
    print("DEBUGGING RULE-BASED ACCURACY ISSUE")
    print("Expected: ~100% accuracy (deterministic dataset)")
    print("Actual: 60% accuracy - WHY?")
    print("🔍" + "="*70 + "🔍")
    
    # Load data
    g_df, ground_truth = load_triadic_data('triadic_perfect_long_dense')
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total edges: {len(g_df)}")
    print(f"  Timestamps: {g_df['ts'].min()} to {g_df['ts'].max()}")
    print(f"  Ground truth entries: {len(ground_truth)}")
    
    # Get all nodes
    all_nodes = sorted(set(g_df['u'].tolist() + g_df['i'].tolist()))
    print(f"  Total nodes: {len(all_nodes)}")
    
    # Analyze a few specific timestamps
    test_timestamps = [22, 23, 24, 25, 26, 27]  # Test set timestamps
    
    total_correct = 0
    total_predictions = 0
    total_gt_edges = 0
    
    for ts in test_timestamps:
        print(f"\n🔍 Analyzing Timestamp {ts}:")
        
        # Get edges up to (but not including) timestamp ts
        edges_before = g_df[g_df['ts'] < ts]
        edges_up_to_t = [(int(row.u), int(row.i)) for _, row in edges_before.iterrows()]
        
        # Get actual edges at timestamp ts (ground truth)
        edges_at_ts = g_df[g_df['ts'] == ts]
        gt_edges_at_ts = set()
        for _, row in edges_at_ts.iterrows():
            gt_edges_at_ts.add((int(row.u), int(row.i)))
            gt_edges_at_ts.add((int(row.i), int(row.u)))  # Both directions
        
        print(f"  Edges before ts {ts}: {len(edges_up_to_t)}")
        print(f"  Ground truth edges at ts {ts}: {len(gt_edges_at_ts)//2}")  # Divide by 2 (undirected)
        
        # Rule-based predictions
        rule_predictions = find_triadic_closures_rule_based(edges_up_to_t, all_nodes)
        print(f"  Rule-based predictions: {len(rule_predictions)}")
        
        # Analyze predictions
        correct_predictions = 0
        false_positives = 0
        
        for (u, v), confidence in rule_predictions.items():
            if (u, v) in gt_edges_at_ts or (v, u) in gt_edges_at_ts:
                correct_predictions += 1
            else:
                false_positives += 1
                if false_positives <= 3:  # Show first few false positives
                    print(f"    FALSE POSITIVE: ({u}, {v}) with confidence {confidence:.3f}")
        
        # Check for missed edges (false negatives)
        rule_edge_set = set()
        for (u, v) in rule_predictions.keys():
            rule_edge_set.add((u, v))
            rule_edge_set.add((v, u))
        
        missed_edges = 0
        for edge in gt_edges_at_ts:
            if edge not in rule_edge_set:
                missed_edges += 1
                if missed_edges <= 3:  # Show first few missed edges
                    print(f"    MISSED EDGE: {edge}")
        
        accuracy = correct_predictions / len(rule_predictions) if len(rule_predictions) > 0 else 0
        recall = correct_predictions / (len(gt_edges_at_ts)//2) if len(gt_edges_at_ts) > 0 else 0
        
        print(f"  Correct: {correct_predictions}, False Pos: {false_positives}, Missed: {missed_edges//2}")
        print(f"  Accuracy: {accuracy:.3f}, Recall: {recall:.3f}")
        
        total_correct += correct_predictions
        total_predictions += len(rule_predictions)
        total_gt_edges += len(gt_edges_at_ts)//2
    
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    overall_recall = total_correct / total_gt_edges if total_gt_edges > 0 else 0
    
    print(f"\n📊 OVERALL ANALYSIS:")
    print(f"  Total rule predictions: {total_predictions}")
    print(f"  Total correct: {total_correct}")
    print(f"  Total GT edges: {total_gt_edges}")
    print(f"  Overall Accuracy: {overall_accuracy:.3f}")
    print(f"  Overall Recall: {overall_recall:.3f}")
    
    print(f"\n🎯 HYPOTHESIS FOR LOW ACCURACY:")
    
    if overall_accuracy < 0.8:
        print(f"  1. 🤔 OVER-PREDICTION: Rule-based method predicts too many edges")
        print(f"     - Rule finds ALL possible triadic closures")
        print(f"     - But dataset may only realize SOME of them")
        print(f"     - Even 'perfect' dataset might have selection criteria")
        
        print(f"\n  2. 🎯 CONFIDENCE THRESHOLD ISSUE:")
        print(f"     - Current evaluation uses 0.5 threshold")
        print(f"     - But rule-based confidences might be systematically low")
        print(f"     - Need to check confidence distribution")
        
        print(f"\n  3. 📊 EVALUATION METHODOLOGY:")
        print(f"     - Balanced sampling creates 50-50 pos-neg split")
        print(f"     - If rule over-predicts, accuracy will be ~50%")
        print(f"     - Perfect AUC/AP but poor accuracy")
    
    else:
        print(f"  ✅ Rule-based method is actually working well!")
        print(f"  The 60% accuracy from previous run might be due to:")
        print(f"     - Random negative sampling variation")
        print(f"     - Confidence threshold issues")

def analyze_confidence_distribution():
    """Analyze the distribution of rule-based confidence scores"""
    
    print(f"\n🎯 CONFIDENCE SCORE ANALYSIS:")
    
    g_df, _ = load_triadic_data('triadic_perfect_long_dense')
    all_nodes = sorted(set(g_df['u'].tolist() + g_df['i'].tolist()))
    
    # Analyze timestamp 23 (has many edges)
    ts = 23
    edges_before = g_df[g_df['ts'] < ts]
    edges_up_to_t = [(int(row.u), int(row.i)) for _, row in edges_before.iterrows()]
    
    edges_at_ts = g_df[g_df['ts'] == ts]
    gt_edges_at_ts = set()
    for _, row in edges_at_ts.iterrows():
        gt_edges_at_ts.add((int(row.u), int(row.i)))
        gt_edges_at_ts.add((int(row.i), int(row.u)))
    
    rule_predictions = find_triadic_closures_rule_based(edges_up_to_t, all_nodes)
    
    # Separate positive and negative predictions
    pos_confidences = []
    neg_confidences = []
    
    for (u, v), confidence in rule_predictions.items():
        if (u, v) in gt_edges_at_ts or (v, u) in gt_edges_at_ts:
            pos_confidences.append(confidence)
        else:
            neg_confidences.append(confidence)
    
    print(f"  Positive predictions: {len(pos_confidences)}")
    print(f"  Negative predictions: {len(neg_confidences)}")
    
    if pos_confidences:
        print(f"  Positive confidence: μ={np.mean(pos_confidences):.3f}, σ={np.std(pos_confidences):.3f}")
        print(f"                      min={np.min(pos_confidences):.3f}, max={np.max(pos_confidences):.3f}")
    
    if neg_confidences:
        print(f"  Negative confidence: μ={np.mean(neg_confidences):.3f}, σ={np.std(neg_confidences):.3f}")
        print(f"                      min={np.min(neg_confidences):.3f}, max={np.max(neg_confidences):.3f}")
    
    # Check what happens with different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n  Accuracy at different thresholds:")
    
    for thresh in thresholds:
        correct = 0
        total = 0
        
        for (u, v), confidence in rule_predictions.items():
            prediction = 1 if confidence > thresh else 0
            actual = 1 if (u, v) in gt_edges_at_ts or (v, u) in gt_edges_at_ts else 0
            
            if prediction == actual:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"    Threshold {thresh:.1f}: {accuracy:.3f}")

if __name__ == "__main__":
    analyze_rule_based_accuracy_issue()
    analyze_confidence_distribution()
    
    print(f"\n" + "🎯" + "="*70 + "🎯")
    print("CONCLUSION: Investigating rule-based accuracy discrepancy")
    print("Expected ~100% for deterministic dataset, got 60%")
    print("🎯" + "="*70 + "🎯")
