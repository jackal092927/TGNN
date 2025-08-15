"""
Rule-based Baseline for Triadic Closure Prediction

Implements the direct triadic closure rule:
- If edges A-B and B-C exist, predict A-C with high probability
- This establishes the theoretical upper bound for triadic closure prediction
"""

import pandas as pd
import numpy as np
import json
import time
import logging
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


def setup_logging(data_name):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'log/rule_based_{data_name}_{int(time.time())}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    # Load ground truth
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    return g_df, e_feat, n_feat, ground_truth


def get_edges_at_timestamp(g_df, timestamp):
    """Get all edges up to and including given timestamp"""
    edges_df = g_df[g_df.ts <= timestamp]
    
    edge_list = []
    edge_indices = []
    
    for _, row in edges_df.iterrows():
        edge_list.append((row.u, row.i))
        edge_indices.append(row.idx)
    
    return edge_list, edge_indices


def find_triadic_closures_rule_based(edge_list, all_nodes):
    """
    Find triadic closure candidates using the direct rule:
    If edges A-B and B-C exist, then A-C is a triadic closure candidate
    
    Args:
        edge_list: list of current edges [(u,v), ...]
        all_nodes: set of all nodes
    
    Returns:
        candidates: dict {(src, dst): (shared_neighbor, confidence)}
    """
    # Build adjacency list
    adj = defaultdict(set)
    existing_edges = set()
    
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
        existing_edges.add((min(u, v), max(u, v)))
    
    candidates = {}
    
    # Apply triadic closure rule
    for shared_node in all_nodes:
        neighbors = list(adj[shared_node])
        
        # For each pair of neighbors of shared_node
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                node_a, node_b = neighbors[i], neighbors[j]
                edge_key = (min(node_a, node_b), max(node_a, node_b))
                
                # If A-B edge doesn't exist, it's a triadic closure candidate
                if edge_key not in existing_edges:
                    # Calculate confidence based on number of shared neighbors
                    shared_neighbors = adj[node_a] & adj[node_b]
                    confidence = len(shared_neighbors) / max(1, len(adj[node_a]) + len(adj[node_b]) - len(shared_neighbors))
                    
                    # Store the best (highest confidence) prediction for this edge
                    if edge_key not in candidates or confidence > candidates[edge_key][1]:
                        candidates[edge_key] = (shared_node, confidence)
    
    return candidates


def evaluate_rule_based_predictions(g_df, timestamp, all_nodes, logger, balance_ratio=1.0):
    """
    Evaluate rule-based predictions for a specific timestamp
    """
    # Get current edges (up to timestamp-1)
    current_edges, _ = get_edges_at_timestamp(g_df, timestamp - 1)
    
    if len(current_edges) == 0:
        return 0.5, 0.5, 0.5  # No edges to work with
    
    # Get actual new edges at this timestamp
    new_edges_df = g_df[g_df.ts == timestamp]
    actual_new_edges = set()
    
    for _, row in new_edges_df.iterrows():
        actual_new_edges.add((min(row.u, row.i), max(row.u, row.i)))
    
    if len(actual_new_edges) == 0:
        return 0.5, 0.5, 0.5  # No new edges to predict
    
    # Get rule-based triadic closure candidates
    candidates = find_triadic_closures_rule_based(current_edges, all_nodes)
    
    if len(candidates) == 0:
        return 0.5, 0.5, 0.5  # No candidates
    
    # Separate positive and negative candidates
    positive_candidates = []
    negative_candidates = []
    
    for edge_key, (shared_neighbor, confidence) in candidates.items():
        if edge_key in actual_new_edges:
            positive_candidates.append((edge_key, confidence))
        else:
            negative_candidates.append((edge_key, confidence))
    
    # Balance sampling
    num_positives = len(positive_candidates)
    if num_positives == 0:
        return 0.5, 0.5, 0.5  # No positive samples
    
    num_negatives_to_sample = int(num_positives * balance_ratio)
    if num_negatives_to_sample > len(negative_candidates):
        sampled_negatives = negative_candidates
    else:
        # Sample negatives (prefer higher confidence negatives for harder evaluation)
        negative_candidates.sort(key=lambda x: x[1], reverse=True)
        sampled_negatives = negative_candidates[:num_negatives_to_sample]
    
    # Combine samples
    eval_samples = positive_candidates + sampled_negatives
    eval_labels = [1] * len(positive_candidates) + [0] * len(sampled_negatives)
    eval_scores = [confidence for _, confidence in eval_samples]
    
    if len(eval_samples) == 0:
        return 0.5, 0.5, 0.5
    
    # Calculate metrics
    eval_labels = np.array(eval_labels)
    eval_scores = np.array(eval_scores)
    
    # Accuracy (threshold = 0.5)
    pred_binary = (eval_scores > 0.5).astype(int)
    accuracy = accuracy_score(eval_labels, pred_binary)
    
    # AUC and AP
    if len(np.unique(eval_labels)) > 1:
        auc = roc_auc_score(eval_labels, eval_scores)
        ap = average_precision_score(eval_labels, eval_scores)
    else:
        auc = 0.5
        ap = np.mean(eval_labels)
    
    return accuracy, auc, ap


def evaluate_rule_based_baseline(data_name='triadic_medium'):
    """
    Evaluate rule-based baseline on triadic closure dataset
    """
    logger = setup_logging(data_name)
    logger.info(f"Evaluating Rule-based Baseline on {data_name}")
    
    # Load data
    g_df, e_feat, n_feat, ground_truth = load_triadic_data(data_name)
    logger.info(f"Dataset: {len(g_df)} edges, {len(n_feat)} nodes")
    logger.info(f"Ground truth triadic closures: {len(ground_truth)}")
    
    # Get all nodes and timestamps
    all_nodes = set(range(len(n_feat)))
    max_timestamp = int(g_df.ts.max())
    
    # Split timestamps
    train_end = int(max_timestamp * 0.6)
    val_end = int(max_timestamp * 0.8)
    
    train_timestamps = list(range(1, train_end + 1))
    val_timestamps = list(range(train_end + 1, val_end + 1))
    test_timestamps = list(range(val_end + 1, max_timestamp + 1))
    
    logger.info(f"Train timestamps: {len(train_timestamps)} (ts 1-{train_end})")
    logger.info(f"Val timestamps: {len(val_timestamps)} (ts {train_end+1}-{val_end})")
    logger.info(f"Test timestamps: {len(test_timestamps)} (ts {val_end+1}-{max_timestamp})")
    
    # Validation evaluation
    logger.info("\n=== VALIDATION EVALUATION ===")
    val_accs, val_aucs, val_aps = [], [], []
    
    for val_ts in val_timestamps:
        acc, auc, ap = evaluate_rule_based_predictions(g_df, val_ts, all_nodes, logger, balance_ratio=1.0)
        val_accs.append(acc)
        val_aucs.append(auc)
        val_aps.append(ap)
        
        logger.info(f"Timestamp {val_ts}: Acc={acc:.4f}, AUC={auc:.4f}, AP={ap:.4f}")
    
    if len(val_aps) > 0:
        val_acc = np.mean(val_accs)
        val_auc = np.mean(val_aucs)
        val_ap = np.mean(val_aps)
        
        logger.info(f"Validation Results: Acc={val_acc:.4f}, AUC={val_auc:.4f}, AP={val_ap:.4f}")
    
    # Test evaluation
    logger.info("\n=== TEST EVALUATION ===")
    test_accs, test_aucs, test_aps = [], [], []
    
    for test_ts in test_timestamps:
        acc, auc, ap = evaluate_rule_based_predictions(g_df, test_ts, all_nodes, logger, balance_ratio=1.0)
        test_accs.append(acc)
        test_aucs.append(auc)
        test_aps.append(ap)
        
        logger.info(f"Timestamp {test_ts}: Acc={acc:.4f}, AUC={auc:.4f}, AP={ap:.4f}")
    
    if len(test_aps) > 0:
        test_acc = np.mean(test_accs)
        test_auc = np.mean(test_aucs)
        test_ap = np.mean(test_aps)
        
        logger.info(f"\nFinal Test Results:")
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
        logger.info(f"  Test AUC: {test_auc:.4f}")
        logger.info(f"  Test AP: {test_ap:.4f}")
        
        # Detailed analysis
        logger.info(f"\n=== DETAILED ANALYSIS ===")
        
        # Analyze ground truth coverage
        total_gt_closures = len(ground_truth)
        test_gt_closures = 0
        
        for test_ts in test_timestamps:
            new_edges_df = g_df[g_df.ts == test_ts]
            for _, row in new_edges_df.iterrows():
                if str(row.idx) in ground_truth:
                    test_gt_closures += 1
        
        logger.info(f"Ground truth triadic closures in test set: {test_gt_closures}/{total_gt_closures}")
        
        # Analyze rule-based predictions
        total_candidates = 0
        correct_predictions = 0
        
        for test_ts in test_timestamps:
            current_edges, _ = get_edges_at_timestamp(g_df, test_ts - 1)
            candidates = find_triadic_closures_rule_based(current_edges, all_nodes)
            
            actual_new_edges = set()
            new_edges_df = g_df[g_df.ts == test_ts]
            for _, row in new_edges_df.iterrows():
                actual_new_edges.add((min(row.u, row.i), max(row.u, row.i)))
            
            total_candidates += len(candidates)
            
            for edge_key in candidates:
                if edge_key in actual_new_edges:
                    correct_predictions += 1
        
        logger.info(f"Rule-based candidates: {total_candidates}")
        logger.info(f"Correct predictions: {correct_predictions}")
        if total_candidates > 0:
            precision = correct_predictions / total_candidates
            logger.info(f"Rule precision: {precision:.4f}")
        
        return {
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'test_ap': test_ap,
            'val_accuracy': val_acc if len(val_aps) > 0 else 0.0,
            'val_auc': val_auc if len(val_aps) > 0 else 0.0,
            'val_ap': val_ap if len(val_aps) > 0 else 0.0,
            'total_candidates': total_candidates,
            'correct_predictions': correct_predictions
        }
    else:
        logger.warning("No test results available")
        return {
            'test_accuracy': 0.0,
            'test_auc': 0.0,
            'test_ap': 0.0,
            'val_accuracy': 0.0,
            'val_auc': 0.0,
            'val_ap': 0.0,
            'total_candidates': 0,
            'correct_predictions': 0
        }


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Rule-based Triadic Closure Baseline')
    parser.add_argument('--data', type=str, default='triadic_medium')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('log', exist_ok=True)
    
    print(f"Evaluating Rule-based Baseline on {args.data}")
    print(f"Rule: If edges A-B and B-C exist → predict A-C")
    print(f"This establishes the theoretical upper bound for the task")
    
    # Evaluate baseline
    results = evaluate_rule_based_baseline(data_name=args.data)
    
    print("\n" + "="*60)
    print("FINAL RESULTS (Rule-based Baseline)")
    print("="*60)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test AUC:      {results['test_auc']:.4f}")
    print(f"Test AP:       {results['test_ap']:.4f}")
    print(f"Val Accuracy:  {results['val_accuracy']:.4f}")
    print(f"Val AUC:       {results['val_auc']:.4f}")
    print(f"Val AP:        {results['val_ap']:.4f}")
    print(f"Rule Precision: {results['correct_predictions']}/{results['total_candidates']}")
    print("="*60)
