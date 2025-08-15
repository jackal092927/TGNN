"""
Fixed Rule-based Baseline with Proper Negative Sampling

This version samples random non-existing edges as negatives for balanced evaluation,
rather than only considering failed triadic closure predictions.
"""

import pandas as pd
import numpy as np
import json
import time
import logging
import random
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


def setup_logging(data_name):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'log/rule_based_fixed_{data_name}_{int(time.time())}.log'),
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
    
    return candidates, existing_edges, adj


def sample_random_negative_edges(all_nodes, existing_edges, actual_new_edges, num_samples):
    """
    Sample random non-existing edges as negative samples
    """
    all_nodes_list = list(all_nodes)
    negative_samples = []
    attempts = 0
    max_attempts = num_samples * 10  # Avoid infinite loop
    
    while len(negative_samples) < num_samples and attempts < max_attempts:
        # Sample two random nodes
        u, v = random.sample(all_nodes_list, 2)
        edge_key = (min(u, v), max(u, v))
        
        # Check if this edge doesn't exist and isn't an actual new edge
        if edge_key not in existing_edges and edge_key not in actual_new_edges:
            negative_samples.append(edge_key)
        
        attempts += 1
    
    return negative_samples


def evaluate_rule_based_predictions_fixed(g_df, timestamp, all_nodes, logger, balance_ratio=1.0):
    """
    Evaluate rule-based predictions with proper negative sampling
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
    candidates, existing_edges, adj = find_triadic_closures_rule_based(current_edges, all_nodes)
    
    # Separate positive and negative candidates from triadic predictions
    positive_candidates = []
    triadic_negative_candidates = []
    
    for edge_key, (shared_neighbor, confidence) in candidates.items():
        if edge_key in actual_new_edges:
            positive_candidates.append((edge_key, confidence))
        else:
            triadic_negative_candidates.append((edge_key, confidence))
    
    num_positives = len(positive_candidates)
    if num_positives == 0:
        return 0.5, 0.5, 0.5  # No positive samples
    
    logger.info(f"Timestamp {timestamp}: {num_positives} positives from triadic predictions")
    logger.info(f"Timestamp {timestamp}: {len(triadic_negative_candidates)} negatives from triadic predictions")
    
    # Sample additional random negative edges if needed
    num_negatives_needed = int(num_positives * balance_ratio)
    
    if len(triadic_negative_candidates) >= num_negatives_needed:
        # Use triadic negatives (prefer higher confidence for harder evaluation)
        triadic_negative_candidates.sort(key=lambda x: x[1], reverse=True)
        sampled_negatives = triadic_negative_candidates[:num_negatives_needed]
        negative_scores = [conf for _, conf in sampled_negatives]
    else:
        # Need to sample additional random negatives
        additional_needed = num_negatives_needed - len(triadic_negative_candidates)
        random_negative_edges = sample_random_negative_edges(
            all_nodes, existing_edges, actual_new_edges, additional_needed
        )
        
        logger.info(f"Timestamp {timestamp}: Sampling {additional_needed} additional random negatives")
        
        # Assign confidence 0.0 to random negatives (they have no triadic support)
        random_negatives = [(edge, 0.0) for edge in random_negative_edges]
        
        # Combine triadic negatives and random negatives
        all_negatives = triadic_negative_candidates + random_negatives
        sampled_negatives = all_negatives[:num_negatives_needed]
        negative_scores = [conf for _, conf in sampled_negatives]
    
    # Combine positive and negative samples
    positive_scores = [conf for _, conf in positive_candidates]
    
    eval_labels = [1] * len(positive_candidates) + [0] * len(sampled_negatives)
    eval_scores = positive_scores + negative_scores
    
    if len(eval_scores) == 0:
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
    
    logger.info(f"Timestamp {timestamp}: Pos scores: {[f'{s:.3f}' for s in positive_scores[:5]]}...")
    logger.info(f"Timestamp {timestamp}: Neg scores: {[f'{s:.3f}' for s in negative_scores[:5]]}...")
    
    return accuracy, auc, ap


def evaluate_rule_based_baseline_fixed(data_name='triadic_perfect_medium'):
    """
    Evaluate rule-based baseline with proper negative sampling
    """
    logger = setup_logging(data_name)
    logger.info(f"Evaluating Fixed Rule-based Baseline on {data_name}")
    
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
        acc, auc, ap = evaluate_rule_based_predictions_fixed(g_df, val_ts, all_nodes, logger, balance_ratio=1.0)
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
        acc, auc, ap = evaluate_rule_based_predictions_fixed(g_df, test_ts, all_nodes, logger, balance_ratio=1.0)
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
        
        return {
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'test_ap': test_ap,
            'val_accuracy': val_acc if len(val_aps) > 0 else 0.0,
            'val_auc': val_auc if len(val_aps) > 0 else 0.0,
            'val_ap': val_ap if len(val_aps) > 0 else 0.0,
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
        }


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Fixed Rule-based Triadic Closure Baseline')
    parser.add_argument('--data', type=str, default='triadic_perfect_medium')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('log', exist_ok=True)
    
    print(f"Evaluating Fixed Rule-based Baseline on {args.data}")
    print(f"Key improvement: Proper negative sampling with random non-existing edges")
    
    # Evaluate baseline
    results = evaluate_rule_based_baseline_fixed(data_name=args.data)
    
    print("\n" + "="*60)
    print("FINAL RESULTS (Fixed Rule-based Baseline)")
    print("="*60)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test AUC:      {results['test_auc']:.4f}")
    print(f"Test AP:       {results['test_ap']:.4f}")
    print(f"Val Accuracy:  {results['val_accuracy']:.4f}")
    print(f"Val AUC:       {results['val_auc']:.4f}")
    print(f"Val AP:        {results['val_ap']:.4f}")
    print("="*60)
