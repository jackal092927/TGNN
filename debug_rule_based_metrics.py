"""
Debug why rule-based baseline doesn't get perfect AUC/validation accuracy
even on perfect triadic datasets
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


def debug_rule_based_evaluation(data_name='triadic_perfect_medium'):
    """
    Debug the rule-based evaluation to understand why AUC isn't perfect
    """
    print(f"=== DEBUGGING RULE-BASED EVALUATION ({data_name}) ===\n")
    
    # Load data
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    print(f"Dataset: {len(g_df)} edges, Ground truth: {len(ground_truth)} triadic closures")
    
    # Get timestamps
    max_timestamp = int(g_df.ts.max())
    train_end = int(max_timestamp * 0.6)
    val_end = int(max_timestamp * 0.8)
    
    val_timestamps = list(range(train_end + 1, val_end + 1))
    test_timestamps = list(range(val_end + 1, max_timestamp + 1))
    
    print(f"Max timestamp: {max_timestamp}")
    print(f"Val timestamps: {val_timestamps}")
    print(f"Test timestamps: {test_timestamps}")
    
    # Analyze each timestamp in detail
    for eval_set, timestamps in [("VALIDATION", val_timestamps), ("TEST", test_timestamps)]:
        print(f"\n=== {eval_set} ANALYSIS ===")
        
        for ts in timestamps:
            print(f"\n--- Timestamp {ts} ---")
            
            # Get edges before this timestamp
            before_edges = g_df[g_df.ts < ts]
            at_timestamp = g_df[g_df.ts == ts]
            
            print(f"Edges before ts {ts}: {len(before_edges)}")
            print(f"New edges at ts {ts}: {len(at_timestamp)}")
            
            if len(at_timestamp) == 0:
                print("No new edges - skipping")
                continue
            
            # Build adjacency list
            adj = defaultdict(set)
            existing_edges = set()
            
            for _, row in before_edges.iterrows():
                adj[row.u].add(row.i)
                adj[row.i].add(row.u)
                existing_edges.add((min(row.u, row.i), max(row.u, row.i)))
            
            # Get actual new edges
            actual_new_edges = set()
            actual_triadic_closures = set()
            
            for _, row in at_timestamp.iterrows():
                edge_key = (min(row.u, row.i), max(row.u, row.i))
                actual_new_edges.add(edge_key)
                
                if str(row.idx) in ground_truth:
                    actual_triadic_closures.add(edge_key)
            
            print(f"Actual new edges: {actual_new_edges}")
            print(f"Actual triadic closures: {actual_triadic_closures}")
            
            # Apply rule-based prediction
            all_nodes = set()
            for _, row in g_df.iterrows():
                all_nodes.add(row.u)
                all_nodes.add(row.i)
            
            candidates = {}
            for shared_node in all_nodes:
                neighbors = list(adj[shared_node])
                
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        node_a, node_b = neighbors[i], neighbors[j]
                        edge_key = (min(node_a, node_b), max(node_a, node_b))
                        
                        if edge_key not in existing_edges:
                            shared_neighbors = adj[node_a] & adj[node_b]
                            confidence = len(shared_neighbors) / max(1, len(adj[node_a]) + len(adj[node_b]) - len(shared_neighbors))
                            
                            if edge_key not in candidates or confidence > candidates[edge_key][1]:
                                candidates[edge_key] = (shared_node, confidence)
            
            print(f"Rule-based candidates: {len(candidates)}")
            for edge_key, (shared_neighbor, confidence) in candidates.items():
                is_actual = edge_key in actual_new_edges
                print(f"  {edge_key}: confidence={confidence:.4f} {'✓' if is_actual else '✗'}")
            
            # Evaluate metrics
            if len(candidates) == 0:
                print("No candidates - metrics undefined")
                continue
            
            # Separate positive and negative candidates
            positive_candidates = []
            negative_candidates = []
            
            for edge_key, (shared_neighbor, confidence) in candidates.items():
                if edge_key in actual_new_edges:
                    positive_candidates.append((edge_key, confidence))
                else:
                    negative_candidates.append((edge_key, confidence))
            
            print(f"Positive candidates: {len(positive_candidates)}")
            print(f"Negative candidates: {len(negative_candidates)}")
            
            # Balance sampling (1:1 ratio)
            num_positives = len(positive_candidates)
            if num_positives == 0:
                print("No positive samples - accuracy undefined")
                continue
            
            num_negatives_to_sample = num_positives
            if num_negatives_to_sample > len(negative_candidates):
                sampled_negatives = negative_candidates
            else:
                # Sample negatives with highest confidence (hardest cases)
                negative_candidates.sort(key=lambda x: x[1], reverse=True)
                sampled_negatives = negative_candidates[:num_negatives_to_sample]
            
            # Combine samples
            eval_samples = positive_candidates + sampled_negatives
            eval_labels = [1] * len(positive_candidates) + [0] * len(sampled_negatives)
            eval_scores = [confidence for _, confidence in eval_samples]
            
            print(f"Evaluation samples: {len(eval_samples)} ({len(positive_candidates)} pos + {len(sampled_negatives)} neg)")
            print(f"Labels: {eval_labels}")
            print(f"Scores: {[f'{s:.4f}' for s in eval_scores]}")
            
            if len(eval_samples) == 0:
                print("No evaluation samples")
                continue
            
            # Calculate metrics
            eval_labels = np.array(eval_labels)
            eval_scores = np.array(eval_scores)
            
            # Accuracy (threshold = 0.5)
            pred_binary = (eval_scores > 0.5).astype(int)
            accuracy = accuracy_score(eval_labels, pred_binary)
            
            print(f"Predictions (>0.5): {pred_binary}")
            print(f"Accuracy: {accuracy:.4f}")
            
            # AUC and AP
            if len(np.unique(eval_labels)) > 1:
                auc = roc_auc_score(eval_labels, eval_scores)
                ap = average_precision_score(eval_labels, eval_scores)
                print(f"AUC: {auc:.4f}")
                print(f"AP: {ap:.4f}")
                
                # Explain AUC
                print(f"\nAUC ANALYSIS:")
                print(f"AUC measures ranking quality - can all positives be ranked above negatives?")
                
                pos_scores = [s for i, s in enumerate(eval_scores) if eval_labels[i] == 1]
                neg_scores = [s for i, s in enumerate(eval_scores) if eval_labels[i] == 0]
                
                print(f"Positive scores: {[f'{s:.4f}' for s in pos_scores]}")
                print(f"Negative scores: {[f'{s:.4f}' for s in neg_scores]}")
                
                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    min_pos = min(pos_scores)
                    max_neg = max(neg_scores)
                    print(f"Min positive score: {min_pos:.4f}")
                    print(f"Max negative score: {max_neg:.4f}")
                    
                    if min_pos <= max_neg:
                        print(f"❌ AUC < 1.0 because some negatives score higher than some positives!")
                        print(f"❌ This happens when confidence scores overlap between positive and negative cases")
                    else:
                        print(f"✅ AUC should be 1.0 because all positives score higher than all negatives")
            else:
                auc = 0.5
                ap = np.mean(eval_labels)
                print(f"AUC: {auc:.4f} (only one class)")
                print(f"AP: {ap:.4f} (only one class)")


if __name__ == "__main__":
    debug_rule_based_evaluation('triadic_perfect_medium')
    print("\n" + "="*80)
    debug_rule_based_evaluation('triadic_perfect_large')
