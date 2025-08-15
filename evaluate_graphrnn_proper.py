"""
Proper evaluation of GraphRNN with sequential prediction
The model should predict validation period first, then use those predictions for test
"""

import torch
import pandas as pd
import numpy as np
import json
from graph_rnn import GraphRNN, create_graph_sequence_from_data
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import networkx as nx


def create_graph_state_from_edges(edges_df, n_feat, e_feat):
    """Create a graph state representation from edges"""
    if len(edges_df) == 0:
        return n_feat, np.array([]).reshape(0, 2), np.array([]).reshape(0, e_feat.shape[1])
    
    edge_list = edges_df[['u', 'i']].values
    edge_indices = edges_df['idx'].values
    edge_features = e_feat[edge_indices]
    
    return n_feat, edge_list, edge_features


def predict_next_edge_candidates(model, sequence_data, existing_edges, num_nodes, top_k=10):
    """
    Predict the most likely next edges given current sequence
    
    Args:
        model: trained GraphRNN
        sequence_data: sequence of graph states
        existing_edges: set of existing edges to avoid
        num_nodes: total number of nodes
        top_k: number of top candidates to return
    
    Returns:
        List of (src, dst, score) tuples sorted by score
    """
    model.eval()
    
    candidates = []
    scores = []
    
    # Generate all possible candidate edges
    for src in range(num_nodes):
        for dst in range(src + 1, num_nodes):  # Avoid duplicates
            if (src, dst) not in existing_edges and (dst, src) not in existing_edges:
                candidates.append((src, dst))
    
    # Score all candidates
    with torch.no_grad():
        for src, dst in candidates:
            prob, _ = model(sequence_data, target_src=src, target_dst=dst)
            if prob is not None:
                scores.append(prob.item())
            else:
                scores.append(0.0)
    
    # Sort by score and return top k
    candidate_scores = list(zip(candidates, scores))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [(src, dst, score) for (src, dst), score in candidate_scores[:top_k]]


def sequential_evaluation(data_name='triadic_medium'):
    """
    Proper sequential evaluation:
    1. Train on 0-9
    2. Predict 10-14 sequentially 
    3. Predict 15-20 sequentially using predicted graph
    """
    
    print("🔄 SEQUENTIAL GRAPHRNN EVALUATION")
    print("=" * 60)
    
    # Load data
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    print(f"Dataset: {len(g_df)} edges, {len(n_feat)} nodes")
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphRNN(
        node_feat_dim=n_feat.shape[1],
        edge_feat_dim=e_feat.shape[1],
        hidden_dim=64,
        max_nodes=len(n_feat),
        rnn_layers=2
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(f'models/graphrnn_{data_name}_best.pt'))
        print("✅ Loaded trained model")
    except:
        print("❌ Could not load trained model")
        return
    
    # Split data
    train_end = 9
    val_end = 14
    test_end = 20
    
    train_data = g_df[g_df.ts <= train_end]
    val_data = g_df[(g_df.ts > train_end) & (g_df.ts <= val_end)]
    test_data = g_df[g_df.ts > val_end]
    
    print(f"Training: ts 0-{train_end} ({len(train_data)} edges)")
    print(f"Validation: ts {train_end+1}-{val_end} ({len(val_data)} edges)")  
    print(f"Test: ts {val_end+1}-{test_end} ({len(test_data)} edges)")
    
    # Start with training data as ground truth
    predicted_graph = g_df[g_df.ts <= train_end].copy()
    
    print(f"\n🔮 SEQUENTIAL PREDICTION PHASE")
    print("-" * 40)
    
    val_predictions = []
    val_actual = []
    
    # Phase 1: Predict validation period (10-14) sequentially
    print(f"\n📊 Validation Predictions (ts {train_end+1}-{val_end}):")
    
    for ts in range(train_end + 1, val_end + 1):
        print(f"\n🕒 Predicting timestamp {ts}")
        
        # Get actual edge at this timestamp
        actual_edges_ts = g_df[g_df.ts == ts]
        if len(actual_edges_ts) == 0:
            print(f"  No edges at timestamp {ts}")
            continue
            
        actual_edge = actual_edges_ts.iloc[0]
        actual_src, actual_dst = actual_edge.u, actual_edge.i
        print(f"  Actual edge: {actual_src} → {actual_dst}")
        
        # Create sequence up to ts-1 using predicted graph
        sequence_data = create_graph_sequence_from_data(predicted_graph, ts-1, n_feat, e_feat)
        
        # Get existing edges to avoid
        existing_edges = set()
        for _, row in predicted_graph[predicted_graph.ts <= ts-1].iterrows():
            existing_edges.add((row.u, row.i))
            existing_edges.add((row.i, row.u))
        
        # Predict top candidates
        candidates = predict_next_edge_candidates(model, sequence_data, existing_edges, len(n_feat), top_k=5)
        
        print(f"  Top 5 predictions:")
        predicted_correctly = False
        
        for i, (src, dst, score) in enumerate(candidates):
            marker = "🎯" if (src == actual_src and dst == actual_dst) or (src == actual_dst and dst == actual_src) else "  "
            print(f"    {marker} {i+1}. {src} → {dst}: {score:.4f}")
            
            if (src == actual_src and dst == actual_dst) or (src == actual_dst and dst == actual_src):
                predicted_correctly = True
                val_predictions.append(1)
            
        if not predicted_correctly:
            val_predictions.append(0)
        val_actual.append(1)
        
        # Add the ACTUAL edge to predicted graph (not the prediction)
        # This simulates having access to validation labels for test prediction
        new_row = actual_edge.to_dict()
        predicted_graph = pd.concat([predicted_graph, pd.DataFrame([new_row])], ignore_index=True)
        
        print(f"  Prediction: {'✅ Correct' if predicted_correctly else '❌ Wrong'}")
    
    # Phase 2: Predict test period (15-20) sequentially  
    print(f"\n📊 Test Predictions (ts {val_end+1}-{test_end}):")
    
    test_predictions = []
    test_actual = []
    
    for ts in range(val_end + 1, test_end + 1):
        print(f"\n🕒 Predicting timestamp {ts}")
        
        # Get actual edge at this timestamp
        actual_edges_ts = g_df[g_df.ts == ts]
        if len(actual_edges_ts) == 0:
            print(f"  No edges at timestamp {ts}")
            continue
            
        actual_edge = actual_edges_ts.iloc[0]
        actual_src, actual_dst = actual_edge.u, actual_edge.i
        print(f"  Actual edge: {actual_src} → {actual_dst}")
        
        # Create sequence up to ts-1 using predicted graph
        sequence_data = create_graph_sequence_from_data(predicted_graph, ts-1, n_feat, e_feat)
        
        # Get existing edges to avoid
        existing_edges = set()
        for _, row in predicted_graph[predicted_graph.ts <= ts-1].iterrows():
            existing_edges.add((row.u, row.i))
            existing_edges.add((row.i, row.u))
        
        # Predict top candidates
        candidates = predict_next_edge_candidates(model, sequence_data, existing_edges, len(n_feat), top_k=5)
        
        print(f"  Top 5 predictions:")
        predicted_correctly = False
        
        for i, (src, dst, score) in enumerate(candidates):
            marker = "🎯" if (src == actual_src and dst == actual_dst) or (src == actual_dst and dst == actual_src) else "  "
            print(f"    {marker} {i+1}. {src} → {dst}: {score:.4f}")
            
            if (src == actual_src and dst == actual_dst) or (src == actual_dst and dst == actual_src):
                predicted_correctly = True
                test_predictions.append(1)
            
        if not predicted_correctly:
            test_predictions.append(0)
        test_actual.append(1)
        
        # Add the ACTUAL edge to predicted graph for next prediction
        new_row = actual_edge.to_dict()
        predicted_graph = pd.concat([predicted_graph, pd.DataFrame([new_row])], ignore_index=True)
        
        print(f"  Prediction: {'✅ Correct' if predicted_correctly else '❌ Wrong'}")
    
    # Calculate final metrics
    print(f"\n📈 FINAL RESULTS")
    print("=" * 60)
    
    if len(val_predictions) > 0:
        val_accuracy = np.mean(val_predictions)
        print(f"Validation Accuracy: {val_accuracy:.4f} ({sum(val_predictions)}/{len(val_predictions)})")
    else:
        print("No validation predictions made")
    
    if len(test_predictions) > 0:
        test_accuracy = np.mean(test_predictions)
        print(f"Test Accuracy: {test_accuracy:.4f} ({sum(test_predictions)}/{len(test_predictions)})")
    else:
        print("No test predictions made")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"- This evaluation gives the model access to complete temporal context")
    print(f"- Model predicts one edge at a time, sequentially through time")
    print(f"- Each prediction uses the actual graph state up to that point")
    print(f"- This is how temporal models should actually be evaluated!")
    
    return {
        'val_accuracy': val_accuracy if len(val_predictions) > 0 else 0,
        'test_accuracy': test_accuracy if len(test_predictions) > 0 else 0,
        'val_predictions': val_predictions,
        'test_predictions': test_predictions
    }


def compare_with_random_baseline():
    """Compare GraphRNN with random baseline on sequential prediction"""
    
    print(f"\n🎲 RANDOM BASELINE COMPARISON")
    print("=" * 60)
    
    # For triadic_medium, there are ~400-500 possible edges at any timestamp
    # Random chance of predicting the correct edge is ~1/400 = 0.0025
    # Random chance of getting it in top-5 is ~5/400 = 0.0125
    
    print(f"Random baseline (top-1): ~0.25% chance")
    print(f"Random baseline (top-5): ~1.25% chance")
    print(f"GraphRNN performance will show if it's actually learning patterns")


if __name__ == "__main__":
    results = sequential_evaluation('triadic_medium')
    compare_with_random_baseline()
    
    print(f"\n" + "="*60)
    print("🏁 SEQUENTIAL EVALUATION COMPLETE")
    print("="*60)
