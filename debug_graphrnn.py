"""
Debug GraphRNN performance on triadic closure
Analyze why test accuracy is low despite training progress
"""

import torch
import pandas as pd
import numpy as np
import json
from graph_rnn import GraphRNN, create_graph_sequence_from_data
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


def debug_triadic_predictions():
    """Debug GraphRNN predictions step by step"""
    
    print("🔍 DEBUGGING GRAPHRNN TRIADIC PREDICTIONS")
    print("=" * 60)
    
    # Load data
    data_name = 'triadic_medium'
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    print(f"Dataset: {len(g_df)} edges, {len(n_feat)} nodes")
    print(f"Ground truth triadic closures: {len(ground_truth)}")
    
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
        print("❌ Could not load trained model, using random weights")
    
    model.eval()
    
    # Analyze each test prediction
    print(f"\n📊 ANALYZING TEST PREDICTIONS")
    print("=" * 60)
    
    # Test data (timestamps 15-20)
    test_timestamps = [15, 16, 17, 18, 19, 20]
    
    all_predictions = []
    all_labels = []
    
    for ts in test_timestamps:
        print(f"\n🕒 TIMESTAMP {ts}")
        print("-" * 30)
        
        # Get the actual edge at this timestamp
        actual_edge = g_df[g_df.ts == ts]
        if len(actual_edge) == 0:
            continue
            
        actual_src = actual_edge.iloc[0].u
        actual_dst = actual_edge.iloc[0].i
        actual_idx = actual_edge.iloc[0].idx
        
        print(f"Actual edge: {actual_src} → {actual_dst} (idx: {actual_idx})")
        
        # Check if this is a triadic closure
        is_triadic = str(actual_idx) in ground_truth
        if is_triadic:
            parent_edges = ground_truth[str(actual_idx)]
            print(f"✅ Triadic closure caused by edges: {parent_edges}")
        else:
            print("❌ Not a triadic closure (noise edge)")
        
        # Create sequence up to ts-1
        sequence_data = create_graph_sequence_from_data(g_df, ts-1, n_feat, e_feat)
        
        print(f"Sequence length: {len(sequence_data)}")
        
        # Get all possible candidate edges at this timestamp
        existing_edges = set()
        edges_up_to_ts = g_df[g_df.ts <= ts-1]
        for _, row in edges_up_to_ts.iterrows():
            existing_edges.add((row.u, row.i))
            existing_edges.add((row.i, row.u))  # Undirected
        
        # Generate candidate edges (all non-existing pairs)
        num_nodes = len(n_feat)
        candidates = []
        for src in range(num_nodes):
            for dst in range(src + 1, num_nodes):  # Avoid duplicates
                if (src, dst) not in existing_edges:
                    candidates.append((src, dst))
        
        print(f"Candidate edges: {len(candidates)}")
        
        if len(candidates) > 100:  # Limit for efficiency
            candidates = candidates[:100]
            print(f"Limited to first {len(candidates)} candidates")
        
        # Score all candidates
        candidate_scores = []
        
        with torch.no_grad():
            for src, dst in candidates:
                prob, _ = model(sequence_data, target_src=src, target_dst=dst)
                if prob is not None:
                    score = prob.item()
                    candidate_scores.append((src, dst, score))
        
        # Sort by score
        candidate_scores.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nTop 5 predicted edges:")
        for i, (src, dst, score) in enumerate(candidate_scores[:5]):
            marker = "🎯" if (src, dst) == (actual_src, actual_dst) or (dst, src) == (actual_src, actual_dst) else "  "
            print(f"  {marker} {i+1}. {src} → {dst}: {score:.4f}")
        
        # Check if actual edge is in top predictions
        actual_score = None
        actual_rank = None
        
        for i, (src, dst, score) in enumerate(candidate_scores):
            if (src, dst) == (actual_src, actual_dst) or (dst, src) == (actual_src, actual_dst):
                actual_score = score
                actual_rank = i + 1
                break
        
        if actual_score is not None:
            print(f"\n🎯 Actual edge score: {actual_score:.4f} (rank {actual_rank}/{len(candidate_scores)})")
            
            # Binary prediction (threshold = 0.5)
            prediction = 1 if actual_score > 0.5 else 0
            all_predictions.append(prediction)
            all_labels.append(1)  # Actual edge is positive
            
            # Add negative samples for evaluation
            for src, dst, score in candidate_scores[:5]:  # Top 5 negatives
                if (src, dst) != (actual_src, actual_dst) and (dst, src) != (actual_src, actual_dst):
                    neg_prediction = 1 if score > 0.5 else 0
                    all_predictions.append(neg_prediction)
                    all_labels.append(0)
        else:
            print(f"❌ Actual edge not found in candidates!")
    
    # Overall evaluation
    print(f"\n📈 OVERALL TEST PERFORMANCE")
    print("=" * 60)
    
    if len(all_predictions) > 0:
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        # Calculate AUC and AP if we have both classes
        if len(np.unique(all_labels)) > 1:
            # For AUC/AP we need probabilities, not binary predictions
            # Let's recalculate with actual scores
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Positive samples: {np.sum(all_labels)}")
            print(f"Negative samples: {len(all_labels) - np.sum(all_labels)}")
            print(f"Predicted positive: {np.sum(all_predictions)}")
        else:
            print(f"Only one class present in evaluation")
    else:
        print("No valid predictions made")


def test_triadic_closure_directly():
    """Test if the model can predict triadic closures directly"""
    
    print(f"\n🔬 DIRECT TRIADIC CLOSURE TEST")
    print("=" * 60)
    
    # Load data
    data_name = 'triadic_medium'
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    # Load model
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
        print("❌ Using random model")
        return
    
    model.eval()
    
    # Test on known triadic closures vs random edges
    print(f"\nTesting known triadic closures...")
    
    triadic_scores = []
    random_scores = []
    
    # Test each triadic closure
    for edge_idx_str, parent_edges in ground_truth.items():
        edge_idx = int(edge_idx_str)
        
        # Find the edge
        edge_row = g_df[g_df.idx == edge_idx]
        if len(edge_row) == 0:
            continue
            
        edge_row = edge_row.iloc[0]
        src, dst, ts = edge_row.u, edge_row.i, edge_row.ts
        
        # Create sequence up to ts-1
        sequence_data = create_graph_sequence_from_data(g_df, ts-1, n_feat, e_feat)
        
        # Score this triadic closure
        with torch.no_grad():
            prob, _ = model(sequence_data, target_src=src, target_dst=dst)
            if prob is not None:
                triadic_scores.append(prob.item())
                print(f"Triadic {edge_idx} ({src}→{dst} at ts={ts}): {prob.item():.4f}")
    
    # Test random non-existing edges
    print(f"\nTesting random non-edges...")
    
    # Get existing edges up to timestamp 10
    existing_edges = set()
    edges_up_to_10 = g_df[g_df.ts <= 10]
    for _, row in edges_up_to_10.iterrows():
        existing_edges.add((row.u, row.i))
        existing_edges.add((row.i, row.u))
    
    # Generate random non-edges
    num_nodes = len(n_feat)
    sequence_data = create_graph_sequence_from_data(g_df, 10, n_feat, e_feat)
    
    random_tested = 0
    for _ in range(50):  # Test 50 random non-edges
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        
        if src != dst and (src, dst) not in existing_edges:
            with torch.no_grad():
                prob, _ = model(sequence_data, target_src=src, target_dst=dst)
                if prob is not None:
                    random_scores.append(prob.item())
                    random_tested += 1
                    if random_tested <= 10:  # Print first 10
                        print(f"Random {src}→{dst}: {prob.item():.4f}")
    
    # Compare distributions
    print(f"\n📊 SCORE COMPARISON")
    print("-" * 30)
    
    if len(triadic_scores) > 0 and len(random_scores) > 0:
        triadic_mean = np.mean(triadic_scores)
        random_mean = np.mean(random_scores)
        
        print(f"Triadic closures ({len(triadic_scores)}): {triadic_mean:.4f} ± {np.std(triadic_scores):.4f}")
        print(f"Random edges ({len(random_scores)}): {random_mean:.4f} ± {np.std(random_scores):.4f}")
        print(f"Difference: {triadic_mean - random_mean:.4f}")
        
        if triadic_mean > random_mean:
            print("✅ Model assigns higher scores to triadic closures!")
        else:
            print("❌ Model does not distinguish triadic closures")
    else:
        print("❌ Insufficient data for comparison")


if __name__ == "__main__":
    debug_triadic_predictions()
    test_triadic_closure_directly()
