"""
Debug script to understand why TGAM accuracy is exactly 0.5
"""

import torch
import pandas as pd
import numpy as np
from tgam_fixed import TGAM_LinkPrediction
from utils import RandEdgeSampler

def debug_model_predictions():
    """Debug the model predictions step by step"""
    
    # Load corrected dataset with proper triadic closure
    data = 'triadic_fixed'
    g_df = pd.read_csv(f'./processed/{data}/ml_{data}.csv')
    e_feat = np.load(f'./processed/{data}/ml_{data}.npy')
    n_feat = np.load(f'./processed/{data}/ml_{data}_node.npy')
    
    print(f"Dataset: {len(g_df)} edges, {len(n_feat)} nodes")
    print(f"Edge features shape: {e_feat.shape}")
    print(f"Node features shape: {n_feat.shape}")
    
    # Correct temporal setup: timestamp 0 = initial state, training starts from timestamp 1
    val_time = np.quantile(g_df.ts, 0.7)
    train_mask = g_df.ts <= val_time
    train_data = g_df[train_mask]
    
    # Use ALL edges in training set (including initial state at timestamp 0)
    train_src_l = train_data.u.values
    train_dst_l = train_data.i.values 
    train_ts_l = train_data.ts.values
    train_e_idx_l = train_data.idx.values
    
    print(f"Temporal structure:")
    print(f"  Timestamp 0 edges: {np.sum(train_ts_l == 0)} (initial state)")
    print(f"  Timestamp 1+ edges: {np.sum(train_ts_l > 0)} (training targets)")
    
    print(f"Training edges: {len(train_src_l)}")
    print(f"Training source nodes: {train_src_l}")
    print(f"Training dest nodes: {train_dst_l}")
    print(f"Training timestamps: {train_ts_l}")
    
    # Initialize model
    device = torch.device('cpu')  # Use CPU for easier debugging
    max_idx = max(g_df.u.max(), g_df.i.max())
    
    model = TGAM_LinkPrediction(
        node_feat_dim=n_feat.shape[1],
        edge_feat_dim=e_feat.shape[1],
        hidden_dim=32,  # Small for debugging
        max_nodes=max_idx + 1,
        num_graph_layers=1,
        num_temporal_layers=1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test single prediction
    k = 3  # Use first 3 edges as history, predict 4th
    
    print(f"\n=== Testing at step k={k} ===")
    print(f"History: src={train_src_l[:k+1]}, dst={train_dst_l[:k+1]}")
    print(f"Target edge: ({train_src_l[k]}, {train_dst_l[k]})")
    
    # Positive sample (real edge)
    pos_prob = model(
        train_src_l[:k+1], train_dst_l[:k+1], train_dst_l[k],
        train_ts_l[:k+1], train_e_idx_l[:k+1],
        n_feat, e_feat
    )
    
    print(f"Positive probability: {pos_prob}")
    print(f"Positive probability value: {pos_prob.item():.6f}")
    
    # Negative sample (random edge)
    sampler = RandEdgeSampler(train_src_l, train_dst_l)
    u_fake, i_fake = sampler.sample(1)
    fake_dst = i_fake[0]
    
    print(f"Fake destination: {fake_dst}")
    
    neg_prob = model(
        train_src_l[:k+1], train_dst_l[:k+1], fake_dst,
        train_ts_l[:k+1], train_e_idx_l[:k+1],
        n_feat, e_feat
    )
    
    print(f"Negative probability: {neg_prob}")
    print(f"Negative probability value: {neg_prob.item():.6f}")
    
    # Test accuracy computation (same as training)
    pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
    pred_label = pred_score > 0.5
    true_label = np.concatenate([np.ones(1), np.zeros(1)])
    
    print(f"\n=== Accuracy Computation ===")
    print(f"Pred scores: {pred_score}")
    print(f"Pred labels (>0.5): {pred_label}")
    print(f"True labels: {true_label}")
    print(f"Accuracy: {(pred_label == true_label).mean()}")
    
    # Find the first edge with timestamp > 0 (first prediction target)
    first_prediction_idx = np.where(train_ts_l > 0)[0][0] if np.any(train_ts_l > 0) else 1
    print(f"\nFirst prediction starts at index {first_prediction_idx} (timestamp {train_ts_l[first_prediction_idx]})")
    
    # Test with correct temporal setup
    print(f"\n=== Testing Correct Temporal Setup ===")
    accuracies = []
    pos_probs = []
    neg_probs = []
    
    for k in range(first_prediction_idx, min(len(train_src_l), first_prediction_idx + 7)):
        print(f"\nStep {k}: predicting edge ({train_src_l[k]}, {train_dst_l[k]}) at timestamp {train_ts_l[k]}")
        print(f"  History: {k+1} edges from timestamp 0 to {train_ts_l[k-1] if k > 0 else 0}")
        
        # Positive (real edge at current timestamp)
        pos_prob = model(
            train_src_l[:k+1], train_dst_l[:k+1], train_dst_l[k],
            train_ts_l[:k+1], train_e_idx_l[:k+1],
            n_feat, e_feat
        )
        
        # Negative (random edge)
        u_fake, i_fake = sampler.sample(1)
        fake_dst = i_fake[0]
        
        neg_prob = model(
            train_src_l[:k+1], train_dst_l[:k+1], fake_dst,
            train_ts_l[:k+1], train_e_idx_l[:k+1],
            n_feat, e_feat
        )
        
        # Accuracy
        pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
        pred_label = pred_score > 0.5
        true_label = np.concatenate([np.ones(1), np.zeros(1)])
        acc = (pred_label == true_label).mean()
        
        accuracies.append(acc)
        pos_probs.append(pos_prob.item())
        neg_probs.append(neg_prob.item())
        
        print(f"  pos={pos_prob.item():.4f}, neg={neg_prob.item():.4f}, acc={acc:.4f}")
    
    print(f"\n=== Summary ===")
    print(f"Mean accuracy: {np.mean(accuracies):.4f}")
    print(f"Pos prob range: [{min(pos_probs):.4f}, {max(pos_probs):.4f}]")
    print(f"Neg prob range: [{min(neg_probs):.4f}, {max(neg_probs):.4f}]")
    
    # Check if model is initialized properly
    print(f"\n=== Model Weight Check ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
            if torch.isnan(param.data).any():
                print(f"WARNING: NaN in {name}")
            if torch.isinf(param.data).any():
                print(f"WARNING: Inf in {name}")


if __name__ == '__main__':
    debug_model_predictions() 