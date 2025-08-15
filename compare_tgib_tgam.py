"""
Comprehensive comparison between TGIB and TGAM with corrected temporal setup
"""

import torch
import pandas as pd
import numpy as np
import time
from sklearn.metrics import average_precision_score, roc_auc_score
from tgam_fixed import TGAM_LinkPrediction
from module_ori import TGIB
from graph_ori import NeighborFinder
from utils import RandEdgeSampler

def compare_models():
    """Compare TGIB and TGAM with identical temporal setup"""
    
    # Load corrected dataset
    data = 'triadic_fixed'
    g_df = pd.read_csv(f'./processed/{data}/ml_{data}.csv')
    e_feat = np.load(f'./processed/{data}/ml_{data}.npy')
    n_feat = np.load(f'./processed/{data}/ml_{data}_node.npy')
    
    print(f"=== Dataset: {data} ===")
    print(f"Total edges: {len(g_df)}, Nodes: {len(n_feat)}")
    
    # Correct temporal setup
    val_time = np.quantile(g_df.ts, 0.7)
    train_mask = g_df.ts <= val_time
    train_data = g_df[train_mask]
    
    train_src_l = train_data.u.values
    train_dst_l = train_data.i.values 
    train_ts_l = train_data.ts.values
    train_e_idx_l = train_data.idx.values
    
    print(f"\nTemporal structure:")
    print(f"  Timestamp 0 edges: {np.sum(train_ts_l == 0)} (initial state)")
    print(f"  Timestamp 1+ edges: {np.sum(train_ts_l > 0)} (training targets)")
    print(f"  Training edges: {len(train_src_l)}")
    
    # Find first prediction index
    first_prediction_idx = np.where(train_ts_l > 0)[0][0] if np.any(train_ts_l > 0) else 1
    print(f"  First prediction at index: {first_prediction_idx}")
    
    # Setup
    device = torch.device('cpu')
    max_idx = max(g_df.u.max(), g_df.i.max())
    sampler = RandEdgeSampler(train_src_l, train_dst_l)
    
    # Initialize TGAM
    print(f"\n=== Initializing TGAM ===")
    tgam = TGAM_LinkPrediction(
        node_feat_dim=n_feat.shape[1],
        edge_feat_dim=e_feat.shape[1],
        hidden_dim=64,
        max_nodes=max_idx + 1,
        num_graph_layers=2,
        num_temporal_layers=2
    ).to(device)
    
    tgam_params = sum(p.numel() for p in tgam.parameters())
    print(f"TGAM parameters: {tgam_params}")
    
    # Initialize TGIB
    print(f"\n=== Initializing TGIB ===")
    
    # Create neighbor finder for TGIB
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list, uniform=False)
    
    tgib = TGIB(train_ngh_finder, n_feat, e_feat, 64,
                num_layers=2, use_time='time', agg_method='attn', attn_mode='prod',
                seq_len=20, n_head=2, drop_out=0.1, node_dim=100, time_dim=100)
    
    tgib_params = sum(p.numel() for p in tgib.parameters())
    print(f"TGIB parameters: {tgib_params}")
    
    # Test both models without training (random initialization)
    print(f"\n=== Testing Untrained Models ===")
    
    def test_model_predictions(model, model_name, steps=7):
        """Test model predictions over multiple steps"""
        accuracies = []
        pos_probs = []
        neg_probs = []
        
        print(f"\n{model_name} Predictions:")
        
        for k in range(first_prediction_idx, min(len(train_src_l), first_prediction_idx + steps)):
            if model_name == "TGAM":
                # TGAM prediction
                pos_prob = model(
                    train_src_l[:k+1], train_dst_l[:k+1], train_dst_l[k],
                    train_ts_l[:k+1], train_e_idx_l[:k+1],
                    n_feat, e_feat
                )
                
                # Negative sample
                u_fake, i_fake = sampler.sample(1)
                neg_prob = model(
                    train_src_l[:k+1], train_dst_l[:k+1], i_fake[0],
                    train_ts_l[:k+1], train_e_idx_l[:k+1],
                    n_feat, e_feat
                )
                
            else:  # TGIB
                # TGIB prediction
                u_fake, i_fake = sampler.sample(1)
                pos_prob, neg_prob, _ = model(
                    train_src_l[:k+1], train_dst_l[:k+1], i_fake,
                    train_ts_l[:k+1], train_e_idx_l[:k+1], 
                    k, 1, training=False, num_neighbors=20
                )
            
            # Calculate accuracy
            pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(1), np.zeros(1)])
            acc = (pred_label == true_label).mean()
            
            accuracies.append(acc)
            pos_probs.append(pos_prob.item())
            neg_probs.append(neg_prob.item())
            
            print(f"  Step {k}: pos={pos_prob.item():.4f}, neg={neg_prob.item():.4f}, acc={acc:.4f}")
        
        mean_acc = np.mean(accuracies)
        print(f"  Mean accuracy: {mean_acc:.4f}")
        print(f"  Pos range: [{min(pos_probs):.4f}, {max(pos_probs):.4f}]")
        print(f"  Neg range: [{min(neg_probs):.4f}, {max(neg_probs):.4f}]")
        
        return mean_acc, accuracies, pos_probs, neg_probs
    
    # Test TGAM
    tgam.eval()
    tgam_acc, tgam_accs, tgam_pos, tgam_neg = test_model_predictions(tgam, "TGAM")
    
    # Test TGIB  
    tgib.eval()
    tgib_acc, tgib_accs, tgib_pos, tgib_neg = test_model_predictions(tgib, "TGIB")
    
    # Compare variability
    print(f"\n=== Model Comparison ===")
    print(f"TGAM mean accuracy: {tgam_acc:.4f}")
    print(f"TGIB mean accuracy: {tgib_acc:.4f}")
    print(f"TGAM prediction variance: {np.var(tgam_pos + tgam_neg):.6f}")
    print(f"TGIB prediction variance: {np.var(tgib_pos + tgib_neg):.6f}")
    
    # Check if models produce different outputs for different inputs
    print(f"\n=== Model Sensitivity Test ===")
    
    # Test TGAM with different inputs
    k = first_prediction_idx + 2
    
    # Same edge
    prob1 = tgam(train_src_l[:k+1], train_dst_l[:k+1], train_dst_l[k],
                 train_ts_l[:k+1], train_e_idx_l[:k+1], n_feat, e_feat)
    
    # Different edge (random)
    u_fake, i_fake = sampler.sample(1)
    prob2 = tgam(train_src_l[:k+1], train_dst_l[:k+1], i_fake[0],
                 train_ts_l[:k+1], train_e_idx_l[:k+1], n_feat, e_feat)
    
    print(f"TGAM - Same edge: {prob1.item():.6f}, Different edge: {prob2.item():.6f}")
    print(f"TGAM - Difference: {abs(prob1.item() - prob2.item()):.6f}")
    
    # Test TGIB sensitivity
    u_fake1, i_fake1 = sampler.sample(1)
    u_fake2, i_fake2 = sampler.sample(1)
    
    pos_prob1, neg_prob1, _ = tgib(train_src_l[:k+1], train_dst_l[:k+1], i_fake1,
                                   train_ts_l[:k+1], train_e_idx_l[:k+1], 
                                   k, 1, training=False, num_neighbors=20)
    
    pos_prob2, neg_prob2, _ = tgib(train_src_l[:k+1], train_dst_l[:k+1], i_fake2,
                                   train_ts_l[:k+1], train_e_idx_l[:k+1], 
                                   k, 1, training=False, num_neighbors=20)
    
    print(f"TGIB - Sample 1: pos={pos_prob1.item():.6f}, neg={neg_prob1.item():.6f}")
    print(f"TGIB - Sample 2: pos={pos_prob2.item():.6f}, neg={neg_prob2.item():.6f}")
    print(f"TGIB - Pos difference: {abs(pos_prob1.item() - pos_prob2.item()):.6f}")
    print(f"TGIB - Neg difference: {abs(neg_prob1.item() - neg_prob2.item()):.6f}")
    
    print(f"\n=== Conclusion ===")
    if tgam_acc > 0.52:
        print("✅ TGAM shows learning capability (>52% accuracy)")
    else:
        print("❌ TGAM shows no learning (≤52% accuracy)")
        
    if tgib_acc > 0.52:
        print("✅ TGIB shows learning capability (>52% accuracy)")
    else:
        print("❌ TGIB shows no learning (≤52% accuracy)")
    
    print(f"\nBoth models now use identical temporal setup:")
    print(f"- Timestamp 0: Initial graph state (context)")
    print(f"- Timestamp 1+: Prediction targets")
    print(f"- Sequential prediction with cumulative history")


if __name__ == '__main__':
    compare_models() 