"""
Training script for TGAM on synthetic triadic closure dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from tgam_fixed import TGAM_LinkPrediction
from utils import RandEdgeSampler

def train_tgam_synthetic():
    """Train TGAM on corrected synthetic triadic closure dataset"""
    
    # Load corrected synthetic dataset
    data = 'triadic_fixed'
    g_df = pd.read_csv(f'./processed/{data}/ml_{data}.csv')
    e_feat = np.load(f'./processed/{data}/ml_{data}.npy')
    n_feat = np.load(f'./processed/{data}/ml_{data}_node.npy')
    
    print(f"=== Training TGAM on {data} ===")
    print(f"Dataset: {len(g_df)} edges, {len(n_feat)} nodes")
    print(f"Edge features: {e_feat.shape}")
    print(f"Node features: {n_feat.shape}")
    
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
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_idx = max(g_df.u.max(), g_df.i.max())
    
    model = TGAM_LinkPrediction(
        node_feat_dim=n_feat.shape[1],
        edge_feat_dim=e_feat.shape[1],
        hidden_dim=64,
        max_nodes=max_idx + 1,
        num_graph_layers=2,
        num_temporal_layers=2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    sampler = RandEdgeSampler(train_src_l, train_dst_l)
    
    # Training loop
    num_epochs = 20
    training_steps_per_epoch = min(50, len(train_src_l) - first_prediction_idx)  # Limit for speed
    
    print(f"\n=== Training Configuration ===")
    print(f"Epochs: {num_epochs}")
    print(f"Steps per epoch: {training_steps_per_epoch}")
    print(f"Device: {device}")
    print(f"Learning rate: 0.001")
    
    # Training metrics
    epoch_losses = []
    epoch_accuracies = []
    
    print(f"\n=== Training Progress ===")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        start_time = time.time()
        
        for step in range(training_steps_per_epoch):
            # Get random training step
            k = first_prediction_idx + (step % (len(train_src_l) - first_prediction_idx))
            
            optimizer.zero_grad()
            
            # Positive sample (real edge)
            pos_prob = model(
                train_src_l[:k+1], train_dst_l[:k+1], train_dst_l[k],
                train_ts_l[:k+1], train_e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Negative sample (random edge)
            u_fake, i_fake = sampler.sample(1)
            neg_prob = model(
                train_src_l[:k+1], train_dst_l[:k+1], i_fake[0],
                train_ts_l[:k+1], train_e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Loss computation
            pos_label = torch.ones(1, device=device)
            neg_label = torch.zeros(1, device=device)
            
            loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
            loss.backward()
            optimizer.step()
            
            # Accuracy computation
            with torch.no_grad():
                pred_score = torch.cat([pos_prob, neg_prob]).cpu().numpy()
                pred_label = pred_score > 0.5
                true_label = np.array([1, 0])
                acc = (pred_label == true_label).mean()
                
                epoch_loss += loss.item()
                epoch_acc += acc
        
        avg_loss = epoch_loss / training_steps_per_epoch
        avg_acc = epoch_acc / training_steps_per_epoch
        epoch_time = time.time() - start_time
        
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(avg_acc)
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Time={epoch_time:.2f}s")
        
        # Early stopping if accuracy is consistently high
        if epoch >= 5 and np.mean(epoch_accuracies[-3:]) > 0.8:
            print(f"Early stopping: High accuracy achieved!")
            break
    
    # Evaluation phase
    print(f"\n=== Final Evaluation ===")
    model.eval()
    
    test_steps = min(20, len(train_src_l) - first_prediction_idx)
    eval_accuracies = []
    eval_aps = []
    eval_aucs = []
    pos_probs = []
    neg_probs = []
    
    with torch.no_grad():
        for step in range(test_steps):
            k = first_prediction_idx + step
            if k >= len(train_src_l):
                break
                
            # Positive prediction
            pos_prob = model(
                train_src_l[:k+1], train_dst_l[:k+1], train_dst_l[k],
                train_ts_l[:k+1], train_e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Negative prediction
            u_fake, i_fake = sampler.sample(1)
            neg_prob = model(
                train_src_l[:k+1], train_dst_l[:k+1], i_fake[0],
                train_ts_l[:k+1], train_e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Metrics
            pred_score = torch.cat([pos_prob, neg_prob]).cpu().numpy()
            pred_label = pred_score > 0.5
            true_label = np.array([1, 0])
            
            acc = (pred_label == true_label).mean()
            ap = average_precision_score(true_label, pred_score)
            auc = roc_auc_score(true_label, pred_score)
            
            eval_accuracies.append(acc)
            eval_aps.append(ap)
            eval_aucs.append(auc)
            pos_probs.append(pos_prob.item())
            neg_probs.append(neg_prob.item())
    
    # Final results
    final_acc = np.mean(eval_accuracies)
    final_ap = np.mean(eval_aps)
    final_auc = np.mean(eval_aucs)
    
    print(f"\nFinal Results:")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  AP Score: {final_ap:.4f}")
    print(f"  AUC Score: {final_auc:.4f}")
    print(f"  Pos prob range: [{min(pos_probs):.4f}, {max(pos_probs):.4f}]")
    print(f"  Neg prob range: [{min(neg_probs):.4f}, {max(neg_probs):.4f}]")
    
    # Check for learning
    print(f"\n=== Learning Assessment ===")
    if final_acc > 0.6:
        print("✅ TGAM shows strong learning capability!")
    elif final_acc > 0.55:
        print("⚠️  TGAM shows moderate learning capability")
    else:
        print("❌ TGAM shows limited learning capability")
    
    # Training progress analysis
    improvement = epoch_accuracies[-1] - epoch_accuracies[0] if len(epoch_accuracies) > 1 else 0
    print(f"Training improvement: {improvement:.4f}")
    
    if improvement > 0.1:
        print("✅ Good training progress observed")
    elif improvement > 0.05:
        print("⚠️  Moderate training progress")
    else:
        print("❌ Limited training progress")
    
    # Prediction variance analysis
    pos_var = np.var(pos_probs)
    neg_var = np.var(neg_probs)
    total_var = np.var(pos_probs + neg_probs)
    
    print(f"\nPrediction variance: {total_var:.6f}")
    if total_var > 0.001:
        print("✅ Model produces diverse predictions")
    else:
        print("❌ Model predictions lack diversity")
    
    # Save model if performing well
    if final_acc > 0.6:
        torch.save(model.state_dict(), f'tgam_{data}_trained.pth')
        print(f"Model saved to tgam_{data}_trained.pth")
    
    return {
        'final_accuracy': final_acc,
        'final_ap': final_ap,
        'final_auc': final_auc,
        'training_losses': epoch_losses,
        'training_accuracies': epoch_accuracies,
        'improvement': improvement,
        'prediction_variance': total_var
    }

if __name__ == '__main__':
    results = train_tgam_synthetic()
    print(f"\n=== Training Complete ===") 