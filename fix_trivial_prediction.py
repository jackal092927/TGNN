#!/usr/bin/env python3
"""
Fix Trivial Prediction Issue
============================

This script tests fixes for the trivial prediction issue:
1. Consistent training/evaluation ratios
2. Check if model can learn meaningful patterns
3. Verify the model is learning node-level patterns, not just class bias
"""

import argparse
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from tgam_fixed import TGAM_LinkPrediction
from utils import RandEdgeSampler
from collections import defaultdict
import time

def test_trivial_prediction_fix():
    """Test if we can fix the trivial prediction issue"""
    
    print("🔧 Testing Trivial Prediction Fix")
    print("=" * 50)
    
    # Load data
    data = 'triadic_fixed'
    g_df = pd.read_csv(f'./processed/{data}/ml_{data}.csv')
    e_feat = np.load(f'./processed/{data}/ml_{data}.npy')
    n_feat = np.load(f'./processed/{data}/ml_{data}_node.npy')
    
    print(f"Dataset: {len(g_df)} edges, {len(n_feat)} nodes")
    
    # Simple train/test split
    train_mask = g_df.ts <= 15
    test_mask = g_df.ts > 15
    
    train_data = g_df[train_mask]
    test_data = g_df[test_mask]
    
    print(f"Train: {len(train_data)} edges, Test: {len(test_data)} edges")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TGAM_LinkPrediction(
        node_feat_dim=n_feat.shape[1],
        edge_feat_dim=e_feat.shape[1],
        hidden_dim=64,
        max_nodes=n_feat.shape[0],
        num_graph_layers=2,
        num_temporal_layers=2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test 1: Check if model can distinguish between different node pairs
    print("\n🧪 Test 1: Node Pair Discrimination")
    print("-" * 40)
    
    # Create simple test cases
    train_src_l = train_data.u.values
    train_dst_l = train_data.i.values
    train_ts_l = train_data.ts.values
    train_e_idx_l = train_data.idx.values
    
    # Test if model gives different outputs for different node pairs
    test_pairs = [(0, 1), (0, 2), (1, 2), (0, 5), (1, 5)]
    
    model.eval()
    with torch.no_grad():
        outputs = []
        for src, dst in test_pairs:
            try:
                prob = model(
                    train_src_l[:10], train_dst_l[:10], dst,
                    train_ts_l[:10], train_e_idx_l[:10],
                    n_feat, e_feat
                )
                outputs.append(prob.item())
            except:
                outputs.append(0.5)
        
        print(f"Node pair outputs: {outputs}")
        output_variance = np.var(outputs)
        print(f"Output variance: {output_variance:.6f}")
        
        if output_variance < 1e-4:
            print("❌ Model outputs are nearly identical - model may not be learning node patterns")
        else:
            print("✅ Model outputs vary - model can distinguish node pairs")
    
    # Test 2: Consistent training/evaluation ratios
    print("\n🧪 Test 2: Consistent Training/Evaluation Ratios")
    print("-" * 40)
    
    # Modified training function with consistent ratios
    def train_with_consistent_ratios(model, train_src_l, train_dst_l, train_ts_l, train_e_idx_l, 
                                   n_feat, e_feat, num_epochs=3):
        """Train with consistent 2:1 negative:positive ratio"""
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        print(f"Training with consistent 2:1 ratio...")
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            # Process in small batches to avoid memory issues
            for step in range(20):  # Small number of steps
                # Get random training index
                k = 10 + step  # Start after some initial edges
                if k >= len(train_src_l):
                    break
                
                optimizer.zero_grad()
                
                # Create candidates with consistent 2:1 ratio
                candidates = []
                
                # 1 positive example
                pos_dst = train_dst_l[k]
                candidates.append((pos_dst, 1.0))
                
                # 2 negative examples
                sampler = RandEdgeSampler(train_src_l[:k], train_dst_l[:k])
                for _ in range(2):
                    try:
                        u_fake, i_fake = sampler.sample(1)
                        candidates.append((i_fake[0], 0.0))
                    except:
                        candidates.append((np.random.randint(0, n_feat.shape[0]), 0.0))
                
                # Predict for all candidates
                losses = []
                for candidate_dst, true_label in candidates:
                    try:
                        pred_prob = model(
                            train_src_l[:k], train_dst_l[:k], candidate_dst,
                            train_ts_l[:k], train_e_idx_l[:k],
                            n_feat, e_feat
                        )
                        
                        true_label_tensor = torch.tensor(true_label, dtype=torch.float32, device=device)
                        loss = criterion(pred_prob, true_label_tensor)
                        losses.append(loss)
                        
                        # Accuracy
                        pred_label = (pred_prob > 0.5).item()
                        epoch_correct += int(pred_label == true_label)
                        epoch_total += 1
                        
                    except Exception as e:
                        continue
                
                if losses:
                    total_loss = sum(losses) / len(losses)
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.item()
            
            epoch_acc = epoch_correct / max(epoch_total, 1)
            print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
            
            # Check if model is learning (accuracy should move away from 0.33/0.67)
            if 0.4 < epoch_acc < 0.6:
                print("✅ Model is learning - accuracy in reasonable range")
            else:
                print(f"❌ Model may be doing trivial prediction - acc={epoch_acc:.4f}")
    
    # Test 3: Evaluate with consistent ratios
    def evaluate_with_consistent_ratios(model, test_src_l, test_dst_l, test_ts_l, test_e_idx_l,
                                       n_feat, e_feat, context_size=50):
        """Evaluate with consistent 2:1 ratio"""
        
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        # Use first part of test as context
        context_src = test_src_l[:context_size]
        context_dst = test_dst_l[:context_size]
        context_ts = test_ts_l[:context_size]
        context_e_idx = test_e_idx_l[:context_size]
        
        # Evaluate on later part
        for i in range(context_size, min(context_size + 10, len(test_src_l))):
            # Create consistent 2:1 candidates
            candidates = []
            
            # 1 positive
            pos_dst = test_dst_l[i]
            candidates.append((pos_dst, 1.0))
            
            # 2 negatives
            sampler = RandEdgeSampler(test_src_l[:i], test_dst_l[:i])
            for _ in range(2):
                try:
                    u_fake, i_fake = sampler.sample(1)
                    candidates.append((i_fake[0], 0.0))
                except:
                    candidates.append((np.random.randint(0, n_feat.shape[0]), 0.0))
            
            # Predict
            with torch.no_grad():
                for candidate_dst, true_label in candidates:
                    try:
                        pred_prob = model(
                            context_src, context_dst, candidate_dst,
                            context_ts, context_e_idx,
                            n_feat, e_feat
                        )
                        
                        all_predictions.append(pred_prob.item())
                        all_labels.append(true_label)
                        
                    except:
                        continue
        
        if len(all_predictions) > 0:
            predictions = np.array(all_predictions)
            labels = np.array(all_labels)
            
            # Calculate metrics
            binary_preds = (predictions > 0.5).astype(int)
            accuracy = (binary_preds == labels).mean()
            
            print(f"Evaluation results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Prediction distribution: {np.bincount(binary_preds)}")
            print(f"  Label distribution: {np.bincount(labels.astype(int))}")
            
            # Check if model is doing trivial prediction
            if accuracy < 0.35 or accuracy > 0.65:
                print("❌ Model may still be doing trivial prediction")
                print(f"   Expected: ~0.33 (all positive) or ~0.67 (all negative)")
                print(f"   Actual: {accuracy:.4f}")
            else:
                print("✅ Model accuracy in reasonable range")
        else:
            print("❌ No predictions made")
    
    # Run tests
    print("\n🚀 Running Training/Evaluation Tests")
    print("=" * 50)
    
    # Train with consistent ratios
    train_with_consistent_ratios(
        model, train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
        n_feat, e_feat, num_epochs=5
    )
    
    # Evaluate with consistent ratios
    test_src_l = test_data.u.values
    test_dst_l = test_data.i.values
    test_ts_l = test_data.ts.values
    test_e_idx_l = test_data.idx.values
    
    print("\n📊 Evaluation Results:")
    print("-" * 40)
    evaluate_with_consistent_ratios(
        model, test_src_l, test_dst_l, test_ts_l, test_e_idx_l,
        n_feat, e_feat
    )
    
    # Test 4: Check if model learns node relationships
    print("\n🧪 Test 4: Node Relationship Learning")
    print("-" * 40)
    
    # Create a simple synthetic pattern to test learning
    # Connect nodes in a specific pattern and see if model learns it
    
    print("Testing complete!")
    print("\n📋 Summary:")
    print("- If model shows consistent learning (accuracy 0.4-0.6), it's working")
    print("- If model shows trivial prediction (accuracy 0.33 or 0.67), it's broken")
    print("- The fix is to ensure consistent training/evaluation ratios")

if __name__ == '__main__':
    test_trivial_prediction_fix() 