"""
Test Script: Compare Original vs Improved TGAM Models

This script tests both models side-by-side to see if the improvements
help with the dying neuron problem and overall learning capability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from collections import defaultdict

from tgam_fixed import TGAM_LinkPrediction
from tgam_improved import TGAM_LinkPrediction_Improved
from utils import RandEdgeSampler

def analyze_model_health(model, model_name):
    """Analyze model initialization and architecture health"""
    print(f"\n=== {model_name} Health Analysis ===")
    
    total_params = 0
    relu_layers = 0
    leaky_relu_layers = 0
    linear_layers = 0
    
    init_stats = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers += 1
            weight = module.weight.data.cpu().numpy()
            init_stats[name] = {
                'mean': np.mean(weight),
                'std': np.std(weight),
                'min': np.min(weight),
                'max': np.max(weight)
            }
        elif isinstance(module, nn.ReLU):
            relu_layers += 1
        elif isinstance(module, nn.LeakyReLU):
            leaky_relu_layers += 1
    
    for param in model.parameters():
        total_params += param.numel()
    
    print(f"Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Linear layers: {linear_layers}")
    print(f"  ReLU layers: {relu_layers}")
    print(f"  LeakyReLU layers: {leaky_relu_layers}")
    
    print(f"\nWeight Initialization (first 3 layers):")
    for i, (name, stats) in enumerate(list(init_stats.items())[:3]):
        print(f"  {name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
              f"range=[{stats['min']:.6f}, {stats['max']:.6f}]")
    
    return init_stats

def test_gradient_flow(model, model_name, src_l, dst_l, ts_l, e_idx_l, n_feat, e_feat, sampler, device):
    """Test gradient flow and dead neuron detection"""
    print(f"\n=== {model_name} Gradient Flow Test ===")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Test for 5 steps
    dead_neuron_history = []
    gradient_norms = []
    
    for step in range(5):
        optimizer.zero_grad()
        
        total_loss = 0.0
        batch_size = 3
        
        for i in range(batch_size):
            k = np.random.randint(1, min(10, len(src_l)))
            
            # Positive example
            pos_prob = model(
                src_l[:k+1], dst_l[:k+1], dst_l[k],
                ts_l[:k+1], e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Negative example
            u_fake, i_fake = sampler.sample(1)
            neg_prob = model(
                src_l[:k+1], dst_l[:k+1], i_fake[0],
                ts_l[:k+1], e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Loss
            pos_loss = criterion(pos_prob.squeeze(), torch.tensor(1.0, device=device))
            neg_loss = criterion(neg_prob.squeeze(), torch.tensor(0.0, device=device))
            loss = pos_loss + neg_loss
            total_loss += loss.item()
            
            loss.backward()
        
        # Analyze gradients
        total_dead_neurons = 0
        total_neurons = 0
        layer_grad_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None and 'weight' in name:
                grad = param.grad.data.cpu().numpy()
                zero_fraction = np.mean(grad == 0.0)
                grad_norm = np.linalg.norm(grad)
                
                total_dead_neurons += np.sum(grad == 0.0)
                total_neurons += grad.size
                layer_grad_norms.append(grad_norm)
        
        dead_fraction = total_dead_neurons / max(total_neurons, 1)
        avg_grad_norm = np.mean(layer_grad_norms) if layer_grad_norms else 0.0
        
        dead_neuron_history.append(dead_fraction)
        gradient_norms.append(avg_grad_norm)
        
        optimizer.step()
        
        print(f"Step {step+1}: Loss={total_loss:.6f}, Dead neurons={dead_fraction*100:.1f}%, "
              f"Avg grad norm={avg_grad_norm:.6e}")
    
    # Summary
    final_dead_fraction = dead_neuron_history[-1]
    final_grad_norm = gradient_norms[-1]
    
    print(f"\nFinal Results:")
    print(f"  Dead neuron percentage: {final_dead_fraction*100:.1f}%")
    print(f"  Average gradient norm: {final_grad_norm:.6e}")
    
    # Health assessment
    if final_dead_fraction > 0.3:
        print("  🔴 HIGH dead neuron rate - model may struggle to learn")
    elif final_dead_fraction > 0.1:
        print("  🟡 MODERATE dead neuron rate - room for improvement")
    else:
        print("  🟢 LOW dead neuron rate - healthy gradient flow")
    
    return dead_neuron_history, gradient_norms

def test_learning_capability(model, model_name, src_l, dst_l, ts_l, e_idx_l, n_feat, e_feat, sampler, device):
    """Test model's learning capability over more steps"""
    print(f"\n=== {model_name} Learning Capability Test ===")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    losses = []
    
    for step in range(20):
        optimizer.zero_grad()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        batch_size = 5
        
        for i in range(batch_size):
            k = np.random.randint(1, min(15, len(src_l)))
            
            # Positive example
            pos_prob = model(
                src_l[:k+1], dst_l[:k+1], dst_l[k],
                ts_l[:k+1], e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Negative example
            u_fake, i_fake = sampler.sample(1)
            neg_prob = model(
                src_l[:k+1], dst_l[:k+1], i_fake[0],
                ts_l[:k+1], e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Loss
            pos_loss = criterion(pos_prob.squeeze(), torch.tensor(1.0, device=device))
            neg_loss = criterion(neg_prob.squeeze(), torch.tensor(0.0, device=device))
            loss = pos_loss + neg_loss
            total_loss += loss.item()
            
            # Accuracy
            pos_pred = (pos_prob.item() > 0.5)
            neg_pred = (neg_prob.item() <= 0.5)
            correct += int(pos_pred) + int(neg_pred)
            total += 2
            
            loss.backward()
        
        optimizer.step()
        losses.append(total_loss)
        
        if step % 5 == 0 or step == 19:
            accuracy = correct / total
            print(f"Step {step+1:2d}: Loss={total_loss:.6f}, Accuracy={accuracy:.4f}")
    
    # Learning assessment
    initial_loss = np.mean(losses[:3])
    final_loss = np.mean(losses[-3:])
    loss_improvement = initial_loss - final_loss
    
    print(f"\nLearning Summary:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Improvement: {loss_improvement:.6f}")
    
    if loss_improvement > 0.1:
        print("  🟢 GOOD learning progress")
    elif loss_improvement > 0.01:
        print("  🟡 MODERATE learning progress")
    else:
        print("  🔴 POOR learning progress")
    
    return losses

def main():
    """Main comparison test"""
    print("🧪 TGAM Model Comparison: Original vs Improved")
    print("=" * 60)
    
    # Load data
    print("Loading triadic_fixed dataset...")
    g_df = pd.read_csv('./processed/triadic_fixed/ml_triadic_fixed.csv')
    n_feat = np.load('./processed/triadic_fixed/ml_triadic_fixed_node.npy')
    e_feat = np.load('./processed/triadic_fixed/ml_triadic_fixed.npy')
    
    # Prepare data
    train_ratio = 0.8
    n_train = int(len(g_df) * train_ratio)
    train_data = g_df.iloc[:n_train]
    
    src_l = train_data.u.values
    dst_l = train_data.i.values
    ts_l = train_data.ts.values
    e_idx_l = train_data.idx.values
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert to tensors
    n_feat_tensor = torch.tensor(n_feat, dtype=torch.float32).to(device)
    e_feat_tensor = torch.tensor(e_feat, dtype=torch.float32).to(device)
    
    sampler = RandEdgeSampler(src_l, dst_l)
    
    # Test both models
    models = [
        ("Original TGAM", TGAM_LinkPrediction(
            node_feat_dim=n_feat.shape[1],
            edge_feat_dim=e_feat.shape[1],
            hidden_dim=128,
            max_nodes=100,
            num_graph_layers=2,
            num_temporal_layers=4
        )),
        ("Improved TGAM", TGAM_LinkPrediction_Improved(
            node_feat_dim=n_feat.shape[1],
            edge_feat_dim=e_feat.shape[1],
            hidden_dim=128,
            max_nodes=100,
            num_graph_layers=2,
            num_temporal_layers=4
        ))
    ]
    
    results = {}
    
    for model_name, model in models:
        model.to(device)
        
        # 1. Architecture analysis
        init_stats = analyze_model_health(model, model_name)
        
        # 2. Gradient flow test
        dead_history, grad_norms = test_gradient_flow(
            model, model_name, src_l, dst_l, ts_l, e_idx_l, 
            n_feat_tensor, e_feat_tensor, sampler, device
        )
        
        # 3. Learning capability test
        losses = test_learning_capability(
            model, model_name, src_l, dst_l, ts_l, e_idx_l,
            n_feat_tensor, e_feat_tensor, sampler, device
        )
        
        results[model_name] = {
            'final_dead_fraction': dead_history[-1],
            'final_grad_norm': grad_norms[-1],
            'loss_improvement': np.mean(losses[:3]) - np.mean(losses[-3:]),
            'final_loss': np.mean(losses[-3:])
        }
    
    # Final comparison
    print("\n" + "=" * 60)
    print("🏆 FINAL COMPARISON")
    print("=" * 60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Dead neurons: {metrics['final_dead_fraction']*100:.1f}%")
        print(f"  Gradient norm: {metrics['final_grad_norm']:.6e}")
        print(f"  Loss improvement: {metrics['loss_improvement']:.6f}")
        print(f"  Final loss: {metrics['final_loss']:.6f}")
    
    # Winner determination
    original_metrics = results["Original TGAM"]
    improved_metrics = results["Improved TGAM"]
    
    print(f"\n🎯 IMPROVEMENTS:")
    dead_improvement = original_metrics['final_dead_fraction'] - improved_metrics['final_dead_fraction']
    loss_improvement = improved_metrics['loss_improvement'] - original_metrics['loss_improvement']
    
    print(f"  Dead neuron reduction: {dead_improvement*100:.1f} percentage points")
    print(f"  Better loss improvement: {loss_improvement:.6f}")
    
    if dead_improvement > 0.1 and loss_improvement > 0.01:
        print("  🟢 SIGNIFICANT IMPROVEMENT with new architecture!")
    elif dead_improvement > 0.05 or loss_improvement > 0.005:
        print("  🟡 MODERATE IMPROVEMENT with new architecture")
    else:
        print("  🔴 LIMITED IMPROVEMENT - may need further architectural changes")

if __name__ == "__main__":
    main() 