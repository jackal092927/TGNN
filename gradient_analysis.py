"""
Comprehensive Gradient and Weight Analysis for TGAM Model

This script analyzes:
1. Weight initialization distribution
2. Gradient flow during training
3. Dead neurons
4. Gradient explosion/vanishing
5. Weight changes during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
import time

from tgam_fixed import TGAM_LinkPrediction
from utils import RandEdgeSampler

class GradientAnalyzer:
    """Analyzes gradients and weights during training"""
    
    def __init__(self, model):
        self.model = model
        self.weight_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        self.activation_history = defaultdict(list)
        
    def analyze_initialization(self):
        """Analyze weight initialization"""
        print("=== Weight Initialization Analysis ===")
        
        init_stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) > 1:  # Skip biases
                weight = param.data.cpu().numpy()
                stats = {
                    'mean': np.mean(weight),
                    'std': np.std(weight),
                    'min': np.min(weight),
                    'max': np.max(weight),
                    'shape': weight.shape
                }
                init_stats[name] = stats
                print(f"{name:50s}: mean={stats['mean']:8.6f}, std={stats['std']:8.6f}, "
                      f"range=[{stats['min']:8.6f}, {stats['max']:8.6f}], shape={stats['shape']}")
        
        return init_stats
    
    def register_hooks(self):
        """Register hooks to monitor activations"""
        self.activation_hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    output_np = output.detach().cpu().numpy()
                    self.activation_history[name].append({
                        'mean': np.mean(output_np),
                        'std': np.std(output_np),
                        'max': np.max(output_np),
                        'min': np.min(output_np),
                        'zero_fraction': np.mean(output_np == 0.0)
                    })
            return hook
        
        # Register hooks for key components
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.Sigmoid)):
                hook = module.register_forward_hook(make_hook(name))
                self.activation_hooks.append(hook)
    
    def remove_hooks(self):
        """Remove activation hooks"""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
    
    def analyze_gradients(self, loss=None):
        """Analyze current gradients"""
        gradient_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data.cpu().numpy()
                stats = {
                    'norm': np.linalg.norm(grad),
                    'mean': np.mean(grad),
                    'std': np.std(grad),
                    'max': np.max(grad),
                    'min': np.min(grad),
                    'zero_fraction': np.mean(grad == 0.0),
                    'nan_fraction': np.mean(np.isnan(grad)),
                    'inf_fraction': np.mean(np.isinf(grad))
                }
                gradient_stats[name] = stats
                self.gradient_history[name].append(stats)
        
        return gradient_stats
    
    def analyze_weight_changes(self, initial_weights):
        """Analyze how weights have changed from initialization"""
        weight_changes = {}
        
        for name, param in self.model.named_parameters():
            if name in initial_weights and param.requires_grad:
                current = param.data.cpu().numpy()
                initial = initial_weights[name]
                
                change = current - initial
                relative_change = np.abs(change) / (np.abs(initial) + 1e-8)
                
                stats = {
                    'abs_change_norm': np.linalg.norm(change),
                    'rel_change_mean': np.mean(relative_change),
                    'rel_change_max': np.max(relative_change),
                    'current_norm': np.linalg.norm(current),
                    'initial_norm': np.linalg.norm(initial)
                }
                weight_changes[name] = stats
        
        return weight_changes
    
    def print_gradient_summary(self, gradient_stats, step=None):
        """Print summary of gradient statistics"""
        step_str = f"Step {step}" if step is not None else "Current"
        print(f"\n=== Gradient Analysis - {step_str} ===")
        
        if not gradient_stats:
            print("❌ No gradients found!")
            return
            
        print(f"{'Layer':50s} {'Norm':12s} {'Mean':12s} {'Std':12s} {'Dead%':8s} {'NaN%':8s}")
        print("-" * 100)
        
        for name, stats in gradient_stats.items():
            print(f"{name:50s} {stats['norm']:12.6e} {stats['mean']:12.6e} "
                  f"{stats['std']:12.6e} {stats['zero_fraction']*100:7.1f}% "
                  f"{stats['nan_fraction']*100:7.1f}%")
    
    def detect_issues(self, gradient_stats):
        """Detect potential training issues"""
        issues = []
        
        # Check for vanishing gradients
        very_small_grads = [name for name, stats in gradient_stats.items() 
                           if stats['norm'] < 1e-8]
        if very_small_grads:
            issues.append(f"⚠️  Very small gradients (< 1e-8): {very_small_grads}")
        
        # Check for exploding gradients
        very_large_grads = [name for name, stats in gradient_stats.items() 
                           if stats['norm'] > 10]
        if very_large_grads:
            issues.append(f"⚠️  Large gradients (> 10): {very_large_grads}")
        
        # Check for dead neurons (high zero fraction)
        dead_layers = [name for name, stats in gradient_stats.items() 
                      if stats['zero_fraction'] > 0.5]
        if dead_layers:
            issues.append(f"⚠️  Potentially dead layers (>50% zero grads): {dead_layers}")
        
        # Check for NaN/Inf gradients
        nan_layers = [name for name, stats in gradient_stats.items() 
                     if stats['nan_fraction'] > 0 or stats['inf_fraction'] > 0]
        if nan_layers:
            issues.append(f"❌ NaN/Inf gradients: {nan_layers}")
        
        return issues


def test_model_with_gradient_analysis():
    """Test TGAM model with comprehensive gradient analysis"""
    
    # Load data
    print("Loading triadic_fixed dataset...")
    g_df = pd.read_csv('./processed/triadic_fixed/ml_triadic_fixed.csv')
    print(f"Dataset: {len(g_df)} edges, {g_df.u.nunique()} nodes, ts range: {g_df.ts.min():.1f}-{g_df.ts.max():.1f}")
    
    # Load features
    n_feat = np.load('./processed/triadic_fixed/ml_triadic_fixed_node.npy')
    e_feat = np.load('./processed/triadic_fixed/ml_triadic_fixed.npy')
    print(f"Node features: {n_feat.shape}, Edge features: {e_feat.shape}")
    
    # Simple train/test split for focused analysis
    train_ratio = 0.8
    n_train = int(len(g_df) * train_ratio)
    train_data = g_df.iloc[:n_train]
    
    src_l = train_data.u.values
    dst_l = train_data.i.values
    ts_l = train_data.ts.values
    e_idx_l = train_data.idx.values
    
    # Initialize model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TGAM_LinkPrediction(
        node_feat_dim=n_feat.shape[1],
        edge_feat_dim=e_feat.shape[1],
        hidden_dim=128,
        max_nodes=100,
        num_graph_layers=2,
        num_temporal_layers=4
    ).to(device)
    
    # Initialize analyzer
    analyzer = GradientAnalyzer(model)
    
    # Analyze initialization
    init_stats = analyzer.analyze_initialization()
    
    # Store initial weights
    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.data.cpu().numpy().copy()
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    sampler = RandEdgeSampler(src_l, dst_l)
    
    # Convert data to tensors
    n_feat_tensor = torch.tensor(n_feat, dtype=torch.float32).to(device)
    e_feat_tensor = torch.tensor(e_feat, dtype=torch.float32).to(device)
    
    print(f"\n=== Training with Gradient Analysis ===")
    
    # Register hooks
    analyzer.register_hooks()
    
    # Training loop with detailed monitoring
    num_steps = 10  # Small number for detailed analysis
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Simple training step
        step_loss = 0.0
        batch_size = 5  # Small batch for analysis
        
        for i in range(batch_size):
            # Random training example
            k = np.random.randint(1, len(src_l))
            
            # Positive example
            pos_prob = model(
                src_l[:k+1], dst_l[:k+1], dst_l[k],
                ts_l[:k+1], e_idx_l[:k+1],
                n_feat_tensor, e_feat_tensor
            )
            
            # Negative example
            u_fake, i_fake = sampler.sample(1)
            neg_prob = model(
                src_l[:k+1], dst_l[:k+1], i_fake[0],
                ts_l[:k+1], e_idx_l[:k+1],
                n_feat_tensor, e_feat_tensor
            )
            
            # Loss
            pos_loss = criterion(pos_prob.squeeze(), torch.tensor(1.0, device=device))
            neg_loss = criterion(neg_prob.squeeze(), torch.tensor(0.0, device=device))
            loss = pos_loss + neg_loss
            step_loss += loss.item()
            
            loss.backward()
        
        # Analyze gradients before optimizer step
        gradient_stats = analyzer.analyze_gradients()
        
        # Print detailed analysis every few steps
        if step % 2 == 0:
            analyzer.print_gradient_summary(gradient_stats, step)
            
            # Check for issues
            issues = analyzer.detect_issues(gradient_stats)
            for issue in issues:
                print(issue)
        
        # Optimizer step
        optimizer.step()
        
        # Analyze weight changes
        if step % 2 == 0:
            weight_changes = analyzer.analyze_weight_changes(initial_weights)
            print(f"\n=== Weight Changes - Step {step} ===")
            for name, stats in weight_changes.items():
                if 'weight' in name:  # Focus on weight matrices
                    print(f"{name:50s}: rel_change={stats['rel_change_mean']:.6e}, "
                          f"norm_ratio={stats['current_norm']/stats['initial_norm']:.6f}")
        
        print(f"Step {step+1}: Loss = {step_loss:.6f}")
        
        # Check if model is learning at all
        if step > 0 and step % 5 == 0:
            print("\n=== Quick Learning Check ===")
            # Test on same input twice to see if outputs change
            test_k = min(5, len(src_l)-1)
            prob1 = model(src_l[:test_k+1], dst_l[:test_k+1], dst_l[test_k],
                         ts_l[:test_k+1], e_idx_l[:test_k+1], n_feat_tensor, e_feat_tensor)
            
            prob2 = model(src_l[:test_k+1], dst_l[:test_k+1], dst_l[test_k],
                         ts_l[:test_k+1], e_idx_l[:test_k+1], n_feat_tensor, e_feat_tensor)
            
            print(f"Model consistency check: prob1={prob1.item():.8f}, prob2={prob2.item():.8f}")
            print(f"Difference: {abs(prob1.item() - prob2.item()):.8e}")
    
    # Remove hooks
    analyzer.remove_hooks()
    
    # Final analysis
    print(f"\n=== Final Analysis ===")
    
    # Check final weight changes
    final_weight_changes = analyzer.analyze_weight_changes(initial_weights)
    print("\nFinal weight changes:")
    significant_changes = 0
    for name, stats in final_weight_changes.items():
        if 'weight' in name:
            if stats['rel_change_mean'] > 1e-6:
                significant_changes += 1
                print(f"✓ {name}: {stats['rel_change_mean']:.6e} relative change")
            else:
                print(f"✗ {name}: {stats['rel_change_mean']:.6e} relative change (too small)")
    
    if significant_changes == 0:
        print("❌ NO SIGNIFICANT WEIGHT CHANGES DETECTED!")
    else:
        print(f"✓ {significant_changes} layers showing significant changes")
    
    # Test model output variation
    print("\n=== Output Variation Test ===")
    test_outputs = []
    for _ in range(10):
        # Slightly different inputs to test sensitivity
        k = np.random.randint(3, min(10, len(src_l)))
        prob = model(src_l[:k+1], dst_l[:k+1], dst_l[k],
                    ts_l[:k+1], e_idx_l[:k+1], n_feat_tensor, e_feat_tensor)
        test_outputs.append(prob.item())
    
    output_std = np.std(test_outputs)
    output_range = max(test_outputs) - min(test_outputs)
    print(f"Output variation: std={output_std:.8f}, range={output_range:.8f}")
    print(f"Outputs: {test_outputs[:5]} ...")
    
    if output_std < 1e-6:
        print("❌ Model outputs are nearly identical - model is not sensitive to inputs!")
    else:
        print(f"✓ Model shows reasonable output variation")


if __name__ == "__main__":
    test_model_with_gradient_analysis() 