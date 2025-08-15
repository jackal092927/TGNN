#!/usr/bin/env python3
"""
Deep Model Debugging Script
===========================

This script performs deep debugging of the TGAM model to identify
exactly why it's outputting constant values and not learning.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tgam_fixed import TGAM_LinkPrediction

def deep_debug_model():
    """Perform deep debugging of the TGAM model"""
    
    print("🔍 Deep Model Debugging")
    print("=" * 50)
    
    # Load minimal data
    data = 'triadic_fixed'
    g_df = pd.read_csv(f'./processed/{data}/ml_{data}.csv')
    e_feat = np.load(f'./processed/{data}/ml_{data}.npy')
    n_feat = np.load(f'./processed/{data}/ml_{data}_node.npy')
    
    print(f"Data loaded: {len(g_df)} edges, {len(n_feat)} nodes")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TGAM_LinkPrediction(
        node_feat_dim=n_feat.shape[1],
        edge_feat_dim=e_feat.shape[1],
        hidden_dim=32,  # Smaller for debugging
        max_nodes=n_feat.shape[0],
        num_graph_layers=2,
        num_temporal_layers=2
    ).to(device)
    
    # Test 1: Check if model parameters are trainable
    print("\n🧪 Test 1: Model Parameter Check")
    print("-" * 40)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
    
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    if trainable_params == 0:
        print("❌ CRITICAL: No trainable parameters!")
        return
    else:
        print("✅ Model has trainable parameters")
    
    # Test 2: Check input processing
    print("\n🧪 Test 2: Input Processing Check")
    print("-" * 40)
    
    # Create simple test inputs
    src_l = g_df.u.values[:10]
    dst_l = g_df.i.values[:10]
    ts_l = g_df.ts.values[:10]
    e_idx_l = g_df.idx.values[:10]
    candidate_dst = 5
    
    print(f"Input shapes:")
    print(f"  src_l: {src_l.shape}")
    print(f"  dst_l: {dst_l.shape}")
    print(f"  ts_l: {ts_l.shape}")
    print(f"  e_idx_l: {e_idx_l.shape}")
    print(f"  node_features: {n_feat.shape}")
    print(f"  edge_features: {e_feat.shape}")
    
    # Test 3: Forward pass debugging
    print("\n🧪 Test 3: Forward Pass Debugging")
    print("-" * 40)
    
    model.eval()
    
    # Hook to capture intermediate outputs
    intermediate_outputs = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                intermediate_outputs[name] = {
                    'shape': output.shape,
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item()
                }
        return hook
    
    # Register hooks
    model.graph_encoder.register_forward_hook(hook_fn('graph_encoder'))
    model.temporal_encoder.register_forward_hook(hook_fn('temporal_encoder'))
    model.link_predictor.register_forward_hook(hook_fn('link_predictor'))
    
    try:
        with torch.no_grad():
            output = model(src_l, dst_l, candidate_dst, ts_l, e_idx_l, n_feat, e_feat)
            
            print(f"Model output: {output.item():.6f}")
            print(f"Output type: {type(output)}")
            print(f"Output shape: {output.shape}")
            
            print("\nIntermediate outputs:")
            for name, stats in intermediate_outputs.items():
                print(f"  {name}: shape={stats['shape']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")
                
                # Check for problematic values
                if stats['std'] < 1e-6:
                    print(f"    ❌ {name} has very low std - outputs are nearly constant!")
                if abs(stats['mean']) < 1e-6:
                    print(f"    ❌ {name} has near-zero mean - might be dead neurons!")
                    
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Gradient flow check
    print("\n🧪 Test 4: Gradient Flow Check")
    print("-" * 40)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Create training example
    target = torch.tensor([1.0], dtype=torch.float32, device=device)
    
    try:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src_l, dst_l, candidate_dst, ts_l, e_idx_l, n_feat, e_feat)
        loss = criterion(output, target)
        
        print(f"Loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = False
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                gradient_stats[name] = grad_norm
                
                if grad_norm < 1e-8:
                    print(f"❌ {name}: gradient norm too small ({grad_norm:.2e})")
                elif grad_norm > 100:
                    print(f"❌ {name}: gradient norm too large ({grad_norm:.2e})")
                else:
                    print(f"✅ {name}: gradient norm OK ({grad_norm:.2e})")
        
        if not has_gradients:
            print("❌ CRITICAL: No gradients computed!")
        else:
            print("✅ Gradients are flowing")
            
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 5: Multiple forward passes with different inputs
    print("\n🧪 Test 5: Input Sensitivity Check")
    print("-" * 40)
    
    model.eval()
    
    test_cases = [
        (src_l, dst_l, 0, ts_l, e_idx_l),  # Different candidate_dst
        (src_l, dst_l, 1, ts_l, e_idx_l),
        (src_l, dst_l, 2, ts_l, e_idx_l),
        (src_l[:5], dst_l[:5], 0, ts_l[:5], e_idx_l[:5]),  # Different history length
        (src_l[:3], dst_l[:3], 0, ts_l[:3], e_idx_l[:3]),
    ]
    
    outputs = []
    with torch.no_grad():
        for i, (s, d, cand, t, e) in enumerate(test_cases):
            try:
                out = model(s, d, cand, t, e, n_feat, e_feat)
                outputs.append(out.item())
                print(f"Test case {i+1}: {out.item():.6f}")
            except Exception as ex:
                print(f"Test case {i+1}: FAILED - {ex}")
                outputs.append(0.0)
    
    # Check sensitivity
    if len(outputs) > 1:
        output_var = np.var(outputs)
        print(f"\nOutput variance: {output_var:.2e}")
        
        if output_var < 1e-6:
            print("❌ CRITICAL: Model is insensitive to input changes!")
        else:
            print("✅ Model responds to input changes")
    
    # Test 6: Component isolation
    print("\n🧪 Test 6: Component Isolation")
    print("-" * 40)
    
    print("Testing individual components...")
    
    # Test graph encoder alone
    try:
        edge_list = torch.stack([torch.tensor(src_l, dtype=torch.long), 
                                torch.tensor(dst_l, dtype=torch.long)], dim=1).to(device)
        seq_edge_features = torch.tensor(e_feat[e_idx_l], dtype=torch.float32).to(device)
        node_features_tensor = torch.tensor(n_feat, dtype=torch.float32).to(device)
        
        graph_emb = model.graph_encoder(node_features_tensor, edge_list, seq_edge_features, model.max_nodes)
        print(f"✅ Graph encoder works: shape={graph_emb.shape}, mean={graph_emb.mean().item():.6f}")
        
    except Exception as e:
        print(f"❌ Graph encoder failed: {e}")
    
    # Test temporal encoder alone
    try:
        ts_tensor = torch.tensor(ts_l, dtype=torch.float32).to(device)
        temp_emb = model.temporal_encoder(seq_edge_features, ts_tensor)
        print(f"✅ Temporal encoder works: shape={temp_emb.shape}, mean={temp_emb.mean().item():.6f}")
        
    except Exception as e:
        print(f"❌ Temporal encoder failed: {e}")
    
    print("\n🎯 Summary")
    print("-" * 40)
    print("Check the output above for:")
    print("1. Are parameters trainable?")
    print("2. Do gradients flow properly?")
    print("3. Does model respond to input changes?")
    print("4. Are intermediate outputs reasonable?")
    print("5. Are any components completely broken?")

if __name__ == '__main__':
    deep_debug_model() 