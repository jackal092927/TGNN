"""
Quick test script for TGAM model to verify it works correctly.

This script runs a basic test on a small dataset to ensure all components work.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from tgam import TGAM, compute_loss
from train_tgam import prepare_sequences, evaluate_link_prediction


def create_toy_dataset():
    """Create a small toy dataset for testing"""
    
    # Create a simple temporal graph with 10 nodes, 20 edges
    num_nodes = 10
    num_edges = 20
    node_feat_dim = 5
    edge_feat_dim = 3
    
    # Generate random edges
    src_nodes = np.random.randint(0, num_nodes, num_edges)
    dst_nodes = np.random.randint(0, num_nodes, num_edges)
    timestamps = np.sort(np.random.uniform(0, 100, num_edges))
    edge_indices = np.arange(num_edges)
    
    # Create DataFrame
    df = pd.DataFrame({
        'u': src_nodes,
        'i': dst_nodes,
        'ts': timestamps,
        'idx': edge_indices,
        'label': np.ones(num_edges)  # All positive edges
    })
    
    # Create node features (random)
    node_features = np.random.randn(num_nodes, node_feat_dim)
    
    # Create edge features (random)
    edge_features = np.random.randn(num_edges, edge_feat_dim)
    
    return df, node_features, edge_features


def test_basic_functionality():
    """Test basic TGAM functionality"""
    
    print("Testing basic TGAM functionality...")
    
    # Create toy data
    df, node_features, edge_features = create_toy_dataset()
    
    print(f"Created toy dataset: {len(df)} edges, {len(node_features)} nodes")
    
    # Test sequence preparation
    sequences = prepare_sequences(
        df.u.values, df.i.values, df.ts.values, df.idx.values,
        node_features, edge_features, sequence_length=5, step_size=3
    )
    
    print(f"Created {len(sequences)} sequences")
    
    if len(sequences) == 0:
        print("ERROR: No sequences created!")
        return False
    
    # Test model initialization
    max_nodes = max(df.u.max(), df.i.max()) + 1
    
    model = TGAM(
        node_feat_dim=node_features.shape[1],
        edge_feat_dim=edge_features.shape[1],
        hidden_dim=32,
        max_nodes=max_nodes,
        num_graph_layers=2,
        num_temporal_layers=2
    )
    
    print(f"Initialized TGAM model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    test_sequence = sequences[0]
    
    try:
        src_logits, dst_logits = model(test_sequence, test_sequence[-1][3])
        
        if src_logits is not None:
            print(f"Forward pass successful:")
            print(f"  Source logits shape: {src_logits.shape}")
            print(f"  Destination logits shape: {dst_logits.shape}")
        else:
            print("ERROR: Forward pass returned None!")
            return False
            
    except Exception as e:
        print(f"ERROR in forward pass: {e}")
        return False
    
    # Test loss computation
    try:
        criterion = torch.nn.CrossEntropyLoss()
        loss = compute_loss(model, [test_sequence], criterion)
        
        print(f"Loss computation successful: {loss.item():.4f}")
        
    except Exception as e:
        print(f"ERROR in loss computation: {e}")
        return False
    
    # Test generation
    try:
        initial_graph = test_sequence[0]
        generated_edges, sequence = model.generate_sequence(
            initial_graph, start_time=100, num_steps=3, time_step=1.0
        )
        
        print(f"Generation successful: created {len(generated_edges)} edges")
        
    except Exception as e:
        print(f"ERROR in generation: {e}")
        return False
    
    print("All basic functionality tests passed!")
    return True


def test_training_step():
    """Test a single training step"""
    
    print("\nTesting training step...")
    
    # Create toy data
    df, node_features, edge_features = create_toy_dataset()
    
    # Prepare sequences
    sequences = prepare_sequences(
        df.u.values, df.i.values, df.ts.values, df.idx.values,
        node_features, edge_features, sequence_length=5, step_size=3
    )
    
    if len(sequences) < 2:
        print("ERROR: Need at least 2 sequences for training test")
        return False
    
    # Initialize model
    max_nodes = max(df.u.max(), df.i.max()) + 1
    
    model = TGAM(
        node_feat_dim=node_features.shape[1],
        edge_feat_dim=edge_features.shape[1],
        hidden_dim=32,
        max_nodes=max_nodes,
        num_graph_layers=2,
        num_temporal_layers=2
    )
    
    # Setup training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training step
    try:
        model.train()
        optimizer.zero_grad()
        
        # Compute loss for batch
        batch_sequences = sequences[:2]
        loss = compute_loss(model, batch_sequences, criterion)
        
        if loss > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            print(f"Training step successful: loss = {loss.item():.4f}")
        else:
            print("ERROR: Loss is zero or negative")
            return False
            
    except Exception as e:
        print(f"ERROR in training step: {e}")
        return False
    
    print("Training step test passed!")
    return True


def test_evaluation():
    """Test evaluation functionality"""
    
    print("\nTesting evaluation...")
    
    # Create toy data
    df, node_features, edge_features = create_toy_dataset()
    
    # Prepare sequences
    sequences = prepare_sequences(
        df.u.values, df.i.values, df.ts.values, df.idx.values,
        node_features, edge_features, sequence_length=5, step_size=3
    )
    
    if len(sequences) < 3:
        print("ERROR: Need at least 3 sequences for evaluation test")
        return False
    
    # Initialize model
    max_nodes = max(df.u.max(), df.i.max()) + 1
    
    model = TGAM(
        node_feat_dim=node_features.shape[1],
        edge_feat_dim=edge_features.shape[1],
        hidden_dim=32,
        max_nodes=max_nodes,
        num_graph_layers=2,
        num_temporal_layers=2
    )
    
    # Test evaluation
    try:
        device = torch.device('cpu')  # Use CPU for testing
        
        acc, auc, ap, f1 = evaluate_link_prediction(
            model, sequences, device, num_samples=3
        )
        
        print(f"Evaluation successful:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
    except Exception as e:
        print(f"ERROR in evaluation: {e}")
        return False
    
    print("Evaluation test passed!")
    return True


def main():
    """Run all tests"""
    
    print("=" * 50)
    print("TGAM Model Testing")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_basic_functionality,
        test_training_step,
        test_evaluation
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"ERROR in {test.__name__}: {e}")
    
    print("\n" + "=" * 50)
    print(f"TESTING COMPLETE: {passed}/{len(tests)} tests passed")
    print("=" * 50)
    
    if passed == len(tests):
        print("All tests passed! TGAM is ready for use.")
        return True
    else:
        print("Some tests failed! Please check the implementation.")
        return False


if __name__ == '__main__':
    main() 