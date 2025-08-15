"""
Test Improved TGAM Model on Full Training Pipeline

This script runs a shorter training session to validate that the improved 
architecture fixes the trivial prediction behavior (0.33 ↔ 0.67 oscillation).
"""

import torch
import numpy as np
from train_tgam_proper_eval import train_tgam_proper_eval

def test_improved_training():
    """Test the improved TGAM with shorter training to validate fixes"""
    
    print("🧪 Testing Improved TGAM on Full Training Pipeline")
    print("=" * 60)
    
    # Configuration for a quick test (reduced epochs/steps for faster validation)
    config = {
        'training_mode': 'autoregressive',  # Use the mode that was showing trivial prediction
        'dataset': 'triadic_large',
        'num_epochs': 15,  # Slightly more epochs for the larger dataset
        'lr': 0.001,
        'steps_per_epoch': 50,  # More steps per epoch for larger dataset
        'hidden_dim': 128,
        'max_nodes': 250,  # Increased for larger dataset (200 nodes)
        'num_graph_layers': 2,
        'num_temporal_layers': 4,
        'max_timestamps': 20,  # Use full timeline
        'use_early_stopping': False,
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'gpu_id': 0
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("🚀 Starting Training with Improved TGAM...")
    print("=" * 60)
    
    # Run training
    results = train_tgam_proper_eval(config)
    
    print("\n" + "=" * 60)
    print("📊 RESULTS ANALYSIS")
    print("=" * 60)
    
    # Analyze training progression
    train_accs = results['train_accuracies']
    val_aps = results['val_aps']
    
    print(f"\n📈 Training Progression:")
    print(f"  Initial training accuracy: {train_accs[0]:.4f}")
    print(f"  Final training accuracy:   {train_accs[-1]:.4f}")
    print(f"  Accuracy improvement:      {train_accs[-1] - train_accs[0]:.4f}")
    
    # Check for trivial prediction pattern (oscillating between 0.33 and 0.67)
    print(f"\n🔍 Trivial Prediction Check:")
    trivial_pattern_count = 0
    for acc in train_accs:
        if abs(acc - 0.333) < 0.05 or abs(acc - 0.667) < 0.05:
            trivial_pattern_count += 1
    
    trivial_percentage = trivial_pattern_count / len(train_accs) * 100
    print(f"  Epochs with trivial predictions: {trivial_pattern_count}/{len(train_accs)} ({trivial_percentage:.1f}%)")
    
    if trivial_percentage > 50:
        print("  🔴 STILL SHOWING TRIVIAL PREDICTION BEHAVIOR")
    elif trivial_percentage > 20:
        print("  🟡 SOME TRIVIAL PREDICTION BEHAVIOR")
    else:
        print("  🟢 TRIVIAL PREDICTION BEHAVIOR FIXED!")
    
    # Stability check
    print(f"\n📊 Training Stability:")
    acc_variance = np.var(train_accs[-5:])  # Variance of last 5 epochs
    print(f"  Accuracy variance (last 5 epochs): {acc_variance:.6f}")
    
    if acc_variance < 0.001:
        print("  🟢 STABLE TRAINING - accuracy converged")
    elif acc_variance < 0.01:
        print("  🟡 MODERATELY STABLE TRAINING")
    else:
        print("  🔴 UNSTABLE TRAINING - high variance")
    
    # Final performance
    print(f"\n🎯 Final Performance:")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Test AP Score: {results['test_ap']:.4f}")
    print(f"  Test AUC:      {results['test_auc']:.4f}")
    print(f"  Best Val AP:   {results['best_val_ap']:.4f}")
    
    # Performance assessment
    if results['test_ap'] > 0.7:
        print("  🟢 GOOD PERFORMANCE")
    elif results['test_ap'] > 0.5:
        print("  🟡 MODERATE PERFORMANCE")
    else:
        print("  🔴 POOR PERFORMANCE")
    
    # Comparison to previous results
    print(f"\n📈 Improvement Summary:")
    print(f"  Model: Improved TGAM (0% dead neurons)")
    print(f"  Architecture: LeakyReLU + LayerNorm + Residuals")
    print(f"  Training mode: {config['training_mode']}")
    
    if trivial_percentage < 20 and acc_variance < 0.01 and results['test_ap'] > 0.5:
        print(f"\n🎉 SUCCESS: Improved architecture appears to have fixed the issues!")
        print(f"  ✅ Eliminated trivial prediction behavior")
        print(f"  ✅ Stable training progression") 
        print(f"  ✅ Reasonable final performance")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: Some issues may remain")
        if trivial_percentage >= 20:
            print(f"  ❌ Still showing trivial prediction behavior")
        if acc_variance >= 0.01:
            print(f"  ❌ Training still unstable")
        if results['test_ap'] <= 0.5:
            print(f"  ❌ Performance still poor")
    
    return results

if __name__ == "__main__":
    # Run the test
    results = test_improved_training()
    
    print(f"\n" + "=" * 60)
    print("🏁 TEST COMPLETE")
    print("=" * 60) 