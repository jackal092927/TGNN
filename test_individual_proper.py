#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_tgam_proper_eval import train_tgam_proper_eval

def test_autoregressive_proper(gpu_id=0):
    """Test the proper autoregressive implementation with class imbalance handling"""
    
    print("🚀 Testing PROPER Autoregressive TGAM")
    print(f"Using GPU: {gpu_id}")
    print("=" * 60)
    
    # Test different configurations to show the effect of weighting
    test_configs = [
        {
            'name': 'Individual Mode (Baseline)',
            'config': {
                'dataset': 'triadic_fixed',
                'training_mode': 'individual',
                'hidden_dim': 64,
                'epochs': 15,
                'steps_per_epoch': 30,
                'lr': 0.001,
                'gpu_id': gpu_id
            }
        },
        {
            'name': 'TRUE Autoregressive - Sequential Graph Building',
            'config': {
                'dataset': 'triadic_fixed',
                'training_mode': 'autoregressive',
                'hidden_dim': 64,
                'epochs': 15,
                'steps_per_epoch': 20,  # Fewer steps since autoregressive is more expensive
                'lr': 0.001,
                'max_timestamps': 6,  # Limit for computational efficiency
                'max_candidates_per_timestamp': 40,
                'use_adaptive_weights': True,  # Auto-calculate from data
                'gpu_id': gpu_id
            }
        },
        {
            'name': 'Autoregressive - Fixed Weights (3:1 ratio)',
            'config': {
                'dataset': 'triadic_fixed',
                'training_mode': 'autoregressive',
                'hidden_dim': 64,
                'epochs': 15,
                'steps_per_epoch': 20,
                'lr': 0.001,
                'max_timestamps': 8,
                'max_candidates_per_timestamp': 40,
                'use_adaptive_weights': False,  # Use fixed weights
                'pos_weight': 3.0,  # Fixed weight for positive examples
                'neg_weight': 1.0,  # Fixed weight for negative examples
                'gpu_id': gpu_id
            }
        },
        {
            'name': 'Autoregressive - Strong Positive Bias (5:1 ratio)',
            'config': {
                'dataset': 'triadic_fixed',
                'training_mode': 'autoregressive',
                'hidden_dim': 64,
                'epochs': 15,
                'steps_per_epoch': 20,
                'lr': 0.001,
                'max_timestamps': 8,
                'max_candidates_per_timestamp': 40,
                'pos_weight': 5.0,  # Very high weight for positive examples
                'neg_weight': 1.0,
                'gpu_id': gpu_id
            }
        },
        {
            'name': 'Hybrid Mode - Balanced (30% Auto + 70% Individual)',
            'config': {
                'dataset': 'triadic_fixed',
                'training_mode': 'hybrid',
                'hidden_dim': 64,
                'epochs': 15,
                'steps_per_epoch': 25,
                'lr': 0.001,
                'max_timestamps': 6,
                'max_candidates_per_timestamp': 30,
                'pos_weight': 3.0,
                'neg_weight': 1.0,
                'gpu_id': gpu_id
            }
        }
    ]
    
    results = {}
    
    for test in test_configs:
        print(f"\n🧪 Testing: {test['name']}")
        print("-" * 50)
        
        try:
            result = train_tgam_proper_eval(test['config'])
            results[test['name']] = result
            
            # Print key metrics
            if result:
                train_acc = result.get('final_train_acc', 0)
                val_acc = result.get('final_val_acc', 0)
                test_acc = result.get('final_test_acc', 0)
                val_ap = result.get('final_val_ap', 0)
                val_auc = result.get('final_val_auc', 0)
                
                print(f"✅ Results:")
                print(f"   Train Acc: {train_acc:.4f}")
                print(f"   Val Acc:   {val_acc:.4f}")
                print(f"   Test Acc:  {test_acc:.4f}")
                print(f"   Val AP:    {val_ap:.4f}")
                print(f"   Val AUC:   {val_auc:.4f}")
                
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results[test['name']] = None
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("📊 SUMMARY COMPARISON")
    print("=" * 60)
    
    print(f"{'Configuration':<40} {'Val Acc':<10} {'Val AP':<10} {'Test Acc':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        if result:
            val_acc = result.get('final_val_acc', 0)
            val_ap = result.get('final_val_ap', 0) 
            test_acc = result.get('final_test_acc', 0)
            print(f"{name:<40} {val_acc:<10.4f} {val_ap:<10.4f} {test_acc:<10.4f}")
        else:
            print(f"{name:<40} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10}")
    
    # Analysis
    print("\n🔍 ANALYSIS:")
    print("-" * 30)
    
    best_auto = None
    best_auto_score = 0
    individual_score = 0
    
    for name, result in results.items():
        if result:
            score = result.get('final_val_ap', 0)
            if 'Individual' in name:
                individual_score = score
            elif 'Autoregressive' in name and score > best_auto_score:
                best_auto = name
                best_auto_score = score
    
    if best_auto and best_auto_score > individual_score:
        improvement = ((best_auto_score - individual_score) / individual_score) * 100
        print(f"✅ Best autoregressive config ({best_auto}) improves over individual by {improvement:.1f}%")
    elif best_auto:
        degradation = ((individual_score - best_auto_score) / individual_score) * 100
        print(f"⚠️  Autoregressive still lags individual by {degradation:.1f}% - need more tuning")
    else:
        print("❌ No successful autoregressive runs")
    
    print("\n💡 Key Insights:")
    print("   - TRUE autoregressive: Sequential graph building with teacher forcing")
    print("   - Training: Use ground truth edges to build next graph state (teacher forcing)")
    print("   - Testing: Use model predictions to build next graph state (multi-step)")
    print("   - ADAPTIVE weighting automatically calculates optimal pos/neg ratio from data")
    print("   - Each prediction depends on previous step's graph state (true sequential dependency)")
    print("   - Computational cost: O(timestamps × candidates) with sequential dependency")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test TGAM Autoregressive with GPU selection')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (0 or 1, default: 0)')
    args = parser.parse_args()
    
    test_autoregressive_proper(gpu_id=args.gpu) 