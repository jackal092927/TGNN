#!/usr/bin/env python3
"""
Test Proper Multi-Step Autoregressive
=====================================

This script tests the corrected multi-step approach:
- Training (60%): Multi-step + teacher forcing
- Validation (20%): Multi-step + autoregressive (no teacher forcing)  
- Testing (20%): Multi-step + autoregressive (no teacher forcing)

All phases use consistent sequential graph building.
Validation tests true autoregressive capability for model selection.
"""

import sys
import argparse
sys.path.append('.')

from train_tgam_proper_eval import train_tgam_proper_eval

def main():
    parser = argparse.ArgumentParser(description='Test Proper Multi-Step TGAM')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (default: 0)')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs (default: 8)')
    parser.add_argument('--max_timestamps', type=int, default=6, help='Max timestamps to process (default: 6)')
    args = parser.parse_args()
    
    print("🔄 Testing Proper Multi-Step Autoregressive")
    print("=" * 50)
    print("Training (60%):   Multi-step + teacher forcing")
    print("Validation (20%): Multi-step + autoregressive")  
    print("Testing (20%):    Multi-step + autoregressive")
    print("=" * 50)
    
    config = {
        'dataset': 'triadic_fixed',
        'training_mode': 'autoregressive',
        'gpu_id': args.gpu,
        'num_epochs': args.epochs,
        'max_timestamps': args.max_timestamps,
        'max_candidates_per_timestamp': 30,
        'use_adaptive_weights': True,
        'learning_rate': 0.001,
        'hidden_dim': 128,
        'steps_per_epoch': 15,
        'train_ratio': 0.6,   # 60% for training
        'val_ratio': 0.2,     # 20% for validation  
        'test_ratio': 0.2,    # 20% for testing
        'early_stopping_patience': 5
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        results = train_tgam_proper_eval(config)
        
        print(f"\n✅ Multi-step training completed!")
        print(f"📊 Results Summary:")
        print(f"   Validation (autoregressive): Acc={results.get('final_val_acc', 0):.4f}, AP={results['best_val_ap']:.4f}")
        print(f"   Testing (autoregressive):    Acc={results['test_accuracy']:.4f}, AP={results['test_ap']:.4f}")
        print(f"")
        print(f"🎯 Key insight: Validation tests true autoregressive capability")
        print(f"   to select the best autoregressive model!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 