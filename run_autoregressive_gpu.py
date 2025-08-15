#!/usr/bin/env python3
"""
Quick example showing how to run TGAM with specific GPU selection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_tgam_proper_eval import train_tgam_proper_eval

def run_on_gpu(gpu_id):
    """Run TRUE TGAM autoregressive training with sequential graph building on specified GPU"""
    
    print(f"🚀 Running TRUE TGAM Autoregressive (Sequential Graph Building) on GPU {gpu_id}")
    print("=" * 70)
    
    config = {
        'dataset': 'triadic_fixed',
        'training_mode': 'autoregressive',
        'hidden_dim': 128,
        'epochs': 20,
        'steps_per_epoch': 20,
        'lr': 0.001,
        'max_timestamps': None,
        'max_candidates_per_timestamp': 30,
        'use_adaptive_weights': True,  # Auto-calculate optimal weights from data
        'gpu_id': gpu_id
    }
    
    print(f"Configuration:")
    print(f"  Mode: {config['training_mode']}")
    print(f"  GPU: {gpu_id}")
    print(f"  Adaptive weights: {config['use_adaptive_weights']}")
    print(f"  Max timestamps: {config['max_timestamps']}")
    print(f"  Max candidates per timestamp: {config['max_candidates_per_timestamp']}")
    print()
    
    try:
        result = train_tgam_proper_eval(config)
        
        if result:
            print(f"\n✅ Training completed successfully!")
            print(f"   Final Test Accuracy: {result.get('final_test_acc', 0):.4f}")
            print(f"   Final Test AP:       {result.get('final_test_ap', 0):.4f}")
            print(f"   Final Test AUC:      {result.get('final_test_auc', 0):.4f}")
            
            return result
        else:
            print(f"❌ Training failed")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TGAM on specific GPU')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU ID to use (0 or 1, default: 0)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different GPU configurations')
    
    args = parser.parse_args()
    
    if args.compare:
        print("🔬 Comparing GPU configurations")
        print("=" * 50)
        
        # Test on both GPUs if available
        results = {}
        for gpu_id in [0, 1]:
            print(f"\nTesting GPU {gpu_id}:")
            result = run_on_gpu(gpu_id)
            results[f'GPU {gpu_id}'] = result
        
        # Show comparison
        print("\n📊 GPU Comparison:")
        print("-" * 30)
        for gpu_name, result in results.items():
            if result:
                acc = result.get('final_test_acc', 0)
                ap = result.get('final_test_ap', 0)
                print(f"{gpu_name}: Acc={acc:.4f}, AP={ap:.4f}")
            else:
                print(f"{gpu_name}: FAILED")
    else:
        # Run on specified GPU
        run_on_gpu(args.gpu) 