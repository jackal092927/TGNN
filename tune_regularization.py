#!/usr/bin/env python3
"""
Regularization Tuning for TGAM Model

This script systematically tests different regularization strategies to address
the overfitting issue (train/test performance gap) identified in the debugging.
"""

import torch
import numpy as np
import pandas as pd
from itertools import product
from train_tgam_proper_eval import train_tgam_proper_eval
import json
import time

def tune_regularization(dataset='triadic_large', base_config=None):
    """
    Systematically tune regularization parameters
    
    Args:
        dataset: Name of the dataset to use
        base_config: Base configuration to start from
    """
    
    print("🔧 TGAM Regularization Tuning")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    
    # Base configuration
    if base_config is None:
        base_config = {
            'training_mode': 'autoregressive',
            'dataset': dataset,
            'num_epochs': 12,  # Reduced for tuning speed
            'lr': 0.001,
            'steps_per_epoch': 30,  # Reduced for tuning speed
            'hidden_dim': 128,
            'max_nodes': 250 if dataset == 'triadic_large' else 600,
            'num_graph_layers': 2,
            'num_temporal_layers': 4,
            'max_timestamps': 15,  # Reduced for speed
            'use_early_stopping': False,
            'train_ratio': 0.6,
            'val_ratio': 0.2,
            'test_ratio': 0.2,
            'gpu_id': 0
        }
    
    # Regularization parameter grid
    regularization_grid = {
        'dropout_rate': [0.0, 0.1, 0.2, 0.3],  # Dropout in model layers
        'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],  # L2 regularization
        'input_dropout': [0.0, 0.05, 0.1],  # Input feature dropout
    }
    
    print(f"\nRegularization Grid:")
    for param, values in regularization_grid.items():
        print(f"  {param}: {values}")
    
    # Generate all combinations
    param_combinations = list(product(*regularization_grid.values()))
    param_names = list(regularization_grid.keys())
    
    print(f"\nTesting {len(param_combinations)} parameter combinations...")
    
    results = []
    best_config = None
    best_generalization = float('inf')  # Lower is better (train_ap - test_ap)
    
    for i, param_values in enumerate(param_combinations):
        print(f"\n" + "=" * 60)
        print(f"🧪 Combination {i+1}/{len(param_combinations)}")
        
        # Create config for this combination
        config = base_config.copy()
        
        # Add regularization parameters
        for param_name, param_value in zip(param_names, param_values):
            config[param_name] = param_value
        
        print(f"Parameters: {dict(zip(param_names, param_values))}")
        
        try:
            start_time = time.time()
            
            # Train model with this configuration
            result = train_tgam_proper_eval(config)
            
            training_time = time.time() - start_time
            
            # Calculate key metrics
            train_ap = result['train_accuracies'][-1] if result['train_accuracies'] else 0.0
            test_ap = result['test_ap']
            val_ap = result['best_val_ap']
            
            # Generalization gap (lower is better)
            generalization_gap = abs(train_ap - test_ap)
            
            # Store result
            result_record = {
                'combination': i + 1,
                'parameters': dict(zip(param_names, param_values)),
                'train_accuracy': train_ap,
                'test_ap': test_ap,
                'val_ap': val_ap,
                'test_accuracy': result['test_accuracy'],
                'test_auc': result['test_auc'],
                'generalization_gap': generalization_gap,
                'training_time': training_time,
                'config': config
            }
            
            results.append(result_record)
            
            print(f"Results:")
            print(f"  Train Accuracy: {train_ap:.4f}")
            print(f"  Test AP:        {test_ap:.4f}")
            print(f"  Test Accuracy:  {result['test_accuracy']:.4f}")
            print(f"  Val AP:         {val_ap:.4f}")
            print(f"  Gen Gap:        {generalization_gap:.4f}")
            print(f"  Time:           {training_time:.1f}s")
            
            # Check if this is the best generalization
            if generalization_gap < best_generalization and test_ap > 0.4:  # Minimum performance threshold
                best_generalization = generalization_gap
                best_config = result_record
                print(f"  🌟 NEW BEST GENERALIZATION!")
            
        except Exception as e:
            print(f"❌ Error in combination {i+1}: {e}")
            continue
    
    # Analysis and recommendations
    print(f"\n" + "=" * 60)
    print("📊 REGULARIZATION TUNING RESULTS")
    print("=" * 60)
    
    if not results:
        print("❌ No successful runs!")
        return None
    
    # Sort by generalization gap
    results.sort(key=lambda x: x['generalization_gap'])
    
    print(f"\n🏆 Top 5 Configurations (by generalization):")
    for i, result in enumerate(results[:5]):
        params = result['parameters']
        print(f"{i+1}. Gap={result['generalization_gap']:.4f}, "
              f"TestAP={result['test_ap']:.4f}, "
              f"Params={params}")
    
    # Sort by test performance
    results.sort(key=lambda x: x['test_ap'], reverse=True)
    
    print(f"\n🎯 Top 5 Configurations (by test performance):")
    for i, result in enumerate(results[:5]):
        params = result['parameters']
        print(f"{i+1}. TestAP={result['test_ap']:.4f}, "
              f"Gap={result['generalization_gap']:.4f}, "
              f"Params={params}")
    
    # Parameter impact analysis
    print(f"\n📈 Parameter Impact Analysis:")
    
    # Analyze each parameter's effect
    for param_name in param_names:
        param_effects = {}
        for result in results:
            param_val = result['parameters'][param_name]
            if param_val not in param_effects:
                param_effects[param_val] = []
            param_effects[param_val].append(result['generalization_gap'])
        
        print(f"\n  {param_name}:")
        for param_val, gaps in param_effects.items():
            avg_gap = np.mean(gaps)
            print(f"    {param_val}: avg_gap={avg_gap:.4f} (n={len(gaps)})")
    
    # Save results
    output_file = f"regularization_tuning_{dataset}_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        # Convert configs to serializable format
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            serializable_result['config'] = {k: v for k, v in result['config'].items() 
                                           if isinstance(v, (int, float, str, bool, type(None)))}
            serializable_results.append(serializable_result)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if best_config:
        best_params = best_config['parameters']
        print(f"  Best regularization: {best_params}")
        print(f"  Generalization gap: {best_config['generalization_gap']:.4f}")
        print(f"  Test performance: {best_config['test_ap']:.4f}")
        
        # Specific recommendations
        if best_params['dropout_rate'] > 0.1:
            print(f"  ✅ Use dropout_rate={best_params['dropout_rate']} (significant)")
        if best_params['weight_decay'] > 1e-5:
            print(f"  ✅ Use weight_decay={best_params['weight_decay']} (L2 regularization)")
        if best_params['input_dropout'] > 0.0:
            print(f"  ✅ Use input_dropout={best_params['input_dropout']} (input noise)")
            
    else:
        print(f"  ⚠️  No configuration achieved good generalization")
        print(f"  Consider: stronger regularization, different architectures, or more data")
    
    return results, best_config

def quick_regularization_test(dataset='triadic_large'):
    """Quick test of a few promising regularization combinations"""
    
    print("⚡ Quick Regularization Test")
    print("=" * 40)
    
    # Test a few promising combinations based on common best practices
    test_configs = [
        {'dropout_rate': 0.0, 'weight_decay': 0.0, 'input_dropout': 0.0},  # Baseline
        {'dropout_rate': 0.2, 'weight_decay': 1e-4, 'input_dropout': 0.05},  # Moderate reg
        {'dropout_rate': 0.3, 'weight_decay': 1e-3, 'input_dropout': 0.1},   # Strong reg
    ]
    
    results = []
    
    for i, reg_params in enumerate(test_configs):
        print(f"\n🧪 Testing configuration {i+1}: {reg_params}")
        
        config = {
            'training_mode': 'autoregressive',
            'dataset': dataset,
            'num_epochs': 8,  # Quick test
            'lr': 0.001,
            'steps_per_epoch': 20,
            'hidden_dim': 128,
            'max_nodes': 250 if dataset == 'triadic_large' else 600,
            'num_graph_layers': 2,
            'num_temporal_layers': 4,
            'max_timestamps': 10,
            'use_early_stopping': False,
            'train_ratio': 0.6,
            'val_ratio': 0.2,
            'test_ratio': 0.2,
            'gpu_id': 0,
            **reg_params
        }
        
        try:
            result = train_tgam_proper_eval(config)
            
            train_acc = result['train_accuracies'][-1] if result['train_accuracies'] else 0.0
            test_ap = result['test_ap']
            gap = abs(train_acc - test_ap)
            
            print(f"  Train Acc: {train_acc:.4f}")
            print(f"  Test AP:   {test_ap:.4f}")
            print(f"  Gap:       {gap:.4f}")
            
            results.append({
                'config': reg_params,
                'train_acc': train_acc,
                'test_ap': test_ap,
                'gap': gap
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Find best
    if results:
        best = min(results, key=lambda x: x['gap'])
        print(f"\n🏆 Best quick config: {best['config']}")
        print(f"   Gap: {best['gap']:.4f}, Test AP: {best['test_ap']:.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune regularization for TGAM')
    parser.add_argument('--dataset', default='triadic_large', 
                       help='Dataset to use for tuning')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test instead of full grid search')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_regularization_test(args.dataset)
    else:
        tune_regularization(args.dataset) 