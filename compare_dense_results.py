"""
Compare results on the dense long perfect dataset
"""

def compare_dense_results():
    """
    Compare all model results on triadic_perfect_long_dense
    """
    print("=" * 80)
    print("RESULTS COMPARISON: triadic_perfect_long_dense")
    print("=" * 80)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Total edges: 1,458")
    print(f"  - Timeline: 28 timestamps (0-27)")
    print(f"  - Triadic closures: 1,328 (91.1%)")
    print(f"  - Nodes: 200")
    
    print(f"\n🏆 Model Performance:")
    
    results = [
        {
            'Model': 'Rule-based (Fixed)',
            'Test Acc': 60.07,
            'Test AUC': 100.00,
            'Test AP': 100.00,
            'Val Acc': 70.41,
            'Val AUC': 90.00,
            'Val AP': 90.00
        },
        {
            'Model': 'GraphRNN',
            'Test Acc': 51.82,
            'Test AUC': 51.78,
            'Test AP': 52.29,
            'Val Acc': 49.21,
            'Val AUC': 51.78,
            'Val AP': 50.86
        }
    ]
    
    print(f"{'Model':<20} {'Test Acc':<10} {'Test AUC':<10} {'Test AP':<10} {'Val AUC':<10} {'Val AP':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['Model']:<20} {r['Test Acc']:<10.1f} {r['Test AUC']:<10.1f} {r['Test AP']:<10.1f} {r['Val AUC']:<10.1f} {r['Val AP']:<10.1f}")
    
    print(f"\n🔍 Key Observations:")
    
    print(f"\n✅ Rule-based Method:")
    print(f"  - Perfect AUC/AP (100%): Confirms perfect triadic structure")
    print(f"  - Good accuracy (60%): Some confidence scores below 0.5 threshold")
    print(f"  - Validates dataset quality: All triadic opportunities correctly identified")
    
    print(f"\n❌ GraphRNN Struggling:")
    print(f"  - Random performance (~50%): Barely better than random guessing")
    print(f"  - Major degradation from previous datasets")
    print(f"  - Possible causes:")
    print(f"    1. Dataset too large/complex for current architecture")
    print(f"    2. Training epochs insufficient for convergence")
    print(f"    3. Learning rate or other hyperparameters suboptimal")
    print(f"    4. Dense graph structure overwhelming the model")
    
    print(f"\n🎯 Comparison with Previous Results:")
    
    previous_results = [
        {'Dataset': 'triadic_perfect_medium', 'GraphRNN Acc': 50.0, 'GraphRNN AUC': 75.0, 'GraphRNN AP': 77.1},
        {'Dataset': 'triadic_perfect_large', 'GraphRNN Acc': 58.3, 'GraphRNN AUC': 94.4, 'GraphRNN AP': 95.8},
        {'Dataset': 'triadic_perfect_long_dense', 'GraphRNN Acc': 51.8, 'GraphRNN AUC': 51.8, 'GraphRNN AP': 52.3}
    ]
    
    print(f"\n{'Dataset':<25} {'GraphRNN Acc':<15} {'GraphRNN AUC':<15} {'GraphRNN AP':<15}")
    print("-" * 70)
    
    for r in previous_results:
        print(f"{r['Dataset']:<25} {r['GraphRNN Acc']:<15.1f} {r['GraphRNN AUC']:<15.1f} {r['GraphRNN AP']:<15.1f}")
    
    print(f"\n💡 Analysis:")
    print(f"  - GraphRNN performed well on smaller perfect datasets (75-95% AUC)")
    print(f"  - Performance collapsed on large dense dataset (52% AUC)")
    print(f"  - Suggests scalability issues with current GraphRNN implementation")
    
    print(f"\n🔧 Recommendations:")
    print(f"  1. 📈 Increase GraphRNN training epochs (50 → 100+)")
    print(f"  2. 🎛️  Adjust learning rate (try 0.0001)")
    print(f"  3. 🏗️  Increase model capacity (hidden_dim, rnn_layers)")
    print(f"  4. 🎯 Use different training strategy (curriculum learning)")
    print(f"  5. 📊 Try TGIB to see if it handles dense graphs better")


if __name__ == "__main__":
    compare_dense_results()
    
    print(f"\n" + "=" * 80)
    print("CONCLUSION: Rule-based perfect, GraphRNN struggling with scale")
    print("Ready to test TGIB on this challenging dense dataset!")
    print("=" * 80)
