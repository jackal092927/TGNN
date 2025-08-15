"""
Test script for hybrid training mode only
"""

from train_tgam_flexible import train_tgam_flexible

# Hybrid mode configuration
config = {
    'training_mode': 'hybrid',
    'teacher_forcing': True,
    'teacher_forcing_ratio': 0.8,  # 80% teacher forcing
    'dataset': 'triadic_fixed',
    'epochs': 15,
    'lr': 0.001,
    'steps_per_epoch': 30,
    'hidden_dim': 128,
    'max_nodes': 100,
    'num_graph_layers': 2,
    'num_temporal_layers': 4
}

print("=== Testing Hybrid Training Mode ===")
print(f"Configuration: {config}")
print()

# Run hybrid training
results = train_tgam_flexible(config)

print("\n=== Hybrid Training Results ===")
print(f"Final Accuracy: {results['final_accuracy']:.4f}")
print(f"Final AP Score: {results['final_ap']:.4f}")
print(f"Final AUC Score: {results['final_auc']:.4f}")
print(f"Training Improvement: {results['improvement']:.4f}")

print("\n=== Comparison with Previous Results ===")
print("TGAM Individual: Acc=80.00%, AP=92.50%, AUC=87.50%")
print("TGIB Original:   Acc=77.50%, AP=95.00%, AUC=90.00%")
print(f"TGAM Hybrid:     Acc={results['final_accuracy']*100:.2f}%, AP={results['final_ap']*100:.2f}%, AUC={results['final_auc']*100:.2f}%") 