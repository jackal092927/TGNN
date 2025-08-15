"""
Train and Evaluate GraphMamba on synthetic_icm_ba
-------------------------------------------------
This script trains a self-explaining GraphMamba model and then runs
comprehensive evaluation to show interpretation metrics and visualizations.
"""

import torch
import logging
import os
from evaluate_graphmamba_contagion import ContagionEvaluator, ContagionVisualizer

def train_and_evaluate_graphmamba(data_name='synthetic_icm_ba', epochs=30, gpu_id=0):
    """
    Train a GraphMamba model and immediately evaluate it
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting GraphMamba training and evaluation on {data_name}")
    
    # Import training function
    from test_graphmamba_IB_contagion_explain import (
        load_contagion_data, 
        create_contagion_graph_sequence,
        train_graphmamba_contagion
    )
    
    # Load data
    logger.info(f"Loading {data_name} dataset...")
    g_df = load_contagion_data(data_name)
    timestamps = sorted(g_df['ts'].unique())
    
    # Create graph sequence
    graph_sequence = create_contagion_graph_sequence(g_df, timestamps)
    max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1
    
    logger.info(f"Dataset: {len(timestamps)} timestamps, {len(g_df)} edges")
    logger.info(f"Graph sequence: {len(graph_sequence)} graphs, max_nodes: {max_nodes}")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available, using CPU")
    
    # Train GraphMamba model
    logger.info(f"Training GraphMamba model for {epochs} epochs...")
    logger.info("This will train a self-explaining model with edge gates for interpretability")
    
    model, metrics = train_graphmamba_contagion(
        data_name=data_name,
        epochs=epochs,
        lr=0.001,
        hidden_dim=64,
        pos_dim=128,
        mamba_state_dim=16,
        gpu_id=gpu_id,
        lambda_sparse=1e-4,
        lambda_tv=1e-3,
        gate_temperature=1.0
    )
    
    logger.info("Training completed! Now running comprehensive evaluation...")
    
    # Move graph sequence to device
    graph_sequence = [adj.to(device) for adj in graph_sequence]
    
    # Run comprehensive evaluation
    logger.info("Running comprehensive TGIB-style evaluation...")
    evaluator = ContagionEvaluator(model, graph_sequence, g_df, timestamps, device, logger)
    results = evaluator.run_comprehensive_evaluation()
    
    # Create visualizations
    save_dir = f"./evaluation_results/{data_name}_graphmamba"
    visualizer = ContagionVisualizer(results, save_dir)
    
    # Generate all plots
    logger.info("Generating visualization plots...")
    visualizer.plot_explanation_performance()
    visualizer.plot_fidelity_curves()
    visualizer.plot_process_grounded_metrics()
    visualizer.plot_parsimony_stability()
    
    # Create summary report
    summary = visualizer.create_summary_report()
    
    # Display results
    print("\n" + "="*80)
    print("🎉 GRAPHMAMBA TRAINING & EVALUATION COMPLETE!")
    print("="*80)
    print(f"Dataset: {data_name}")
    print(f"Training epochs: {epochs}")
    print(f"Model: Self-Explaining GraphMamba with edge gates")
    print("\n📊 TRAINING RESULTS:")
    print(f"Best Validation AP: {metrics['ap']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test AUC: {metrics['test_auc']:.4f}")
    print(f"Test AP: {metrics['test_ap']:.4f}")
    
    print("\n🔍 INTERPRETATION EVALUATION RESULTS:")
    print(summary)
    
    logger.info(f"All results saved to {save_dir}")
    
    return model, metrics, results, visualizer

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and Evaluate GraphMamba on Contagion Data')
    parser.add_argument('--data', type=str, default='synthetic_icm_ba', 
                       help='Dataset name (default: synthetic_icm_ba)')
    parser.add_argument('--epochs', type=int, default=30, 
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Train and evaluate
    model, metrics, results, visualizer = train_and_evaluate_graphmamba(
        data_name=args.data,
        epochs=args.epochs,
        gpu_id=args.gpu
    )
    
    print(f"\n🎯 Next steps:")
    print(f"1. Check the plots in {visualizer.save_dir}")
    print(f"2. Review the evaluation summary above")
    print(f"3. Analyze how well your model explains contagion dynamics")
    print(f"4. Use the trained model for further analysis")

if __name__ == "__main__":
    main()
