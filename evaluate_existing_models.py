"""
Evaluate Existing Trained GraphMamba Models on Contagion Data
------------------------------------------------------------
This script loads your previously trained models and runs comprehensive evaluation.
"""

import torch
import logging
import os
from evaluate_graphmamba_contagion import ContagionEvaluator, ContagionVisualizer

def load_existing_model(model_path, max_nodes, device):
    """Load an existing trained GraphMamba model"""
    from graphmamba_IB_explain import GraphMamba
    
    # Create model with same architecture
    model = GraphMamba(max_nodes=max_nodes, pos_dim=128, hidden_dim=64,
                       gnn_layers=2, mamba_state_dim=16, dropout=0.1,
                       use_edge_gates=True, gate_temperature=1.0).to(device)
    
    # Load trained weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        return None

def evaluate_existing_models():
    """Evaluate all existing trained models"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Available models and their datasets
    models_to_evaluate = [
        {
            'name': 'GraphMamba-ICM-BA',
            'model_path': 'saved_models/-attn-prod-synthetic_icm_ba-42.pth',
            'dataset': 'synthetic_icm_ba',
            'description': 'Independent Cascade Model on Barabási-Albert'
        },
        {
            'name': 'GraphMamba-LTM-BA', 
            'model_path': 'saved_models/-attn-prod-synthetic_ltm_ba-42.pth',
            'dataset': 'synthetic_ltm_ba',
            'description': 'Linear Threshold Model on Barabási-Albert'
        }
    ]
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    all_results = {}
    
    for model_info in models_to_evaluate:
        model_name = model_info['name']
        model_path = model_info['model_path']
        dataset = model_info['dataset']
        description = model_info['description']
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Dataset: {dataset}")
        print(f"Description: {description}")
        print(f"{'='*60}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            continue
        
        # Load data
        try:
            from test_graphmamba_IB_contagion_explain import load_contagion_data, create_contagion_graph_sequence
            
            logger.info(f"Loading {dataset} dataset...")
            g_df = load_contagion_data(dataset)
            timestamps = sorted(g_df['ts'].unique())
            
            # Create graph sequence
            graph_sequence = create_contagion_graph_sequence(g_df, timestamps)
            max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1
            
            logger.info(f"Dataset: {len(timestamps)} timestamps, {len(g_df)} edges")
            logger.info(f"Graph sequence: {len(graph_sequence)} graphs, max_nodes: {max_nodes}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset}: {e}")
            continue
        
        # Load model
        model = load_existing_model(model_path, max_nodes, device)
        if model is None:
            continue
        
        # Move graph sequence to device
        graph_sequence = [adj.to(device) for adj in graph_sequence]
        
        # Run comprehensive evaluation
        logger.info(f"Running comprehensive evaluation for {model_name}...")
        try:
            evaluator = ContagionEvaluator(model, graph_sequence, g_df, timestamps, device, logger)
            results = evaluator.run_comprehensive_evaluation()
            
            # Store results
            all_results[model_name] = results
            
            # Create visualizations
            save_dir = f"./evaluation_results/{model_name.replace(' ', '_')}"
            visualizer = ContagionVisualizer(results, save_dir)
            
            # Generate all plots
            visualizer.plot_explanation_performance()
            visualizer.plot_fidelity_curves()
            visualizer.plot_process_grounded_metrics()
            visualizer.plot_parsimony_stability()
            
            # Create summary report
            summary = visualizer.create_summary_report()
            print(f"\n📊 Evaluation Summary for {model_name}:")
            print(summary)
            
            logger.info(f"✅ Evaluation completed for {model_name}! Results saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for {model_name}: {e}")
            continue
    
    # Create comparison report
    if len(all_results) > 1:
        create_comparison_report(all_results)
    
    return all_results

def create_comparison_report(all_results):
    """Create a comparison report across all models"""
    print(f"\n{'='*80}")
    print("COMPARISON REPORT ACROSS ALL MODELS")
    print(f"{'='*80}")
    
    # Compare explanation-only performance
    print("\n📈 EXPLANATION-ONLY PERFORMANCE COMPARISON:")
    print("-" * 50)
    
    for model_name, results in all_results.items():
        exp_data = results['explanation_only']
        best_ap_idx = np.argmax(exp_data['ap_scores'])
        best_auc_idx = np.argmax(exp_data['auc_scores'])
        
        print(f"\n{model_name}:")
        print(f"  Best AP: {exp_data['ap_scores'][best_ap_idx]:.4f} at sparsity {exp_data['sparsity_levels'][best_ap_idx]:.2f}")
        print(f"  Best AUC: {exp_data['auc_scores'][best_auc_idx]:.4f} at sparsity {exp_data['sparsity_levels'][best_auc_idx]:.2f}")
        print(f"  AP at 20% sparsity: {exp_data['ap_scores'][3]:.4f}")
        print(f"  AUC at 20% sparsity: {exp_data['auc_scores'][3]:.4f}")
    
    # Compare fidelity
    print("\n🎯 FIDELITY COMPARISON:")
    print("-" * 50)
    
    for model_name, results in all_results.items():
        fid_data = results['fidelity']
        print(f"\n{model_name}:")
        print(f"  Deletion AUC range: [{min(fid_data['deletion_auc']):.4f}, {max(fid_data['deletion_auc']):.4f}]")
        print(f"  Insertion AUC range: [{min(fid_data['insertion_auc']):.4f}, {max(fid_data['insertion_auc']):.4f}]")
        print(f"  Max prediction drop: {max(fid_data['prediction_drops']):.4f}")
    
    # Compare process-grounded metrics
    print("\n🔍 PROCESS-GROUNDED METRICS COMPARISON:")
    print("-" * 50)
    
    for model_name, results in all_results.items():
        proc_data = results['process_grounded']
        print(f"\n{model_name}:")
        print(f"  Path Recall@10: {proc_data['path_recall'][1]:.4f}")
        print(f"  Path Recall@20: {proc_data['path_recall'][2]:.4f}")
        print(f"  Counterfactual drop@20: {proc_data['counterfactual_drops'][2]:.4f}")
        print(f"  Temporal coverage@20: {proc_data['temporal_coverage'][2]:.4f}")
    
    print(f"\n{'='*80}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Existing GraphMamba Models')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--compare', action='store_true', help='Create comparison report')
    
    args = parser.parse_args()
    
    # Set GPU
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Run evaluation
    results = evaluate_existing_models()
    
    if args.compare and len(results) > 1:
        create_comparison_report(results)
    
    print(f"\n🎉 All evaluations completed!")
    print(f"Check the './evaluation_results/' directory for detailed results.")

if __name__ == "__main__":
    import numpy as np
    main()
