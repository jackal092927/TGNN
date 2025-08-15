"""
Comprehensive comparison between TGIB and TGAM on real and synthetic datasets.

This script evaluates both models to determine which approach works better
for different types of temporal graph data.
"""

import os
import sys
import argparse
import logging
import time
import pickle
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging():
    """Setup logging for comparison script"""
    Path("comparison_results").mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    timestamp = int(time.time())
    fh = logging.FileHandler(f'comparison_results/comparison_{timestamp}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def get_available_datasets():
    """Find all available datasets in the processed folder"""
    processed_dir = Path("./processed")
    if not processed_dir.exists():
        return []
    
    datasets = []
    for item in processed_dir.iterdir():
        if item.is_dir():
            csv_file = item / f"ml_{item.name}.csv"
            npy_file = item / f"ml_{item.name}.npy"
            node_file = item / f"ml_{item.name}_node.npy"
            
            if csv_file.exists() and npy_file.exists() and node_file.exists():
                datasets.append(item.name)
    
    return sorted(datasets)


def classify_datasets(datasets):
    """Classify datasets as real or synthetic based on naming patterns"""
    real_datasets = []
    synthetic_datasets = []
    
    # Common real dataset names
    real_patterns = ['wikipedia', 'reddit', 'uci', 'canparl', 'enron', 'dblp', 'mooc']
    
    # Common synthetic dataset patterns
    synthetic_patterns = ['synthetic', 'triadic', 'ba', 'er', 'ws', 'icm', 'ltm', 'cc', 'sd']
    
    for dataset in datasets:
        dataset_lower = dataset.lower()
        
        is_real = any(pattern in dataset_lower for pattern in real_patterns)
        is_synthetic = any(pattern in dataset_lower for pattern in synthetic_patterns)
        
        if is_real and not is_synthetic:
            real_datasets.append(dataset)
        elif is_synthetic:
            synthetic_datasets.append(dataset)
        else:
            # Default classification based on common naming
            if any(name in dataset_lower for name in ['wikipedia', 'reddit', 'uci']):
                real_datasets.append(dataset)
            else:
                synthetic_datasets.append(dataset)
    
    return real_datasets, synthetic_datasets


def run_tgib_experiment(dataset, gpu=0, epochs=20):
    """Run TGIB experiment on a dataset"""
    cmd = [
        'python', 'learn_edge_ori.py',
        '-d', dataset,
        '--gpu', str(gpu),
        '--n_epoch', str(epochs),
        '--bs', '200',
        '--lr', '0.00001'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            # Parse the output to extract performance metrics
            output_lines = result.stdout.split('\n')
            
            # Look for test statistics in the output
            test_stats = {}
            for line in output_lines:
                if "Test statistics: Old nodes" in line:
                    # Parse line like: "Test statistics: Old nodes -- acc: 0.8234, auc: 0.7123, ap: 0.6789"
                    parts = line.split('--')[1].strip()
                    metrics = {}
                    for metric in parts.split(','):
                        if ':' in metric:
                            key, value = metric.split(':')
                            key = key.strip()
                            try:
                                value = float(value.strip())
                                metrics[key] = value
                            except:
                                continue
                    test_stats['old_nodes'] = metrics
                
                elif "Test statistics: New nodes" in line:
                    parts = line.split('--')[1].strip()
                    metrics = {}
                    for metric in parts.split(','):
                        if ':' in metric:
                            key, value = metric.split(':')
                            key = key.strip()
                            try:
                                value = float(value.strip())
                                metrics[key] = value
                            except:
                                continue
                    test_stats['new_nodes'] = metrics
            
            return test_stats, result.stdout, None
        else:
            return None, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        return None, "", "Timeout after 1 hour"
    except Exception as e:
        return None, "", str(e)


def run_tgam_experiment(dataset, gpu=0, epochs=20):
    """Run TGAM experiment on a dataset"""
    cmd = [
        'python', 'train_tgam.py',
        '-d', dataset,
        '--gpu', str(gpu),
        '--n_epoch', str(epochs),
        '--bs', '32',
        '--lr', '0.001',
        '--seq_len', '10'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            # Look for results file
            timestamp = int(time.time())
            results_pattern = f"results_tgam_{dataset}_*.pkl"
            
            # Find the most recent results file
            import glob
            results_files = glob.glob(results_pattern)
            if results_files:
                latest_file = max(results_files, key=os.path.getctime)
                with open(latest_file, 'rb') as f:
                    results = pickle.load(f)
                return results, result.stdout, None
            else:
                # Parse stdout for metrics
                output_lines = result.stdout.split('\n')
                results = {}
                for line in output_lines:
                    if "Final Test Results:" in line:
                        continue
                    elif "Accuracy:" in line:
                        results['test_acc'] = float(line.split(':')[1].strip())
                    elif "AUC:" in line:
                        results['test_auc'] = float(line.split(':')[1].strip())
                    elif "Average Precision:" in line:
                        results['test_ap'] = float(line.split(':')[1].strip())
                    elif "F1-Score:" in line:
                        results['test_f1'] = float(line.split(':')[1].strip())
                
                return results, result.stdout, None
        else:
            return None, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        return None, "", "Timeout after 1 hour"
    except Exception as e:
        return None, "", str(e)


def create_comparison_plots(results_df, output_dir):
    """Create comparison plots between TGIB and TGAM"""
    plt.style.use('seaborn-v0_8')
    
    # Performance comparison by dataset type
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['ap', 'auc', 'acc', 'f1']
    metric_names = ['Average Precision', 'AUC', 'Accuracy', 'F1-Score']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        # Create comparison data
        comparison_data = []
        for _, row in results_df.iterrows():
            if row['tgib_success'] and row[f'tgib_{metric}'] is not None:
                comparison_data.append({
                    'Dataset': row['dataset'],
                    'Type': row['type'],
                    'Model': 'TGIB',
                    'Score': row[f'tgib_{metric}']
                })
            
            if row['tgam_success'] and row[f'tgam_{metric}'] is not None:
                comparison_data.append({
                    'Dataset': row['dataset'],
                    'Type': row['type'],
                    'Model': 'TGAM',
                    'Score': row[f'tgam_{metric}']
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Box plot comparing models by dataset type
            sns.boxplot(data=comp_df, x='Type', y='Score', hue='Model', ax=ax)
            ax.set_title(f'{name} by Dataset Type')
            ax.set_ylabel(name)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Dataset-specific comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create dataset comparison for AP metric
    dataset_comparison = []
    for _, row in results_df.iterrows():
        if row['tgib_success'] and row['tgam_success']:
            dataset_comparison.append({
                'Dataset': row['dataset'],
                'Type': row['type'],
                'TGIB_AP': row['tgib_ap'],
                'TGAM_AP': row['tgam_ap'],
                'Difference': row['tgam_ap'] - row['tgib_ap']
            })
    
    if dataset_comparison:
        comp_df = pd.DataFrame(dataset_comparison)
        
        # Create bar plot showing which model performs better
        colors = ['red' if x < 0 else 'blue' for x in comp_df['Difference']]
        bars = ax.bar(range(len(comp_df)), comp_df['Difference'], color=colors, alpha=0.7)
        
        ax.set_xlabel('Datasets')
        ax.set_ylabel('TGAM AP - TGIB AP')
        ax.set_title('Performance Difference (TGAM - TGIB) by Dataset')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Customize x-axis labels
        ax.set_xticks(range(len(comp_df)))
        labels = [f"{row['Dataset']}\n({row['Type']})" for _, row in comp_df.iterrows()]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, comp_df['Difference'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                   f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_specific_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(results_df, output_dir, logger):
    """Generate a summary report of the comparison"""
    
    report_path = output_dir / 'comparison_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("TGIB vs TGAM Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        total_datasets = len(results_df)
        tgib_success = results_df['tgib_success'].sum()
        tgam_success = results_df['tgam_success'].sum()
        both_success = (results_df['tgib_success'] & results_df['tgam_success']).sum()
        
        f.write(f"Total datasets tested: {total_datasets}\n")
        f.write(f"TGIB successful runs: {tgib_success}\n")
        f.write(f"TGAM successful runs: {tgam_success}\n")
        f.write(f"Both models successful: {both_success}\n\n")
        
        # Dataset type breakdown
        real_datasets = results_df[results_df['type'] == 'Real']
        synthetic_datasets = results_df[results_df['type'] == 'Synthetic']
        
        f.write(f"Real datasets: {len(real_datasets)}\n")
        f.write(f"Synthetic datasets: {len(synthetic_datasets)}\n\n")
        
        # Performance comparison on successful runs
        successful_runs = results_df[results_df['tgib_success'] & results_df['tgam_success']]
        
        if len(successful_runs) > 0:
            f.write("Performance Comparison (Average Precision)\n")
            f.write("-" * 40 + "\n")
            
            # Overall comparison
            tgib_mean_ap = successful_runs['tgib_ap'].mean()
            tgam_mean_ap = successful_runs['tgam_ap'].mean()
            
            f.write(f"Overall TGIB AP: {tgib_mean_ap:.4f}\n")
            f.write(f"Overall TGAM AP: {tgam_mean_ap:.4f}\n")
            f.write(f"Difference (TGAM - TGIB): {tgam_mean_ap - tgib_mean_ap:.4f}\n\n")
            
            # By dataset type
            for dataset_type in ['Real', 'Synthetic']:
                subset = successful_runs[successful_runs['type'] == dataset_type]
                if len(subset) > 0:
                    tgib_ap = subset['tgib_ap'].mean()
                    tgam_ap = subset['tgam_ap'].mean()
                    
                    f.write(f"{dataset_type} datasets:\n")
                    f.write(f"  TGIB AP: {tgib_ap:.4f}\n")
                    f.write(f"  TGAM AP: {tgam_ap:.4f}\n")
                    f.write(f"  Difference: {tgam_ap - tgib_ap:.4f}\n")
                    f.write(f"  Count: {len(subset)}\n\n")
            
            # Individual dataset results
            f.write("Individual Dataset Results\n")
            f.write("-" * 40 + "\n")
            for _, row in successful_runs.iterrows():
                f.write(f"{row['dataset']} ({row['type']}):\n")
                f.write(f"  TGIB: AP={row['tgib_ap']:.4f}, AUC={row['tgib_auc']:.4f}\n")
                f.write(f"  TGAM: AP={row['tgam_ap']:.4f}, AUC={row['tgam_auc']:.4f}\n")
                f.write(f"  Difference: {row['tgam_ap'] - row['tgib_ap']:.4f}\n\n")
        
        # Failed runs analysis
        f.write("Failed Runs Analysis\n")
        f.write("-" * 40 + "\n")
        
        tgib_failures = results_df[~results_df['tgib_success']]
        tgam_failures = results_df[~results_df['tgam_success']]
        
        if len(tgib_failures) > 0:
            f.write("TGIB failures:\n")
            for _, row in tgib_failures.iterrows():
                f.write(f"  {row['dataset']} ({row['type']}): {row['tgib_error']}\n")
            f.write("\n")
        
        if len(tgam_failures) > 0:
            f.write("TGAM failures:\n")
            for _, row in tgam_failures.iterrows():
                f.write(f"  {row['dataset']} ({row['type']}): {row['tgam_error']}\n")
            f.write("\n")
    
    logger.info(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser('Model Comparison Script')
    parser.add_argument('--datasets', nargs='+', help='specific datasets to test')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for training')
    parser.add_argument('--skip-tgib', action='store_true', help='skip TGIB experiments')
    parser.add_argument('--skip-tgam', action='store_true', help='skip TGAM experiments')
    parser.add_argument('--real-only', action='store_true', help='test only real datasets')
    parser.add_argument('--synthetic-only', action='store_true', help='test only synthetic datasets')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting model comparison")
    
    # Get available datasets
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = get_available_datasets()
    
    if not datasets:
        logger.error("No datasets found!")
        return
    
    # Classify datasets
    real_datasets, synthetic_datasets = classify_datasets(datasets)
    
    # Filter based on arguments
    if args.real_only:
        datasets_to_test = real_datasets
    elif args.synthetic_only:
        datasets_to_test = synthetic_datasets
    else:
        datasets_to_test = datasets
    
    logger.info(f"Found {len(real_datasets)} real datasets: {real_datasets}")
    logger.info(f"Found {len(synthetic_datasets)} synthetic datasets: {synthetic_datasets}")
    logger.info(f"Testing {len(datasets_to_test)} datasets: {datasets_to_test}")
    
    # Results storage
    results = []
    
    # Run experiments
    for dataset in datasets_to_test:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing dataset: {dataset}")
        logger.info(f"{'='*50}")
        
        dataset_type = 'Real' if dataset in real_datasets else 'Synthetic'
        
        result_entry = {
            'dataset': dataset,
            'type': dataset_type,
            'tgib_success': False,
            'tgam_success': False,
            'tgib_ap': None,
            'tgib_auc': None,
            'tgib_acc': None,
            'tgib_f1': None,
            'tgam_ap': None,
            'tgam_auc': None,
            'tgam_acc': None,
            'tgam_f1': None,
            'tgib_error': None,
            'tgam_error': None
        }
        
        # Run TGIB
        if not args.skip_tgib:
            logger.info(f"Running TGIB on {dataset}...")
            tgib_results, tgib_stdout, tgib_error = run_tgib_experiment(dataset, args.gpu, args.epochs)
            
            if tgib_results and 'old_nodes' in tgib_results:
                result_entry['tgib_success'] = True
                old_nodes = tgib_results['old_nodes']
                result_entry['tgib_ap'] = old_nodes.get('ap', None)
                result_entry['tgib_auc'] = old_nodes.get('auc', None)
                result_entry['tgib_acc'] = old_nodes.get('acc', None)
                result_entry['tgib_f1'] = old_nodes.get('f1', None)
                logger.info(f"TGIB succeeded: AP={result_entry['tgib_ap']:.4f}")
            else:
                result_entry['tgib_error'] = tgib_error or "Failed to parse results"
                logger.error(f"TGIB failed: {result_entry['tgib_error']}")
        
        # Run TGAM
        if not args.skip_tgam:
            logger.info(f"Running TGAM on {dataset}...")
            tgam_results, tgam_stdout, tgam_error = run_tgam_experiment(dataset, args.gpu, args.epochs)
            
            if tgam_results and isinstance(tgam_results, dict):
                result_entry['tgam_success'] = True
                result_entry['tgam_ap'] = tgam_results.get('test_ap', None)
                result_entry['tgam_auc'] = tgam_results.get('test_auc', None)
                result_entry['tgam_acc'] = tgam_results.get('test_acc', None)
                result_entry['tgam_f1'] = tgam_results.get('test_f1', None)
                logger.info(f"TGAM succeeded: AP={result_entry['tgam_ap']:.4f}")
            else:
                result_entry['tgam_error'] = tgam_error or "Failed to parse results"
                logger.error(f"TGAM failed: {result_entry['tgam_error']}")
        
        results.append(result_entry)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path("comparison_results")
    timestamp = int(time.time())
    
    results_df.to_csv(output_dir / f'comparison_results_{timestamp}.csv', index=False)
    
    # Generate plots and reports
    if len(results_df) > 0:
        create_comparison_plots(results_df, output_dir)
        generate_summary_report(results_df, output_dir, logger)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("COMPARISON COMPLETE")
    logger.info("="*50)
    
    successful_both = results_df[results_df['tgib_success'] & results_df['tgam_success']]
    
    if len(successful_both) > 0:
        tgib_wins = (successful_both['tgib_ap'] > successful_both['tgam_ap']).sum()
        tgam_wins = (successful_both['tgam_ap'] > successful_both['tgib_ap']).sum()
        ties = (successful_both['tgib_ap'] == successful_both['tgam_ap']).sum()
        
        logger.info(f"Datasets where both models succeeded: {len(successful_both)}")
        logger.info(f"TGIB wins: {tgib_wins}")
        logger.info(f"TGAM wins: {tgam_wins}")
        logger.info(f"Ties: {ties}")
        
        # By dataset type
        for dataset_type in ['Real', 'Synthetic']:
            subset = successful_both[successful_both['type'] == dataset_type]
            if len(subset) > 0:
                type_tgib_wins = (subset['tgib_ap'] > subset['tgam_ap']).sum()
                type_tgam_wins = (subset['tgam_ap'] > subset['tgib_ap']).sum()
                logger.info(f"{dataset_type} datasets - TGIB: {type_tgib_wins}, TGAM: {type_tgam_wins}")


if __name__ == '__main__':
    main() 