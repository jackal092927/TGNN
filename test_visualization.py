#!/usr/bin/env python3
"""
Test visualization script for the new CSV data format.
Creates plots to verify visualization functionality works.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def test_basic_visualization():
    """Test basic visualization with the CSV data"""
    print("Testing basic visualization...")
    
    # Load the visualization data
    viz_file = "./test_small_model/synthetic_icm_ba_viz_data.csv"
    
    if not os.path.exists(viz_file):
        print(f"❌ Visualization file not found: {viz_file}")
        return False
    
    # Load data
    viz_data = pd.read_csv(viz_file)
    print(f"✅ Loaded visualization data: {viz_data.shape}")
    print(f"   Columns: {list(viz_data.columns)}")
    print(f"   Data preview:")
    print(viz_data.head())
    
    # Create a simple plot
    plt.figure(figsize=(10, 6))
    
    # Plot 1: Activation rate over time
    plt.subplot(2, 2, 1)
    plt.plot(viz_data['timestamp'], viz_data['activation_rate'], 'b-o', linewidth=2, markersize=6)
    plt.title('Activation Rate Over Time', fontweight='bold')
    plt.xlabel('Timestamp')
    plt.ylabel('Activation Rate')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Active nodes over time
    plt.subplot(2, 2, 2)
    plt.plot(viz_data['timestamp'], viz_data['active_nodes'], 'r-s', linewidth=2, markersize=6)
    plt.title('Active Nodes Over Time', fontweight='bold')
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Active Nodes')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Bar chart of activation rates
    plt.subplot(2, 2, 3)
    bars = plt.bar(viz_data['timestamp'], viz_data['activation_rate'], color='skyblue', alpha=0.7)
    plt.title('Activation Rate Distribution', fontweight='bold')
    plt.xlabel('Timestamp')
    plt.ylabel('Activation Rate')
    
    # Add value labels on bars
    for bar, rate in zip(bars, viz_data['activation_rate']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.2f}', ha='center', va='bottom')
    
    # Plot 4: Cumulative activation
    plt.subplot(2, 2, 4)
    cumulative_active = viz_data['active_nodes'].cumsum()
    plt.plot(viz_data['timestamp'], cumulative_active, 'g-^', linewidth=2, markersize=6)
    plt.title('Cumulative Active Nodes', fontweight='bold')
    plt.xlabel('Timestamp')
    plt.ylabel('Cumulative Active Nodes')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "./test_small_model/visualization_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    return True

def test_model_results_visualization():
    """Test visualization of model training results"""
    print("\nTesting model results visualization...")
    
    # Check if results file exists
    results_file = "./test_small_model/synthetic_icm_ba_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return False
    
    # Load results
    import json
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"✅ Loaded results: {list(results.keys())}")
    
    # Create training metrics visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Model performance metrics
    plt.subplot(2, 3, 1)
    metrics = results.get('test_metrics', {})
    if metrics:
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Test Metrics', fontweight='bold')
        plt.ylabel('Score')
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 2: Hyperparameters
    plt.subplot(2, 3, 2)
    hyperparams = results.get('hyperparameters', {})
    if hyperparams:
        # Filter out non-numeric hyperparameters
        numeric_params = {k: v for k, v in hyperparams.items() if isinstance(v, (int, float))}
        if numeric_params:
            param_names = list(numeric_params.keys())
            param_values = list(numeric_params.values())
            
            bars = plt.bar(param_names, param_values, color='lightblue')
            plt.title('Hyperparameters', fontweight='bold')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
    
    # Plot 3: Best validation AP
    plt.subplot(2, 3, 3)
    best_ap = results.get('best_val_ap', 0)
    plt.bar(['Best Val AP'], [best_ap], color='gold', alpha=0.7)
    plt.title('Best Validation AP', fontweight='bold')
    plt.ylabel('Score')
    plt.text(0, best_ap + 0.01, f'{best_ap:.4f}', ha='center', va='bottom')
    
    # Plot 4: Dataset info
    plt.subplot(2, 3, 4)
    dataset_name = results.get('data_name', 'Unknown')
    plt.text(0.1, 0.5, f'Dataset: {dataset_name}', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.3, f'Best Val AP: {best_ap:.4f}', fontsize=10)
    plt.text(0.1, 0.1, f'Test AP: {metrics.get("ap", "N/A")}', fontsize=10)
    plt.axis('off')
    
    # Plot 5: Check if details exist
    plt.subplot(2, 3, 5)
    details = results.get('details', {})
    if details:
        detail_keys = list(details.keys())
        plt.text(0.1, 0.5, f'Details available:\n{", ".join(detail_keys)}', 
                fontsize=10, verticalalignment='center')
    else:
        plt.text(0.1, 0.5, 'No detailed results\navailable', 
                fontsize=10, verticalalignment='center', color='red')
    plt.axis('off')
    
    # Plot 6: Summary statistics
    plt.subplot(2, 3, 6)
    summary_stats = [
        f"Epochs: {hyperparams.get('epochs', 'N/A')}",
        f"LR: {hyperparams.get('lr', 'N/A')}",
        f"Hidden Dim: {hyperparams.get('hidden_dim', 'N/A')}",
        f"Pos Dim: {hyperparams.get('pos_dim', 'N/A')}"
    ]
    
    for i, stat in enumerate(summary_stats):
        plt.text(0.1, 0.8 - i*0.2, stat, fontsize=9)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "./test_small_model/model_results_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Model results visualization saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    return True

def main():
    """Run all visualization tests"""
    print("🧪 TESTING VISUALIZATION FUNCTIONALITY")
    print("="*50)
    
    # Test 1: Basic CSV visualization
    success1 = test_basic_visualization()
    
    # Test 2: Model results visualization
    success2 = test_model_results_visualization()
    
    # Summary
    print("\n" + "="*50)
    print("VISUALIZATION TEST SUMMARY")
    print("="*50)
    print(f"✅ Basic CSV Visualization: {'PASSED' if success1 else 'FAILED'}")
    print(f"✅ Model Results Visualization: {'PASSED' if success2 else 'FAILED'}")
    
    if success1 and success2:
        print("\n🎉 ALL VISUALIZATION TESTS PASSED!")
        print("   - CSV data visualization works")
        print("   - Model results visualization works")
        print("   - Check ./test_small_model/ for generated plots")
    else:
        print("\n❌ Some visualization tests failed.")
    
    print("\nGenerated files:")
    if success1:
        print("   - ./test_small_model/visualization_test.png")
    if success2:
        print("   - ./test_small_model/model_results_visualization.png")

if __name__ == "__main__":
    main()
