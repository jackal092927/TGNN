#!/usr/bin/env python3
"""
Manually capture interpretation data from the final resumed model.
"""

import os
import json
import torch
from test_graphmamba_contagion_explain_enhanced import (
    load_saved_model,
    evaluate_contagion_prediction_with_details,
    load_contagion_data,
    create_contagion_graph_sequence
)

def capture_final_interpretation():
    """Capture interpretation data from the final resumed model"""
    print("🔍 Capturing interpretation data from final resumed model...")
    
    # Use the final model from resume training
    model_file = "./test_small_model_interpretation/synthetic_icm_ba_resumed_final_model.pth"
    
    if not os.path.exists(model_file):
        print(f"❌ Final model not found: {model_file}")
        return None
    
    # Load the model
    print(f"📥 Loading model from: {model_file}")
    model, checkpoint = load_saved_model(model_file, 'cpu')
    print(f"✅ Model loaded successfully!")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Best AP: {checkpoint['best_val_ap']}")
    
    # Load dataset
    print("📊 Loading dataset...")
    g_df = load_contagion_data('synthetic_icm_ba')
    timestamps = sorted(g_df['ts'].unique())
    print(f"✅ Dataset loaded: {len(g_df)} edges, {len(timestamps)} timestamps")
    
    # Create graph sequence
    print("🔗 Creating graph sequence...")
    graph_sequence = create_contagion_graph_sequence(g_df, timestamps)
    print(f"✅ Graph sequence created: {len(graph_sequence)} graphs")
    
    # Capture interpretation data
    print("🔍 Capturing interpretation data...")
    test_metrics = evaluate_contagion_prediction_with_details(
        model, graph_sequence, g_df, timestamps, 'cpu', None
    )
    
    if test_metrics and 'details' in test_metrics:
        print("✅ Interpretation data captured successfully!")
        details = test_metrics['details']
        print(f"   - {len(details['predictions'])} predictions")
        print(f"   - {len(details['gates'])} gate sets")
        print(f"   - {len(details['embeddings'])} embedding sets")
        print(f"   - {len(details['pairs'])} node pairs")
        print(f"   - {len(details['timestamps'])} timestamps")
        
        # Save the results with interpretation data
        results_data = {
            'data_name': 'synthetic_icm_ba',
            'best_val_ap': checkpoint['best_val_ap'],
            'test_metrics': {
                'accuracy': test_metrics['accuracy'],
                'auc': test_metrics['auc'],
                'ap': test_metrics['ap']
            },
            'hyperparameters': checkpoint['hyperparameters'],
            'model_config': checkpoint['model_config'],
            'details': details
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if 'details' in results_data:
            details = results_data['details'].copy()
            for key, value in details.items():
                if hasattr(value, 'tolist'):  # Check if it's a numpy array
                    details[key] = value.tolist()
            results_data['details'] = details
        
        # Save results
        output_file = "./test_small_model_interpretation/synthetic_icm_ba_results_with_interpretation.json"
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
        print(f"   File size: {os.path.getsize(output_file)} bytes")
        
        return output_file
        
    else:
        print("❌ Failed to capture interpretation data")
        return None

if __name__ == "__main__":
    output_file = capture_final_interpretation()
    if output_file:
        print(f"\n🎉 Success! You can now run the interpretation visualization:")
        print(f"   python visualize_graph_interpretation.py")
        print(f"   (Make sure to update the results file path in the script)")
    else:
        print("\n❌ Failed to capture interpretation data")
