"""
Debug why GraphRNN fails on the dense dataset
"""

import torch
import numpy as np
import pandas as pd
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from graph_rnn_correct import GraphRNN_Correct
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    # Load ground truth
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    return g_df, e_feat, n_feat, ground_truth

def analyze_dataset_complexity(g_df):
    """Analyze the complexity characteristics of the dataset"""
    logger.info("=== DATASET COMPLEXITY ANALYSIS ===")
    
    # Basic stats from dataframe
    edges = g_df[['u', 'i', 'ts']].values  # u, i, timestamp
    timestamps = edges[:, 2]
    nodes = set(edges[:, 0].tolist() + edges[:, 1].tolist())
    
    logger.info(f"Total edges: {len(edges)}")
    logger.info(f"Total nodes: {len(nodes)}")
    logger.info(f"Timestamps: {timestamps.min():.0f} to {timestamps.max():.0f}")
    logger.info(f"Timeline length: {timestamps.max() - timestamps.min() + 1:.0f}")
    
    # Edges per timestamp
    edges_per_ts = defaultdict(int)
    for ts in timestamps:
        edges_per_ts[int(ts)] += 1
    
    edge_counts = list(edges_per_ts.values())
    logger.info(f"Edges per timestamp - Mean: {np.mean(edge_counts):.1f}, Std: {np.std(edge_counts):.1f}")
    logger.info(f"Edges per timestamp - Min: {min(edge_counts)}, Max: {max(edge_counts)}")
    
    # Graph density over time
    densities = []
    cumulative_edges = set()
    
    for ts in range(int(timestamps.min()), int(timestamps.max()) + 1):
        # Add edges from this timestamp
        ts_mask = timestamps == ts
        ts_edges = edges[ts_mask]
        for edge in ts_edges:
            cumulative_edges.add((int(edge[0]), int(edge[1])))
        
        # Calculate density
        max_edges = len(nodes) * (len(nodes) - 1) / 2
        density = len(cumulative_edges) / max_edges
        densities.append(density)
    
    logger.info(f"Graph density - Start: {densities[0]:.4f}, End: {densities[-1]:.4f}")
    logger.info(f"Density growth: {(densities[-1] - densities[0]):.4f}")
    
    return {
        'edges_per_ts': edge_counts,
        'densities': densities,
        'total_edges': len(edges),
        'total_nodes': len(nodes),
        'timeline_length': int(timestamps.max() - timestamps.min() + 1)
    }

def analyze_model_predictions(model_path, g_df, device):
    """Analyze what the model is actually learning"""
    logger.info("=== MODEL PREDICTION ANALYSIS ===")
    
    # Load trained model
    model = GraphRNN_Correct(
        node_feat_dim=1,  # Dummy node features
        hidden_dim=128,
        max_nodes=201,  # +1 for padding
        rnn_layers=2
    ).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Create graph sequence from dataframe
    edges = g_df[['u', 'i', 'ts']].values
    timestamps = edges[:, 2]
    max_timestamp = int(timestamps.max())
    
    graph_sequence = []
    for ts in range(max_timestamp + 1):
        ts_mask = timestamps == ts
        ts_edges = edges[ts_mask]
        edge_pairs = [(int(e[0]), int(e[1])) for e in ts_edges]
        graph_sequence.append(edge_pairs)
    
    # Analyze predictions at different timestamps
    with torch.no_grad():
        hidden = None
        prediction_stats = []
        
        for ts in range(len(graph_sequence)):
            # Current graph state
            current_edges = set()
            for t in range(ts + 1):
                current_edges.update(graph_sequence[t])
            
            # Create adjacency matrix
            adj_matrix = torch.zeros(201, 201)
            for i, j in current_edges:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
            
            adj_matrix = adj_matrix.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get model prediction
            logits, hidden = model(adj_matrix, hidden)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Analyze prediction distribution
            stats = {
                'timestamp': ts,
                'num_current_edges': len(current_edges),
                'pred_mean': np.mean(probs),
                'pred_std': np.std(probs),
                'pred_min': np.min(probs),
                'pred_max': np.max(probs),
                'pred_above_05': np.sum(probs > 0.5) / len(probs.flatten()),
                'logits_mean': np.mean(logits.cpu().numpy()),
                'logits_std': np.std(logits.cpu().numpy())
            }
            prediction_stats.append(stats)
            
            if ts < 5 or ts % 5 == 0:  # Log first few and every 5th
                logger.info(f"TS {ts}: Edges={stats['num_current_edges']}, "
                          f"Pred μ={stats['pred_mean']:.4f}±{stats['pred_std']:.4f}, "
                          f"Logits μ={stats['logits_mean']:.4f}±{stats['logits_std']:.4f}, "
                          f">0.5: {stats['pred_above_05']:.1%}")
    
    return prediction_stats

def analyze_gradient_flow(model_path, g_df, device):
    """Analyze if gradients are flowing properly during training"""
    logger.info("=== GRADIENT FLOW ANALYSIS ===")
    
    model = GraphRNN_Correct(
        node_feat_dim=1,
        hidden_dim=128,
        max_nodes=201,
        rnn_layers=2
    ).to(device)
    
    # Simulate one training step
    edges = g_df[['u', 'i', 'ts']].values
    timestamps = edges[:, 2]
    
    # Get first few timestamps for analysis
    first_edges = set()
    for ts in range(3):  # First 3 timestamps
        ts_mask = timestamps == ts
        ts_edges = edges[ts_mask]
        for edge in ts_edges:
            first_edges.add((int(edge[0]), int(edge[1])))
    
    # Create adjacency matrix
    adj_matrix = torch.zeros(201, 201)
    for i, j in first_edges:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    
    adj_matrix = adj_matrix.unsqueeze(0).to(device)
    adj_matrix.requires_grad_(True)
    
    # Forward pass
    logits, _ = model(adj_matrix)
    
    # Create dummy target (some positive edges)
    target = torch.zeros_like(logits)
    # Set some edges as positive targets
    target[0, :10, :10] = 1  # First 10x10 block as positive
    
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    
    # Backward pass
    loss.backward()
    
    # Analyze gradients
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            grad_stats[name] = {
                'grad_norm': grad_norm,
                'param_norm': param_norm,
                'grad_param_ratio': grad_norm / (param_norm + 1e-8)
            }
            
            logger.info(f"{name}: grad_norm={grad_norm:.6f}, param_norm={param_norm:.6f}, ratio={grad_norm/(param_norm+1e-8):.6f}")
    
    return grad_stats

def compare_with_smaller_datasets():
    """Compare characteristics with smaller datasets that worked"""
    logger.info("=== COMPARISON WITH WORKING DATASETS ===")
    
    datasets = ['triadic_perfect_medium', 'triadic_perfect_large', 'triadic_perfect_long_dense']
    
    for dataset_name in datasets:
        try:
            g_df, _, _, _ = load_triadic_data(dataset_name)
            edges = g_df[['u', 'i', 'ts']].values
            timestamps = edges[:, 2]
            nodes = set(edges[:, 0].tolist() + edges[:, 1].tolist())
            
            timeline_length = int(timestamps.max() - timestamps.min() + 1)
            edges_per_ts = len(edges) / timeline_length
            
            logger.info(f"{dataset_name}:")
            logger.info(f"  Edges: {len(edges)}, Nodes: {len(nodes)}, Timeline: {timeline_length}")
            logger.info(f"  Edges/timestamp: {edges_per_ts:.1f}")
            logger.info(f"  Complexity score: {len(edges) * timeline_length / len(nodes):.1f}")
            
        except Exception as e:
            logger.info(f"{dataset_name}: Failed to load - {e}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load the problematic dataset
    g_df, e_feat, n_feat, ground_truth = load_triadic_data('triadic_perfect_long_dense')
    
    # 1. Analyze dataset complexity
    complexity_stats = analyze_dataset_complexity(g_df)
    
    # 2. Compare with working datasets
    compare_with_smaller_datasets()
    
    # 3. Analyze model predictions (if model exists)
    model_path = 'models/graphrnn_correct_triadic_perfect_long_dense_best.pt'
    try:
        prediction_stats = analyze_model_predictions(model_path, g_df, device)
        
        if prediction_stats:
            # Check for prediction collapse
            final_stats = prediction_stats[-1]
            if final_stats['pred_std'] < 0.01:
                logger.warning("⚠️  PREDICTION COLLAPSE: Model outputs have very low variance")
            
            if abs(final_stats['pred_mean'] - 0.5) < 0.05:
                logger.warning("⚠️  RANDOM PREDICTIONS: Model predictions centered around 0.5")
                
            if final_stats['logits_mean'] < -2 or final_stats['logits_mean'] > 2:
                logger.warning("⚠️  EXTREME LOGITS: Model may be saturating")
    
    except FileNotFoundError:
        logger.warning("Model file not found, skipping prediction analysis")
    
    # 4. Analyze gradient flow
    try:
        grad_stats = analyze_gradient_flow(model_path, g_df, device)
        
        # Check for vanishing gradients
        small_gradients = sum(1 for stats in grad_stats.values() if stats['grad_norm'] < 1e-6)
        if small_gradients > len(grad_stats) / 2:
            logger.warning("⚠️  VANISHING GRADIENTS: Many parameters have very small gradients")
            
    except Exception as e:
        logger.info(f"Gradient analysis failed: {e}")
    
    # 5. Hypotheses for failure
    logger.info("=== FAILURE HYPOTHESES ===")
    
    hypotheses = [
        "1. 🔢 SCALE ISSUE: 1,458 edges >> previous datasets (40-100 edges)",
        "2. ⏱️  SEQUENCE LENGTH: 28 timestamps may cause vanishing gradients in LSTM",
        "3. 🕸️  GRAPH DENSITY: Dense graphs harder to learn than sparse ones",
        "4. 🎯 OPTIMIZATION: Current hyperparameters not suitable for large scale",
        "5. 🧠 CAPACITY: Model too small for complex temporal patterns",
        "6. 📊 DATA DISTRIBUTION: Dense dataset has different edge patterns"
    ]
    
    for hypothesis in hypotheses:
        logger.info(hypothesis)
    
    logger.info("=== RECOMMENDED FIXES ===")
    fixes = [
        "1. 📈 Increase training epochs: 50 → 200+",
        "2. 🎛️  Reduce learning rate: 0.001 → 0.0001",
        "3. 🏗️  Increase model capacity: hidden_dim=256, rnn_layers=3",
        "4. 🔄 Add gradient clipping: max_norm=1.0",
        "5. 📚 Use curriculum learning: start with smaller subgraphs",
        "6. 🎯 Better initialization: Xavier/He initialization",
        "7. 📊 Different loss weighting: focus on positive edges"
    ]
    
    for fix in fixes:
        logger.info(fix)

if __name__ == "__main__":
    main()
