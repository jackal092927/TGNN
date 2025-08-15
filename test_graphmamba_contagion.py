"""
Test GraphMamba on Contagion Data

This script tests the GraphMamba model on synthetic contagion datasets
including Independent Cascade Model (ICM) and Linear Threshold Model (LTM).

Key Features:
- 2:1 downsampling strategy for balanced training (positive:negative ratio)
- 1:1 balanced sampling for validation/testing evaluation
- Handles class imbalance during training while ensuring fair evaluation
- Maintains temporal graph sequence integrity
- Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import logging
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import math
import argparse
import os
from graph_mamba import GraphMamba, create_sincos_positional_embeddings, PositionalGNNLayer, MambaBlock


def load_contagion_data(data_name):
    """Load contagion dataset"""
    data_path = f'./processed/{data_name}/ml_{data_name}.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    # Load graph data
    g_df = pd.read_csv(data_path)
    
    # Ensure proper column names
    if 'label' not in g_df.columns:
        raise ValueError(f"Dataset {data_name} missing 'label' column")
    
    print(f"Loaded {data_name}: {len(g_df)} edges, {g_df['label'].nunique()} unique labels")
    print(f"Label distribution: {g_df['label'].value_counts().to_dict()}")
    
    return g_df


def create_contagion_graph_sequence(g_df, timestamps):
    """Create sequence of adjacency matrices from contagion dataframe"""
    max_node = max(g_df['u'].max(), g_df['i'].max()) + 1
    graph_sequence = []
    
    for ts in timestamps:
        # Get edges up to current timestamp
        edges_up_to_ts = g_df[g_df['ts'] <= ts]
        
        # Create adjacency matrix
        adj_matrix = torch.zeros(max_node, max_node)
        
        for _, row in edges_up_to_ts.iterrows():
            u, v = int(row['u']), int(row['i'])
            adj_matrix[u, v] = 1.0
            adj_matrix[v, u] = 1.0  # Undirected graph
        
        graph_sequence.append(adj_matrix)
    
    return graph_sequence


def evaluate_contagion_prediction(model, graph_sequence, g_df, timestamps, device, logger):
    """Evaluate GraphMamba on contagion prediction task with 1:1 balanced sampling"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        # Get temporal embeddings for entire sequence
        sequence_embeddings = model.forward_sequence(graph_sequence)
        
        for i in range(len(timestamps) - 1):
            current_ts = timestamps[i]
            next_ts = timestamps[i + 1]
            
            # Get current node embeddings
            current_embeddings = sequence_embeddings[i]  # [num_nodes, hidden_dim]
            
            # Get edges that appear at the next timestamp
            next_edges = g_df[g_df['ts'] == next_ts]
            
            if len(next_edges) == 0:
                continue
            
            # Generate all possible node pairs
            num_nodes = current_embeddings.shape[0]
            all_pairs = []
            for u in range(num_nodes):
                for v in range(u + 1, num_nodes):
                    all_pairs.append((u, v))
            
            if len(all_pairs) == 0:
                continue
            
            # Separate positive and negative pairs
            positive_pairs = []
            negative_pairs = []
            
            for pair in all_pairs:
                u, v = pair
                # Check if this edge exists in next timestamp
                edge_exists = len(next_edges[(next_edges['u'] == u) & (next_edges['i'] == v)]) > 0 or \
                            len(next_edges[(next_edges['u'] == v) & (next_edges['i'] == u)]) > 0
                
                if edge_exists:
                    positive_pairs.append(pair)
                else:
                    negative_pairs.append(pair)
            
            if len(positive_pairs) == 0:
                continue
            
            # 1:1 balanced sampling for validation/testing evaluation
            # This ensures fair evaluation on balanced data while training uses 2:1 sampling
            num_samples = min(len(positive_pairs), len(negative_pairs))
            if num_samples == 0:
                continue
            
            # Sample equal numbers of positive and negative pairs
            sampled_positive_indices = torch.randperm(len(positive_pairs))[:num_samples]
            sampled_negative_indices = torch.randperm(len(negative_pairs))[:num_samples]
            
            sampled_positive_pairs = [positive_pairs[idx] for idx in sampled_positive_indices]
            sampled_negative_pairs = [negative_pairs[idx] for idx in sampled_negative_indices]
            
            if len(sampled_positive_pairs) == 0 or len(sampled_negative_pairs) == 0:
                continue
            
            # Combine samples
            eval_pairs = sampled_positive_pairs + sampled_negative_pairs
            eval_labels = [1.0] * len(sampled_positive_pairs) + [0.0] * len(sampled_negative_pairs)
            
            if len(eval_pairs) == 0:
                continue
            
            # Convert to tensors
            edge_pairs_tensor = torch.tensor(eval_pairs, device=device)
            
            # Get predictions
            predictions = model.predict_next_edges(current_embeddings, edge_pairs_tensor)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(eval_labels)
    
    if len(all_predictions) == 0:
        return {"accuracy": 0.0, "auc": 0.5, "ap": 0.0}
    
    # Calculate metrics
    predictions_np = np.array(all_predictions)
    labels_np = np.array(all_labels)
    
    accuracy = accuracy_score(labels_np, predictions_np > 0.5)
    auc = roc_auc_score(labels_np, predictions_np)
    ap = average_precision_score(labels_np, predictions_np)
    
    return {
        "accuracy": accuracy,
        "auc": auc, 
        "ap": ap,
        "num_samples": len(all_predictions)
    }


def train_graphmamba_contagion(data_name='synthetic_icm_ba', epochs=50, lr=0.001, 
                              hidden_dim=64, pos_dim=128, mamba_state_dim=16, gpu_id=0):
    """Train GraphMamba model on contagion data"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info(f"Loading {data_name} contagion dataset...")
    g_df = load_contagion_data(data_name)
    
    # Get timestamps and create sequence
    timestamps = sorted(g_df['ts'].unique())
    logger.info(f"Dataset: {len(timestamps)} timestamps, {len(g_df)} edges")
    logger.info(f"Downsampling strategy: 2:1 positive:negative ratio for balanced training")
    
    # Create graph sequence
    graph_sequence = create_contagion_graph_sequence(g_df, timestamps)
    max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1
    
    logger.info(f"Graph sequence: {len(graph_sequence)} graphs, max_nodes: {max_nodes}")
    
    # Temporal split (70% train, 15% val, 15% test)
    train_ts = int(len(timestamps) * 0.7)
    val_ts = int(len(timestamps) * 0.15)
    
    train_timestamps = timestamps[:train_ts]
    val_timestamps = timestamps[train_ts:train_ts + val_ts]
    test_timestamps = timestamps[train_ts + val_ts:]
    
    logger.info(f"Split: {len(train_timestamps)} train, {len(val_timestamps)} val, {len(test_timestamps)} test")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        logger.info(f"CUDA not available, using CPU")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = GraphMamba(
        max_nodes=max_nodes,
        pos_dim=pos_dim,
        hidden_dim=hidden_dim,
        gnn_layers=2,
        mamba_state_dim=mamba_state_dim,
        dropout=0.1
    ).to(device)
    
    # Move graph sequence to device
    graph_sequence = [adj.to(device) for adj in graph_sequence]
    
    logger.info(f"Model parameters: pos_dim={pos_dim}, hidden_dim={hidden_dim}, mamba_state_dim={mamba_state_dim}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    # Training loop
    best_val_ap = 0.0
    best_metrics = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Training on transitions within training period
        for i in range(len(train_timestamps) - 1):
            current_ts = train_timestamps[i]
            next_ts = train_timestamps[i + 1]
            
            # Recompute forward pass for each transition
            train_sequence = graph_sequence[:i + 2]  # Up to next timestamp
            sequence_embeddings = model.forward_sequence(train_sequence)
            
            # Get current embeddings
            current_embeddings = sequence_embeddings[i]
            
            # Get next timestamp edges
            next_edges = g_df[g_df['ts'] == next_ts]
            if len(next_edges) == 0:
                continue
            
            # Generate training pairs
            num_nodes = current_embeddings.shape[0]
            all_pairs = []
            for u in range(num_nodes):
                for v in range(u + 1, num_nodes):
                    all_pairs.append((u, v))
            
            # Separate positive/negative based on next timestamp
            positive_pairs = []
            negative_pairs = []
            
            for pair in all_pairs:
                u, v = pair
                edge_exists = len(next_edges[(next_edges['u'] == u) & (next_edges['i'] == v)]) > 0 or \
                            len(next_edges[(next_edges['u'] == v) & (next_edges['i'] == u)]) > 0
                
                if edge_exists:
                    positive_pairs.append(pair)
                else:
                    negative_pairs.append(pair)
            
            if len(positive_pairs) == 0:
                continue
            
            # 2:1 downsampling (positive:negative ratio)
            # If we have more positives than negatives, downsample positives
            # If we have more negatives than positives, downsample negatives
            if len(positive_pairs) > len(negative_pairs) * 2:
                # Too many positives, downsample to 2:1 ratio
                num_positives = len(negative_pairs) * 2
                sampled_positive_indices = torch.randperm(len(positive_pairs))[:num_positives]
                sampled_positives = [positive_pairs[idx] for idx in sampled_positive_indices]
                sampled_negatives = negative_pairs
            elif len(negative_pairs) > len(positive_pairs) * 2:
                # Too many negatives, downsample to 2:1 ratio
                num_negatives = len(positive_pairs) * 2
                sampled_negative_indices = torch.randperm(len(negative_pairs))[:num_negatives]
                sampled_negatives = [negative_pairs[idx] for idx in sampled_negative_indices]
                sampled_positives = positive_pairs
            else:
                # Already roughly balanced, use all
                sampled_positives = positive_pairs
                sampled_negatives = negative_pairs
            
            if len(sampled_positives) == 0 or len(sampled_negatives) == 0:
                continue
            
            # Combine and create batch
            train_pairs = sampled_positives + sampled_negatives
            train_labels = torch.tensor([1.0] * len(sampled_positives) + [0.0] * len(sampled_negatives), 
                                      device=device)
            
            edge_pairs_tensor = torch.tensor(train_pairs, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model.predict_next_edges(current_embeddings, edge_pairs_tensor)
            loss = criterion(predictions, train_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        
        # Validation
        if epoch % 10 == 0 or epoch == epochs - 1:
            val_sequence = graph_sequence[:train_ts + val_ts + 1]
            val_metrics = evaluate_contagion_prediction(
                model, val_sequence, g_df, 
                timestamps[:train_ts + val_ts], device, logger
            )
            
            logger.info(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                       f"Val Acc={val_metrics['accuracy']:.4f}, "
                       f"Val AUC={val_metrics['auc']:.4f}, "
                       f"Val AP={val_metrics['ap']:.4f}")
            
            if val_metrics['ap'] > best_val_ap:
                best_val_ap = val_metrics['ap']
                best_metrics = val_metrics.copy()
                
                # Test evaluation
                test_metrics = evaluate_contagion_prediction(
                    model, graph_sequence, g_df, timestamps, device, logger
                )
                best_metrics.update({
                    'test_accuracy': test_metrics['accuracy'],
                    'test_auc': test_metrics['auc'], 
                    'test_ap': test_metrics['ap']
                })
        else:
            logger.info(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}")
    
    # Final results
    logger.info("\n" + "="*50)
    logger.info("GRAPHMAMBA CONTAGION RESULTS")
    logger.info("="*50)
    logger.info(f"Dataset: {data_name}")
    logger.info(f"Training: 2:1 positive:negative ratio")
    logger.info(f"Evaluation: 1:1 balanced sampling")
    logger.info(f"Best Val AP: {best_val_ap:.4f}")
    logger.info(f"Test Accuracy: {best_metrics['test_accuracy']:.4f}")
    logger.info(f"Test AUC: {best_metrics['test_auc']:.4f}")
    logger.info(f"Test AP: {best_metrics['test_ap']:.4f}")
    logger.info("="*50)
    
    return model, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GraphMamba on Contagion Data')
    parser.add_argument('--data', type=str, default='synthetic_icm_ba',
                       help='Contagion dataset name (e.g., synthetic_icm_ba, synthetic_ltm_ba)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--pos_dim', type=int, default=128,
                       help='Positional embedding dimension')
    parser.add_argument('--mamba_state_dim', type=int, default=16,
                       help='Mamba state dimension')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use (default: 0)')
    
    args = parser.parse_args()
    
    # Test on the specified contagion dataset
    train_graphmamba_contagion(
        data_name=args.data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        pos_dim=args.pos_dim,
        mamba_state_dim=args.mamba_state_dim,
        gpu_id=args.gpu
    )
