"""
Training script for GIN-based Edge Convolution GNN on triadic closure datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import json
import logging
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from gin_edge_conv_gnn import GINEdgeConvGNN, EdgeGraph


def setup_logging(data_name):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'log/gin_edgeconv_gnn_{data_name}_{int(time.time())}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_triadic_data(data_name):
    """Load triadic closure dataset"""
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    e_feat = np.load(f'./processed/{data_name}/ml_{data_name}.npy')
    n_feat = np.load(f'./processed/{data_name}/ml_{data_name}_node.npy')
    
    # Load ground truth
    with open(f'./processed/{data_name}/ml_{data_name}_gt.json', 'r') as f:
        ground_truth = json.load(f)
    
    return g_df, e_feat, n_feat, ground_truth


def get_edges_at_timestamp(g_df, timestamp):
    """Get all edges up to and including given timestamp"""
    edges_df = g_df[g_df.ts <= timestamp]
    
    edge_list = []
    edge_indices = []
    
    for _, row in edges_df.iterrows():
        edge_list.append((row.u, row.i))
        edge_indices.append(row.idx)
    
    return edge_list, edge_indices


def get_triadic_candidates_at_timestamp(g_df, timestamp, all_nodes):
    """Get triadic closure candidates at given timestamp"""
    # Current edges (up to timestamp-1)
    current_edges, _ = get_edges_at_timestamp(g_df, timestamp - 1)
    
    # Build edge graph to find candidates
    edge_graph = EdgeGraph()
    edge_graph.build_from_edges(current_edges)
    
    # Find triadic candidates
    candidates = edge_graph.get_triadic_candidates(current_edges, all_nodes)
    
    return candidates


def evaluate_timestamp_predictions(model, g_df, e_feat, timestamp, all_nodes, logger, balance_ratio=1.0):
    """
    Evaluate model predictions for a specific timestamp with balanced sampling
    """
    device = next(model.parameters()).device
    
    # Get current edges (up to timestamp-1)
    current_edges, current_indices = get_edges_at_timestamp(g_df, timestamp - 1)
    
    if len(current_edges) == 0:
        return 0.5, 0.5, 0.5  # No edges to work with
    
    # Get actual new edges at this timestamp
    new_edges_df = g_df[g_df.ts == timestamp]
    actual_new_edges = set()
    
    for _, row in new_edges_df.iterrows():
        actual_new_edges.add((min(row.u, row.i), max(row.u, row.i)))
    
    if len(actual_new_edges) == 0:
        return 0.5, 0.5, 0.5  # No new edges to predict
    
    # Get triadic candidates
    candidates = get_triadic_candidates_at_timestamp(g_df, timestamp, all_nodes)
    
    if len(candidates) == 0:
        return 0.5, 0.5, 0.5  # No candidates
    
    # Separate positive and negative candidates
    positive_candidates = []
    negative_candidates = []
    
    for src, dst, shared_neighbor in candidates:
        edge_key = (min(src, dst), max(src, dst))
        if edge_key in actual_new_edges:
            positive_candidates.append((src, dst, shared_neighbor))
        else:
            negative_candidates.append((src, dst, shared_neighbor))
    
    # Balance sampling
    num_positives = len(positive_candidates)
    if num_positives == 0:
        return 0.5, 0.5, 0.5  # No positive samples
    
    num_negatives_to_sample = int(num_positives * balance_ratio)
    if num_negatives_to_sample > len(negative_candidates):
        sampled_negatives = negative_candidates
    else:
        sampled_negatives = np.random.choice(
            len(negative_candidates), num_negatives_to_sample, replace=False
        ).tolist()
        sampled_negatives = [negative_candidates[i] for i in sampled_negatives]
    
    # Combine samples
    eval_candidates = positive_candidates + sampled_negatives
    eval_labels = [1] * len(positive_candidates) + [0] * len(sampled_negatives)
    
    if len(eval_candidates) == 0:
        return 0.5, 0.5, 0.5
    
    # Get edge features for current edges
    current_edge_features = e_feat[current_indices]
    current_edge_features = torch.tensor(current_edge_features, dtype=torch.float32, device=device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        predictions, _ = model(current_edges, current_edge_features, eval_candidates)
    
    if len(predictions) == 0:
        return 0.5, 0.5, 0.5
    
    # Calculate metrics
    predictions = predictions.cpu().numpy()
    eval_labels = np.array(eval_labels)
    
    # Accuracy (threshold = 0.5)
    pred_binary = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(eval_labels, pred_binary)
    
    # AUC and AP
    if len(np.unique(eval_labels)) > 1:
        auc = roc_auc_score(eval_labels, predictions)
        ap = average_precision_score(eval_labels, predictions)
    else:
        auc = 0.5
        ap = np.mean(eval_labels)
    
    return accuracy, auc, ap


def train_gin_edge_conv_gnn(data_name='triadic_medium', num_epochs=100, learning_rate=0.001, 
                           hidden_dim=128, num_layers=3, dropout=0.1):
    """
    Train GIN-based Edge Convolution GNN on triadic closure dataset
    """
    logger = setup_logging(data_name)
    logger.info(f"Training GIN-based Edge Convolution GNN on {data_name}")
    
    # Load data
    g_df, e_feat, n_feat, ground_truth = load_triadic_data(data_name)
    logger.info(f"Dataset: {len(g_df)} edges, {len(n_feat)} nodes")
    
    # Get all nodes and timestamps
    all_nodes = set(range(len(n_feat)))
    max_timestamp = int(g_df.ts.max())
    
    # Split timestamps
    train_end = int(max_timestamp * 0.6)
    val_end = int(max_timestamp * 0.8)
    
    train_timestamps = list(range(1, train_end + 1))  # Start from 1 (need edges at ts=0)
    val_timestamps = list(range(train_end + 1, val_end + 1))
    test_timestamps = list(range(val_end + 1, max_timestamp + 1))
    
    logger.info(f"Train timestamps: {len(train_timestamps)} (ts 1-{train_end})")
    logger.info(f"Val timestamps: {len(val_timestamps)} (ts {train_end+1}-{val_end})")
    logger.info(f"Test timestamps: {len(test_timestamps)} (ts {val_end+1}-{max_timestamp})")
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = GINEdgeConvGNN(
        edge_feat_dim=e_feat.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup training with advanced techniques
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))  # Weight positive samples more
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, epochs=num_epochs, 
        steps_per_epoch=len(train_timestamps), pct_start=0.1
    )
    
    # Training loop
    best_val_ap = 0.0
    train_losses = []
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        # Shuffle training timestamps for better generalization
        train_ts_shuffled = np.random.permutation(train_timestamps)
        
        # Train on each timestamp
        for ts in train_ts_shuffled:
            # Get current edges (up to ts-1)
            current_edges, current_indices = get_edges_at_timestamp(g_df, ts - 1)
            
            if len(current_edges) == 0:
                continue
            
            # Get actual new edges at timestamp ts
            new_edges_df = g_df[g_df.ts == ts]
            actual_new_edges = set()
            
            for _, row in new_edges_df.iterrows():
                actual_new_edges.add((min(row.u, row.i), max(row.u, row.i)))
            
            if len(actual_new_edges) == 0:
                continue
            
            # Get triadic candidates
            candidates = get_triadic_candidates_at_timestamp(g_df, ts, all_nodes)
            
            if len(candidates) == 0:
                continue
            
            # Separate positive and negative candidates
            positive_candidates = []
            negative_candidates = []
            
            for src, dst, shared_neighbor in candidates:
                edge_key = (min(src, dst), max(src, dst))
                if edge_key in actual_new_edges:
                    positive_candidates.append((src, dst, shared_neighbor))
                else:
                    negative_candidates.append((src, dst, shared_neighbor))
            
            # Balance sampling (1:3 ratio for training - more negatives)
            num_positives = len(positive_candidates)
            if num_positives == 0:
                continue
            
            num_negatives_to_sample = min(num_positives * 3, len(negative_candidates))
            if num_negatives_to_sample == 0:
                continue
            
            sampled_negatives = np.random.choice(
                len(negative_candidates), num_negatives_to_sample, replace=False
            ).tolist()
            sampled_negatives = [negative_candidates[i] for i in sampled_negatives]
            
            # Combine samples
            train_candidates = positive_candidates + sampled_negatives
            train_labels = [1.0] * len(positive_candidates) + [0.0] * len(sampled_negatives)
            
            # Get edge features
            current_edge_features = e_feat[current_indices]
            current_edge_features = torch.tensor(current_edge_features, dtype=torch.float32, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            
            predictions, logits = model(current_edges, current_edge_features, train_candidates)
            
            if len(predictions) > 0:
                labels_tensor = torch.tensor(train_labels, dtype=torch.float32, device=device)
                
                # Ensure same length
                min_len = min(len(predictions), len(labels_tensor))
                predictions = predictions[:min_len]
                logits = logits[:min_len]
                labels_tensor = labels_tensor[:min_len]
                
                loss = criterion(logits, labels_tensor)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_losses.append(loss.item())
        
        if len(epoch_losses) > 0:
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                # Evaluate on validation timestamps
                val_accs, val_aucs, val_aps = [], [], []
                
                for val_ts in val_timestamps:
                    acc, auc, ap = evaluate_timestamp_predictions(
                        model, g_df, e_feat, val_ts, all_nodes, logger, balance_ratio=1.0
                    )
                    val_accs.append(acc)
                    val_aucs.append(auc)
                    val_aps.append(ap)
                
                if len(val_aps) > 0:
                    val_acc = np.mean(val_accs)
                    val_auc = np.mean(val_aucs)
                    val_ap = np.mean(val_aps)
                    
                    logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, "
                              f"Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}, Val AP={val_ap:.4f}")
                    
                    # Save best model with early stopping
                    if val_ap > best_val_ap:
                        best_val_ap = val_ap
                        patience_counter = 0
                        torch.save(model.state_dict(), f'models/gin_edgeconv_gnn_{data_name}_best.pt')
                        logger.info(f"New best model saved with Val AP: {val_ap:.4f}")
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                else:
                    logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")
        else:
            logger.warning(f"Epoch {epoch}: No valid training samples")
    
    # Final test evaluation
    logger.info("Final test evaluation...")
    
    try:
        model.load_state_dict(torch.load(f'models/gin_edgeconv_gnn_{data_name}_best.pt'))
        logger.info("Loaded best model for testing")
    except:
        logger.warning("Could not load best model, using current model")
    
    # Evaluate on test timestamps
    test_accs, test_aucs, test_aps = [], [], []
    
    for test_ts in test_timestamps:
        acc, auc, ap = evaluate_timestamp_predictions(
            model, g_df, e_feat, test_ts, all_nodes, logger, balance_ratio=1.0
        )
        test_accs.append(acc)
        test_aucs.append(auc)
        test_aps.append(ap)
    
    if len(test_aps) > 0:
        test_acc = np.mean(test_accs)
        test_auc = np.mean(test_aucs)
        test_ap = np.mean(test_aps)
        
        logger.info(f"Final Test Results:")
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
        logger.info(f"  Test AUC: {test_auc:.4f}")
        logger.info(f"  Test AP: {test_ap:.4f}")
        logger.info(f"  Best Val AP: {best_val_ap:.4f}")
        
        return {
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'test_ap': test_ap,
            'best_val_ap': best_val_ap,
            'train_losses': train_losses
        }
    else:
        logger.warning("No test results available")
        return {
            'test_accuracy': 0.0,
            'test_auc': 0.0,
            'test_ap': 0.0,
            'best_val_ap': best_val_ap,
            'train_losses': train_losses
        }


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Train GIN-based Edge Convolution GNN')
    parser.add_argument('--data', type=str, default='triadic_medium')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('log', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    print(f"Training GIN-based Edge Convolution GNN on {args.data}")
    print(f"Architecture: GIN backbone with {args.num_layers} layers")
    print(f"Features: Attention-based triadic predictor, balanced sampling, advanced optimization")
    
    # Train model
    results = train_gin_edge_conv_gnn(
        data_name=args.data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS (GIN-based Edge Convolution GNN)")
    print("="*60)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test AUC:      {results['test_auc']:.4f}")
    print(f"Test AP:       {results['test_ap']:.4f}")
    print(f"Best Val AP:   {results['best_val_ap']:.4f}")
    print("="*60)
