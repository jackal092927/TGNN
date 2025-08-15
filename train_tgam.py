"""
Training script for Temporal Graph Autoregressive Model (TGAM)

This script trains TGAM on temporal graph datasets and evaluates its performance
on both real and synthetic datasets. Unlike TGIB, TGAM learns the sequential
generation process of the graph.
"""

import math
import logging
import time
import random
import sys
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from tgam import TGAM, compute_loss
from utils import EarlyStopMonitor, RandEdgeSampler


def setup_logging(data_name):
    """Setup logging configuration"""
    Path("log").mkdir(parents=True, exist_ok=True)
    Path("saved_models").mkdir(parents=True, exist_ok=True)
    Path("saved_checkpoints").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'log/tgam_{data_name}_{int(time.time())}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def prepare_sequences(src_l, dst_l, ts_l, e_idx_l, node_features, edge_features, 
                     sequence_length=10, step_size=5):
    """
    Prepare training sequences from temporal graph data.
    
    Args:
        src_l, dst_l: source and destination nodes
        ts_l: timestamps
        e_idx_l: edge indices
        node_features: node feature matrix
        edge_features: edge feature matrix
        sequence_length: length of each training sequence
        step_size: step size for sliding window
    
    Returns:
        List of sequences, where each sequence is a list of graph states
    """
    # Sort by timestamp
    sorted_indices = np.argsort(ts_l)
    src_sorted = src_l[sorted_indices]
    dst_sorted = dst_l[sorted_indices]
    ts_sorted = ts_l[sorted_indices]
    e_idx_sorted = e_idx_l[sorted_indices]
    
    sequences = []
    
    # Create sequences using sliding window
    for start_idx in range(0, len(src_sorted) - sequence_length + 1, step_size):
        sequence = []
        
        current_edges = []
        current_edge_features = []
        
        for i in range(start_idx, start_idx + sequence_length):
            # Add current edge
            current_edges.append([src_sorted[i], dst_sorted[i]])
            current_edge_features.append(edge_features[e_idx_sorted[i]])
            
            # Create graph state
            edge_list = np.array(current_edges)
            edge_feat = np.array(current_edge_features)
            timestamp = ts_sorted[i]
            
            # Get relevant node features (all nodes that appear so far)
            nodes_in_graph = set()
            for edge in current_edges:
                nodes_in_graph.update(edge)
            
            nodes_list = sorted(list(nodes_in_graph))
            max_node = max(nodes_list) if nodes_list else 0
            
            # Create node feature matrix (zero for unseen nodes)
            node_feat = np.zeros((max_node + 1, node_features.shape[1]))
            for node in nodes_list:
                if node < len(node_features):
                    node_feat[node] = node_features[node]
            
            graph_state = (node_feat, edge_list, edge_feat, timestamp)
            sequence.append(graph_state)
        
        sequences.append(sequence)
    
    return sequences


def evaluate_link_prediction(model, test_sequences, device, num_samples=100):
    """
    Evaluate model on link prediction task.
    
    For each test sequence, use first part to predict next edges and compare
    with ground truth.
    """
    model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for sequence in test_sequences[:num_samples]:
            if len(sequence) < 3:
                continue
                
            # Use first 2/3 to predict last 1/3
            split_point = len(sequence) * 2 // 3
            input_seq = sequence[:split_point]
            target_seq = sequence[split_point:]
            
            for target_state in target_seq:
                target_node_feat, target_edge_list, target_edge_feat, target_time = target_state
                
                if len(target_edge_list) == 0:
                    continue
                
                # Get the last edge as target
                last_edge = target_edge_list[-1]
                src_target, dst_target = last_edge[0], last_edge[1]
                
                # Predict
                try:
                    src_logits, dst_logits = model(input_seq, target_time)
                    
                    if src_logits is not None and dst_logits is not None:
                        # Get probabilities
                        src_probs = torch.softmax(src_logits, dim=0)
                        dst_probs = torch.softmax(dst_logits, dim=0)
                        
                        # Compute edge probability
                        if src_target < len(src_probs) and dst_target < len(dst_probs):
                            edge_prob = (src_probs[src_target] * dst_probs[dst_target]).item()
                            predictions.append(edge_prob)
                            ground_truth.append(1.0)  # Real edge
                            
                            # Add negative sample (random edge)
                            random_dst = random.randint(0, len(dst_probs) - 1)
                            while random_dst == dst_target:
                                random_dst = random.randint(0, len(dst_probs) - 1)
                            
                            neg_prob = (src_probs[src_target] * dst_probs[random_dst]).item()
                            predictions.append(neg_prob)
                            ground_truth.append(0.0)  # Fake edge
                            
                except Exception as e:
                    continue
                
                # Update input sequence for next prediction
                input_seq.append(target_state)
    
    if len(predictions) == 0:
        return 0.5, 0.5, 0.5, 0.5
    
    # Compute metrics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Convert to binary predictions
    binary_preds = (predictions > 0.5).astype(int)
    
    acc = (binary_preds == ground_truth).mean()
    auc = roc_auc_score(ground_truth, predictions) if len(np.unique(ground_truth)) > 1 else 0.5
    ap = average_precision_score(ground_truth, predictions) if len(np.unique(ground_truth)) > 1 else 0.5
    f1 = f1_score(ground_truth, binary_preds) if len(np.unique(ground_truth)) > 1 else 0.5
    
    return acc, auc, ap, f1


def main():
    parser = argparse.ArgumentParser('TGAM Training')
    parser.add_argument('-d', '--data', type=str, help='dataset name', default='wikipedia')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--seq_len', type=int, default=10, help='sequence length')
    parser.add_argument('--step_size', type=int, default=5, help='step size for sliding window')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--num_graph_layers', type=int, default=2, help='number of graph layers')
    parser.add_argument('--num_temporal_layers', type=int, default=4, help='number of temporal layers')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.data)
    logger.info(f"Training TGAM on {args.data}")
    logger.info(args)
    
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Load data
    try:
        g_df = pd.read_csv(f'./processed/{args.data}/ml_{args.data}.csv')
        e_feat = np.load(f'./processed/{args.data}/ml_{args.data}.npy')
        n_feat = np.load(f'./processed/{args.data}/ml_{args.data}_node.npy')
        logger.info(f"Loaded data: {len(g_df)} edges, {len(n_feat)} nodes, {e_feat.shape[1]} edge features")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Data splitting
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
    
    # Training data
    train_mask = g_df.ts <= val_time
    train_data = g_df[train_mask]
    
    # Validation data
    val_mask = (g_df.ts > val_time) & (g_df.ts <= test_time)
    val_data = g_df[val_mask]
    
    # Test data
    test_mask = g_df.ts > test_time
    test_data = g_df[test_mask]
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Prepare sequences
    logger.info("Preparing training sequences...")
    train_sequences = prepare_sequences(
        train_data.u.values, train_data.i.values, train_data.ts.values, 
        train_data.idx.values, n_feat, e_feat, args.seq_len, args.step_size
    )
    
    logger.info("Preparing validation sequences...")
    val_sequences = prepare_sequences(
        val_data.u.values, val_data.i.values, val_data.ts.values,
        val_data.idx.values, n_feat, e_feat, args.seq_len, args.step_size
    )
    
    logger.info("Preparing test sequences...")
    test_sequences = prepare_sequences(
        test_data.u.values, test_data.i.values, test_data.ts.values,
        test_data.idx.values, n_feat, e_feat, args.seq_len, args.step_size
    )
    
    logger.info(f"Created {len(train_sequences)} training sequences")
    logger.info(f"Created {len(val_sequences)} validation sequences")
    logger.info(f"Created {len(test_sequences)} test sequences")
    
    # Get max node id for model initialization
    max_node_id = max(g_df.u.max(), g_df.i.max()) + 1
    
    # Initialize model
    model = TGAM(
        node_feat_dim=n_feat.shape[1],
        edge_feat_dim=e_feat.shape[1],
        hidden_dim=args.hidden_dim,
        max_nodes=max_node_id,
        num_graph_layers=args.num_graph_layers,
        num_temporal_layers=args.num_temporal_layers
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    # Early stopping
    early_stopper = EarlyStopMonitor(max_round=10)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    best_val_score = 0
    best_model_path = f'saved_models/tgam_{args.data}_best.pth'
    
    for epoch in range(1, args.n_epoch + 1):
        model.train()
        
        # Shuffle training sequences
        random.shuffle(train_sequences)
        
        epoch_loss = 0
        num_batches = 0
        
        # Create batches
        for batch_start in tqdm(range(0, len(train_sequences), args.bs), 
                               desc=f"Epoch {epoch}"):
            batch_sequences = train_sequences[batch_start:batch_start + args.bs]
            
            optimizer.zero_grad()
            
            # Compute loss for batch
            loss = compute_loss(model, batch_sequences, criterion)
            
            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        
        # Validation
        if epoch % 5 == 0:  # Evaluate every 5 epochs
            logger.info(f"Evaluating at epoch {epoch}...")
            val_acc, val_auc, val_ap, val_f1 = evaluate_link_prediction(
                model, val_sequences, device, num_samples=50
            )
            
            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, "
                       f"Val ACC={val_acc:.4f}, AUC={val_auc:.4f}, "
                       f"AP={val_ap:.4f}, F1={val_f1:.4f}")
            
            # Save best model
            if val_ap > best_val_score:
                best_val_score = val_ap
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved with AP={val_ap:.4f}")
            
            # Early stopping
            if early_stopper.early_stop_check(val_ap):
                logger.info("Early stopping triggered")
                break
        else:
            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")
    
    # Load best model for testing
    logger.info("Loading best model for testing...")
    model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation
    logger.info("Final evaluation on test set...")
    test_acc, test_auc, test_ap, test_f1 = evaluate_link_prediction(
        model, test_sequences, device, num_samples=100
    )
    
    logger.info(f"Final Test Results:")
    logger.info(f"Accuracy: {test_acc:.4f}")
    logger.info(f"AUC: {test_auc:.4f}")
    logger.info(f"Average Precision: {test_ap:.4f}")
    logger.info(f"F1-Score: {test_f1:.4f}")
    
    # Save final model
    final_model_path = f'saved_models/tgam_{args.data}_final.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save results
    results = {
        'dataset': args.data,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'test_f1': test_f1,
        'best_val_ap': best_val_score,
        'args': vars(args)
    }
    
    results_path = f'results_tgam_{args.data}_{int(time.time())}.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    main() 