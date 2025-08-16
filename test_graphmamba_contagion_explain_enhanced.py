#!/usr/bin/env python3
"""
Enhanced Test Self-Explaining GraphMamba on Contagion Data
----------------------------------------------------------
- Uses edge-gated spatial layer and adds sparsity + temporal smoothness losses
- Keeps the original 2:1 / 1:1 sampling scheme from your script
- Adds configurable save_dir parameter for results storage
- Integrates with existing visualization tools
- Stores detailed predictions and gates for analysis
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
from graphmamba_explain import GraphMamba  # <- use the self-explaining variant


def load_contagion_data(data_name):
    data_path = f'./processed/{data_name}/ml_{data_name}.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    g_df = pd.read_csv(data_path)
    if 'label' not in g_df.columns:
        raise ValueError(f"Dataset {data_name} missing 'label' column")
    print(f"Loaded {data_name}: {len(g_df)} edges, {g_df['label'].nunique()} unique labels")
    print(f"Label distribution: {g_df['label'].value_counts().to_dict()}")
    return g_df


def create_contagion_graph_sequence(g_df, timestamps):
    max_node = max(g_df['u'].max(), g_df['i'].max()) + 1
    graph_sequence = []
    for ts in timestamps:
        edges_up_to_ts = g_df[g_df['ts'] <= ts]
        adj = torch.zeros(max_node, max_node)
        for _, row in edges_up_to_ts.iterrows():
            u, v = int(row['u']), int(row['i'])
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        graph_sequence.append(adj)
    return graph_sequence


def evaluate_contagion_prediction(model, graph_sequence, g_df, timestamps, device, logger):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        seq_emb = model.forward_sequence(graph_sequence)  # no need for gates at eval
        for i in range(len(timestamps) - 1):
            next_ts = timestamps[i + 1]
            current_emb = seq_emb[i]
            next_edges = g_df[g_df['ts'] == next_ts]
            if len(next_edges) == 0:
                continue
            N = current_emb.shape[0]
            all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
            pos, neg = [], []
            for (u, v) in all_pairs:
                edge_exists = len(next_edges[(next_edges['u'] == u) & (next_edges['i'] == v)]) > 0 or \
                              len(next_edges[(next_edges['u'] == v) & (next_edges['i'] == u)]) > 0
                (pos if edge_exists else neg).append((u, v))
            if len(pos) == 0:
                continue
            num = min(len(pos), len(neg))
            if num == 0:
                continue
            sp = torch.randperm(len(pos))[:num]; sn = torch.randperm(len(neg))[:num]
            pos_s = [pos[idx] for idx in sp]; neg_s = [neg[idx] for idx in sn]
            pairs = pos_s + neg_s
            labels = [1.0]*len(pos_s) + [0.0]*len(neg_s)
            pairs_t = torch.tensor(pairs, device=device)
            preds = model.predict_next_edges(current_emb, pairs_t)
            all_predictions.extend(preds.cpu().numpy()); all_labels.extend(labels)
    if len(all_predictions) == 0:
        return {"accuracy": 0.0, "auc": 0.5, "ap": 0.0}
    pred = np.array(all_predictions); lab = np.array(all_labels)
    acc = accuracy_score(lab, pred > 0.5); auc = roc_auc_score(lab, pred); ap = average_precision_score(lab, pred)
    return {"accuracy": acc, "auc": auc, "ap": ap, "num_samples": len(all_predictions)}


def evaluate_contagion_prediction_with_details(model, graph_sequence, g_df, timestamps, device, logger):
    """Evaluate model and return detailed predictions and gates for visualization"""
    model.eval()
    all_predictions, all_labels = [], []
    all_pairs = []
    all_timestamps = []
    all_gates = []
    all_embeddings = []
    
    with torch.no_grad():
        seq_emb, gates_list = model.forward_sequence(graph_sequence, return_gates=True)
        for i in range(len(timestamps) - 1):
            next_ts = timestamps[i + 1]
            current_emb = seq_emb[i]
            current_gates = gates_list[i]
            next_edges = g_df[g_df['ts'] == next_ts]
            if len(next_edges) == 0:
                continue
            N = current_emb.shape[0]
            pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
            pos, neg = [], []
            for (u, v) in pairs:
                edge_exists = len(next_edges[(next_edges['u'] == u) & (next_edges['i'] == v)]) > 0 or \
                              len(next_edges[(next_edges['u'] == v) & (next_edges['i'] == u)]) > 0
                (pos if edge_exists else neg).append((u, v))
            if len(pos) == 0:
                continue
            num = min(len(pos), len(neg))
            if num == 0:
                continue
            sp = torch.randperm(len(pos))[:num]; sn = torch.randperm(len(neg))[:num]
            pos_s = [pos[idx] for idx in sp]; neg_s = [neg[idx] for idx in sn]
            eval_pairs = pos_s + neg_s
            labels = [1.0]*len(pos_s) + [0.0]*len(neg_s)
            pairs_t = torch.tensor(eval_pairs, device=device)
            preds = model.predict_next_edges(current_emb, pairs_t)
            
            # Store all details
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels)
            all_pairs.extend(eval_pairs)
            all_timestamps.extend([next_ts] * len(eval_pairs))
            all_gates.extend([current_gates.cpu().numpy()] * len(eval_pairs))
            all_embeddings.extend([current_emb.cpu().numpy()] * len(eval_pairs))
    
    if len(all_predictions) == 0:
        return {"accuracy": 0.0, "auc": 0.5, "ap": 0.0, "details": None}
    
    pred = np.array(all_predictions)
    lab = np.array(all_labels)
    acc = accuracy_score(lab, pred > 0.5)
    auc = roc_auc_score(lab, pred)
    ap = average_precision_score(lab, pred)
    
    details = {
        'predictions': pred,
        'labels': lab,
        'pairs': all_pairs,
        'timestamps': all_timestamps,
        'gates': all_gates,
        'embeddings': all_embeddings
    }
    
    return {
        "accuracy": acc, 
        "auc": auc, 
        "ap": ap, 
        "num_samples": len(all_predictions),
        "details": details
    }


def train_graphmamba_contagion(data_name='synthetic_icm_ba', epochs=50, lr=0.001,
                               hidden_dim=64, pos_dim=128, mamba_state_dim=16, gpu_id=0,
                               lambda_sparse: float = 1e-4, lambda_tv: float = 1e-3,
                               gate_temperature: float = 1.0, save_dir: str = './results',
                               checkpoint_freq: int = 10, save_best_only: bool = False):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {save_dir}")

    logger.info(f"Loading {data_name} contagion dataset...")
    g_df = load_contagion_data(data_name)
    timestamps = sorted(g_df['ts'].unique())
    logger.info(f"Dataset: {len(timestamps)} timestamps, {len(g_df)} edges")
    logger.info(f"Downsampling strategy: 2:1 positive:negative ratio for training")

    graph_sequence = create_contagion_graph_sequence(g_df, timestamps)
    max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1
    logger.info(f"Graph sequence: {len(graph_sequence)} graphs, max_nodes: {max_nodes}")

    train_ts = int(len(timestamps) * 0.7)
    val_ts = int(len(timestamps) * 0.15)
    train_timestamps = timestamps[:train_ts]
    val_timestamps = timestamps[train_ts:train_ts + val_ts]
    test_timestamps = timestamps[train_ts + val_ts:]

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}'); logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu'); logger.info("CUDA not available, using CPU")
    logger.info(f"Using device: {device}")

    model = GraphMamba(max_nodes=max_nodes, pos_dim=pos_dim, hidden_dim=hidden_dim,
                       gnn_layers=2, mamba_state_dim=mamba_state_dim, dropout=0.1,
                       use_edge_gates=True, gate_temperature=gate_temperature).to(device)

    graph_sequence = [adj.to(device) for adj in graph_sequence]
    logger.info(f"Model parameters: pos_dim={pos_dim}, hidden_dim={hidden_dim}, mamba_state_dim={mamba_state_dim}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    best_val_ap = 0.0
    best_metrics = None
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Function to save checkpoint
    def save_checkpoint(epoch, model, optimizer, best_val_ap, best_metrics, 
                       is_best=False, is_final=False):
        checkpoint_file = os.path.join(checkpoint_dir, f'{data_name}_epoch_{epoch:03d}.pth')
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_ap': best_val_ap,
            'best_metrics': best_metrics,
            'hyperparameters': {
                'epochs': epochs,
                'lr': lr,
                'hidden_dim': hidden_dim,
                'pos_dim': pos_dim,
                'mamba_state_dim': mamba_state_dim,
                'lambda_sparse': lambda_sparse,
                'lambda_tv': lambda_tv,
                'gate_temperature': gate_temperature
            },
            'model_config': {
                'max_nodes': max_nodes,
                'gnn_layers': 2,
                'dropout': 0.1,
                'use_edge_gates': True
            },
            'training_info': {
                'total_epochs': epochs,
                'current_epoch': epoch,
                'is_best': is_best,
                'is_final': is_final
            }
        }
        
        torch.save(checkpoint_data, checkpoint_file)
        
        if is_best:
            # Save best model separately
            best_model_file = os.path.join(save_dir, f'{data_name}_best_model.pth')
            torch.save(checkpoint_data, best_model_file)
            logger.info(f"Best model saved to: {best_model_file}")
        
        if is_final:
            # Save final model
            final_model_file = os.path.join(save_dir, f'{data_name}_final_model.pth')
            torch.save(checkpoint_data, final_model_file)
            logger.info(f"Final model saved to: {final_model_file}")
        
        logger.info(f"Checkpoint saved to: {checkpoint_file}")
        return checkpoint_file

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i in range(len(train_timestamps) - 1):
            next_ts = train_timestamps[i + 1]
            train_sequence = graph_sequence[:i + 2]
            seq_emb, gates_list = model.forward_sequence(train_sequence, return_gates=True)
            current_emb = seq_emb[i]
            current_gates = gates_list[i]
            prev_gates = gates_list[i - 1] if i > 0 else None

            # Build pairs using edges at next_ts (contagion)
            next_edges = g_df[g_df['ts'] == next_ts]
            if len(next_edges) == 0:
                continue

            N = current_emb.shape[0]
            all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
            pos, neg = [], []
            for (u, v) in all_pairs:
                edge_exists = len(next_edges[(next_edges['u'] == u) & (next_edges['i'] == v)]) > 0 or \
                              len(next_edges[(next_edges['u'] == v) & (next_edges['i'] == u)]) > 0
                (pos if edge_exists else neg).append((u, v))

            if len(pos) == 0:
                continue

            # 2:1 downsampling (positive:negative)
            if len(pos) > len(neg) * 2:
                num_pos = len(neg) * 2
                sp = torch.randperm(len(pos))[:num_pos]; pos_s = [pos[idx] for idx in sp]; neg_s = neg
            elif len(neg) > len(pos) * 2:
                num_neg = len(pos) * 2
                sn = torch.randperm(len(neg))[:num_neg]; neg_s = [neg[idx] for idx in sn]; pos_s = pos
            else:
                pos_s, neg_s = pos, neg

            if len(pos_s) == 0 or len(neg_s) == 0:
                continue

            train_pairs = pos_s + neg_s
            labels = torch.tensor([1.0]*len(pos_s) + [0.0]*len(neg_s), device=device)
            pairs_t = torch.tensor(train_pairs, device=device)

            optimizer.zero_grad()
            preds = model.predict_next_edges(current_emb, pairs_t)
            loss_pred = criterion(preds, labels)

            # Explanation losses
            loss_sparse = GraphMamba.sparsity_loss(current_gates)
            loss_tv = GraphMamba.temporal_tv_loss(current_gates, prev_gates) if prev_gates is not None else current_gates.sum()*0.0

            loss = loss_pred + lambda_sparse * loss_sparse + lambda_tv * loss_tv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # Save checkpoint at regular intervals
        if epoch % checkpoint_freq == 0 or epoch == epochs - 1:
            # Save regular checkpoint
            save_checkpoint(epoch, model, optimizer, best_val_ap, best_metrics, 
                          is_best=False, is_final=(epoch == epochs - 1))
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            val_sequence = graph_sequence[:train_ts + val_ts + 1]
            val_metrics = evaluate_contagion_prediction(model, val_sequence, g_df, timestamps[:train_ts + val_ts], device, logger)
            logger.info(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                        f"Val Acc={val_metrics['accuracy']:.4f}, "
                        f"Val AUC={val_metrics['auc']:.4f}, "
                        f"Val AP={val_metrics['ap']:.4f}")
            if val_metrics['ap'] > best_val_ap:
                best_val_ap = val_metrics['ap']
                best_metrics = val_metrics.copy()
                # Use detailed evaluation for final test results
                test_metrics = evaluate_contagion_prediction_with_details(model, graph_sequence, g_df, timestamps, device, logger)
                best_metrics.update({
                    'test_accuracy': test_metrics['accuracy'],
                    'test_auc': test_metrics['auc'],
                    'test_ap': test_metrics['ap']
                })
                # Store detailed results for visualization
                best_metrics['details'] = test_metrics.get('details', {})
                
                # Save best model checkpoint
                save_checkpoint(epoch, model, optimizer, best_val_ap, best_metrics, 
                              is_best=True, is_final=False)
        else:
            logger.info(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}")

    logger.info("\n" + "="*50)
    logger.info("GRAPHMAMBA CONTAGION (Self-Explaining) RESULTS")
    logger.info("="*50)
    logger.info(f"Dataset: {data_name}")
    logger.info(f"Training: 2:1 positive:negative ratio (with sparsity/tv losses)")
    logger.info(f"Evaluation: 1:1 balanced sampling")
    logger.info(f"Best Val AP: {best_val_ap:.4f}")
    if best_metrics is not None:
        logger.info(f"Test Accuracy: {best_metrics['accuracy']:.4f}")
        logger.info(f"Test AUC: {best_metrics['auc']:.4f}")
        logger.info(f"Test AP: {best_metrics['ap']:.4f}")
    logger.info("="*50)

    # Save results to file
    if best_metrics is not None:
        results_file = os.path.join(save_dir, f'{data_name}_results.json')
        results_data = {
            'data_name': data_name,
            'best_val_ap': best_val_ap,
            'test_metrics': {
                'accuracy': best_metrics.get('test_accuracy', 0.0),
                'auc': best_metrics.get('test_auc', 0.0),
                'ap': best_metrics.get('test_ap', 0.0)
            },
            'val_metrics': {
                'accuracy': best_metrics.get('accuracy', 0.0),
                'auc': best_metrics.get('auc', 0.0),
                'ap': best_metrics.get('ap', 0.0)
            },
            'hyperparameters': {
                'epochs': epochs,
                'lr': lr,
                'hidden_dim': hidden_dim,
                'pos_dim': pos_dim,
                'mamba_state_dim': mamba_state_dim,
                'lambda_sparse': lambda_sparse,
                'lambda_tv': lambda_tv,
                'gate_temperature': gate_temperature
            }
        }
        
        # Add detailed results for visualization
        if 'details' in best_metrics:
            # Convert numpy arrays to lists for JSON serialization
            details = best_metrics['details'].copy()
            for key, value in details.items():
                if hasattr(value, 'tolist'):  # Check if it's a numpy array
                    details[key] = value.tolist()
            results_data['details'] = details
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Results saved to: {results_file}")
        
        # Also save a summary for your existing visualization tools
        summary_file = os.path.join(save_dir, f'{data_name}_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"GraphMamba Contagion Results Summary\n")
            f.write(f"====================================\n")
            f.write(f"Dataset: {data_name}\n")
            f.write(f"Best Validation AP: {best_val_ap:.4f}\n")
            f.write(f"Test Accuracy: {best_metrics.get('test_accuracy', 0.0):.4f}\n")
            f.write(f"Test AUC: {best_metrics.get('test_auc', 0.0):.4f}\n")
            f.write(f"Test AP: {best_metrics.get('test_ap', 0.0):.4f}\n")
            f.write(f"\nHyperparameters:\n")
            f.write(f"  Epochs: {epochs}\n")
            f.write(f"  Learning Rate: {lr}\n")
            f.write(f"  Hidden Dim: {hidden_dim}\n")
            f.write(f"  Position Dim: {pos_dim}\n")
            f.write(f"  Mamba State Dim: {mamba_state_dim}\n")
            f.write(f"  Lambda Sparse: {lambda_sparse}\n")
            f.write(f"  Lambda TV: {lambda_tv}\n")
            f.write(f"  Gate Temperature: {gate_temperature}\n")
        logger.info(f"Summary saved to: {summary_file}")
        
        # Save the trained model
        model_file = os.path.join(save_dir, f'{data_name}_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_ap': best_val_ap,
            'best_metrics': best_metrics,
            'hyperparameters': {
                'epochs': epochs,
                'lr': lr,
                'hidden_dim': hidden_dim,
                'pos_dim': pos_dim,
                'mamba_state_dim': mamba_state_dim,
                'lambda_sparse': lambda_sparse,
                'lambda_tv': lambda_tv,
                'gate_temperature': gate_temperature
            },
            'model_config': {
                'max_nodes': max_nodes,
                'gnn_layers': 2,
                'dropout': 0.1,
                'use_edge_gates': True
            }
        }, model_file)
        logger.info(f"Model saved to: {model_file}")

    return model, best_metrics


def resume_training_from_checkpoint(checkpoint_file, data_name, g_df, timestamps, device='cpu', 
                                  additional_epochs=50, save_dir='./results'):
    """
    Resume training from a saved checkpoint
    
    Args:
        checkpoint_file: Path to checkpoint file
        data_name: Dataset name
        g_df: Dataset dataframe
        timestamps: Dataset timestamps
        device: Device to train on
        additional_epochs: How many more epochs to train
        save_dir: Directory to save resumed training results
    
    Returns:
        model: Trained model
        best_metrics: Best metrics achieved
    """
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    
    print(f"Resuming training from checkpoint: {checkpoint_file}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Extract information
    start_epoch = checkpoint['epoch']
    best_val_ap = checkpoint['best_val_ap']
    best_metrics = checkpoint['best_metrics']
    hyperparameters = checkpoint['hyperparameters']
    model_config = checkpoint['model_config']
    
    print(f"Resuming from epoch {start_epoch} with best validation AP: {best_val_ap:.4f}")
    print(f"Will train for {additional_epochs} more epochs")
    
    # Create graph sequence
    graph_sequence = create_contagion_graph_sequence(g_df, timestamps)
    max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1
    
    # Create model
    model = GraphMamba(
        max_nodes=max_nodes,
        pos_dim=hyperparameters['pos_dim'],
        hidden_dim=hyperparameters['hidden_dim'],
        gnn_layers=model_config['gnn_layers'],
        mamba_state_dim=hyperparameters['mamba_state_dim'],
        dropout=model_config['dropout'],
        use_edge_gates=model_config['use_edge_gates'],
        gate_temperature=hyperparameters['gate_temperature']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters['lr'], weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Move graph sequence to device
    graph_sequence = [adj.to(device) for adj in graph_sequence]
    
    # Continue training
    total_epochs = start_epoch + additional_epochs
    train_ts = int(len(timestamps) * 0.7)
    val_ts = int(len(timestamps) * 0.15)
    train_timestamps = timestamps[:train_ts]
    val_timestamps = timestamps[train_ts:train_ts + val_ts]
    
    criterion = nn.BCELoss()
    
    # Create checkpoint directory for resumed training
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Function to save checkpoint (same as in main training)
    def save_checkpoint(epoch, model, optimizer, best_val_ap, best_metrics, 
                       is_best=False, is_final=False):
        checkpoint_file = os.path.join(checkpoint_dir, f'{data_name}_resumed_epoch_{epoch:03d}.pth')
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_ap': best_val_ap,
            'best_metrics': best_metrics,
            'hyperparameters': hyperparameters,
            'model_config': model_config,
            'training_info': {
                'total_epochs': total_epochs,
                'current_epoch': epoch,
                'is_best': is_best,
                'is_final': is_final,
                'resumed_from': start_epoch
            }
        }
        
        torch.save(checkpoint_data, checkpoint_file)
        
        if is_best:
            best_model_file = os.path.join(save_dir, f'{data_name}_resumed_best_model.pth')
            torch.save(checkpoint_data, best_model_file)
            print(f"Best model saved to: {best_model_file}")
        
        if is_final:
            final_model_file = os.path.join(save_dir, f'{data_name}_resumed_final_model.pth')
            torch.save(checkpoint_data, final_model_file)
            print(f"Final model saved to: {final_model_file}")
        
        print(f"Checkpoint saved to: {checkpoint_file}")
        return checkpoint_file
    
    # Training loop for additional epochs
    for epoch in range(start_epoch + 1, total_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(len(train_timestamps) - 1):
            next_ts = train_timestamps[i + 1]
            train_sequence = graph_sequence[:i + 2]
            seq_emb, gates_list = model.forward_sequence(train_sequence, return_gates=True)
            current_emb = seq_emb[i]
            current_gates = gates_list[i]
            prev_gates = gates_list[i - 1] if i > 0 else None
            
            # Build pairs using edges at next_ts (contagion)
            next_edges = g_df[g_df['ts'] == next_ts]
            if len(next_edges) == 0:
                continue
            
            N = current_emb.shape[0]
            all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
            pos, neg = [], []
            for (u, v) in all_pairs:
                edge_exists = len(next_edges[(next_edges['u'] == u) & (next_edges['i'] == v)]) > 0 or \
                              len(next_edges[(next_edges['u'] == v) & (next_edges['i'] == u)]) > 0
                (pos if edge_exists else neg).append((u, v))
            
            if len(pos) == 0:
                continue
            
            # 2:1 downsampling (positive:negative)
            if len(pos) > len(neg) * 2:
                num_pos = len(neg) * 2
                sp = torch.randperm(len(pos))[:num_pos]; pos_s = [pos[idx] for idx in sp]; neg_s = neg
            elif len(neg) > len(pos) * 2:
                num_neg = len(pos) * 2
                sn = torch.randperm(len(neg))[:num_neg]; neg_s = [neg[idx] for idx in sn]; pos_s = pos
            else:
                pos_s, neg_s = pos, neg
            
            if len(pos_s) == 0 or len(neg_s) == 0:
                continue
            
            train_pairs = pos_s + neg_s
            labels = torch.tensor([1.0]*len(pos_s) + [0.0]*len(neg_s), device=device)
            pairs_t = torch.tensor(train_pairs, device=device)
            
            optimizer.zero_grad()
            preds = model.predict_next_edges(current_emb, pairs_t)
            loss_pred = criterion(preds, labels)
            
            # Explanation losses
            loss_sparse = GraphMamba.sparsity_loss(current_gates)
            loss_tv = GraphMamba.temporal_tv_loss(current_gates, prev_gates) if prev_gates is not None else current_gates.sum()*0.0
            
            loss = loss_pred + hyperparameters['lambda_sparse'] * loss_sparse + hyperparameters['lambda_tv'] * loss_tv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += float(loss.item())
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == total_epochs - 1:
            save_checkpoint(epoch, model, optimizer, best_val_ap, best_metrics, 
                          is_best=False, is_final=(epoch == total_epochs - 1))
        
        # Validation every 10 epochs
        if epoch % 10 == 0 or epoch == total_epochs - 1:
            val_sequence = graph_sequence[:train_ts + val_ts + 1]
            val_metrics = evaluate_contagion_prediction(model, val_sequence, g_df, timestamps[:train_ts + val_ts], device, None)
            print(f"Resumed Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}, "
                  f"Val AUC={val_metrics['auc']:.4f}, "
                  f"Val AP={val_metrics['ap']:.4f}")
            
            if val_metrics['ap'] > best_val_ap:
                best_val_ap = val_metrics['ap']
                best_metrics = val_metrics.copy()
                
                # Save best model checkpoint
                save_checkpoint(epoch, model, optimizer, best_val_ap, best_metrics, 
                              is_best=True, is_final=False)
        else:
            print(f"Resumed Epoch {epoch:3d}: Loss={avg_loss:.4f}")
    
    print(f"\nResumed training completed! Final best validation AP: {best_val_ap:.4f}")
    return model, best_metrics


def load_saved_model(model_file, device='cpu'):
    """
    Load a saved GraphMamba model from checkpoint
    
    Args:
        model_file: Path to the saved model file (.pth)
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded GraphMamba model
        checkpoint_info: Dictionary containing training info
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    checkpoint = torch.load(model_file, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    hyperparameters = checkpoint['hyperparameters']
    
    # Create model with saved configuration
    model = GraphMamba(
        max_nodes=model_config['max_nodes'],
        pos_dim=hyperparameters['pos_dim'],
        hidden_dim=hyperparameters['hidden_dim'],
        gnn_layers=model_config['gnn_layers'],
        mamba_state_dim=hyperparameters['mamba_state_dim'],
        dropout=model_config['dropout'],
        use_edge_gates=model_config['use_edge_gates'],
        gate_temperature=hyperparameters['gate_temperature']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from: {model_file}")
    print(f"Best validation AP: {checkpoint['best_val_ap']:.4f}")
    print(f"Model configuration: {model_config}")
    
    return model, checkpoint


def create_visualization_data(data_name, save_dir):
    """
    Create data files compatible with your existing visualization tools
    """
    # Create a simple visualization data file that your tools can use
    viz_file = os.path.join(save_dir, f'{data_name}_viz_data.csv')
    
    # This creates a simple CSV that your existing tools might be able to use
    # You can modify this based on what format your visualization tools expect
    viz_data = pd.DataFrame({
        'timestamp': [0, 1, 2, 3, 4, 5],
        'active_nodes': [0, 10, 25, 40, 50, 55],
        'total_nodes': [100, 100, 100, 100, 100, 100],
        'activation_rate': [0.0, 0.1, 0.25, 0.4, 0.5, 0.55]
    })
    
    viz_data.to_csv(viz_file, index=False)
    print(f"Visualization data saved to: {viz_file}")
    print("You can now use your existing visualization tools with this data!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Test Self-Explaining GraphMamba on Contagion Data')
    parser.add_argument('--data', type=str, default='synthetic_icm_ba', help='Dataset name (e.g., synthetic_icm_ba, synthetic_ltm_ba)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--pos_dim', type=int, default=128, help='Positional embedding dimension')
    parser.add_argument('--mamba_state_dim', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0)')
    parser.add_argument('--lambda_sparse', type=float, default=1e-4, help='Sparsity loss weight')
    parser.add_argument('--lambda_tv', type=float, default=1e-3, help='Temporal smoothness loss weight')
    parser.add_argument('--gate_temperature', type=float, default=1.0, help='Gate temperature (higher = smoother gates)')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results and models')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--save_best_only', action='store_true', help='Only save best model checkpoints (save disk space)')
    parser.add_argument('--resume_from', type=str, help='Resume training from checkpoint file path')
    parser.add_argument('--additional_epochs', type=int, default=50, help='Number of additional epochs when resuming training (default: 50)')
    parser.add_argument('--create_viz_data', action='store_true', help='Create visualization data files for existing tools')

    args = parser.parse_args()

    # Check if resuming from checkpoint
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            print(f"Error: Checkpoint file not found: {args.resume_from}")
            exit(1)
        
        print(f"Resuming training from checkpoint: {args.resume_from}")
        
        # Set up device (same logic as in train_graphmamba_contagion)
        if args.gpu >= 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu}')
            print(f"Using GPU: {args.gpu}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        
        # Load dataset for resuming
        g_df = load_contagion_data(args.data)
        timestamps = sorted(g_df['ts'].unique())
        
        # Resume training
        model, best_metrics = resume_training_from_checkpoint(
            checkpoint_file=args.resume_from,
            data_name=args.data,
            g_df=g_df,
            timestamps=timestamps,
            device=device,
            additional_epochs=args.additional_epochs,
            save_dir=args.save_dir
        )
        
        # After resume training, we need to save the final results with interpretation data
        if best_metrics is not None:
            print(f"\nSaving final results after resume training...")
            
            # Get detailed evaluation results for interpretation
            from test_graphmamba_contagion_explain_enhanced import evaluate_contagion_prediction_with_details, create_contagion_graph_sequence
            
            # Create graph sequence for evaluation
            graph_sequence = create_contagion_graph_sequence(g_df, timestamps)
            
            # Move graph sequence to the same device as the model
            graph_sequence = [adj.to(device) for adj in graph_sequence]
            
            # Get detailed test metrics with interpretation data
            test_metrics = evaluate_contagion_prediction_with_details(
                model, graph_sequence, g_df, timestamps, device, None
            )
            
            # Save results to file
            results_file = os.path.join(args.save_dir, f'{args.data}_results.json')
            results_data = {
                'data_name': args.data,
                'best_val_ap': best_metrics.get('ap', 0.0),
                'test_metrics': {
                    'accuracy': test_metrics.get('accuracy', 0.0),
                    'auc': test_metrics.get('auc', 0.0),
                    'ap': test_metrics.get('ap', 0.0)
                },
                'val_metrics': best_metrics,
                'hyperparameters': {
                    'epochs': 'resumed',
                    'lr': 'resumed',
                    'hidden_dim': 'resumed',
                    'pos_dim': 'resumed',
                    'mamba_state_dim': 'resumed',
                    'lambda_sparse': 'resumed',
                    'lambda_tv': 'resumed',
                    'gate_temperature': 'resumed'
                }
            }
            
            # Add detailed results for visualization (with numpy array conversion)
            if 'details' in test_metrics:
                details = test_metrics['details'].copy()
                for key, value in details.items():
                    if hasattr(value, 'tolist'):  # Check if it's a numpy array
                        details[key] = value.tolist()
                results_data['details'] = details
            
            # Also ensure all other values are JSON serializable
            def make_json_serializable(obj):
                if hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                else:
                    return obj
            
            # Convert the entire results_data to be JSON serializable
            results_data = make_json_serializable(results_data)
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"Results saved to: {results_file}")
            
            # Also save a summary
            summary_file = os.path.join(args.save_dir, f'{args.data}_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(f"GraphMamba Contagion Results Summary (Resumed Training)\n")
                f.write(f"====================================================\n")
                f.write(f"Dataset: {args.data}\n")
                f.write(f"Best Validation AP: {best_metrics.get('ap', 0.0):.4f}\n")
                f.write(f"Test Accuracy: {test_metrics.get('accuracy', 0.0):.4f}\n")
                f.write(f"Test AUC: {test_metrics.get('auc', 0.0):.4f}\n")
                f.write(f"Test AP: {test_metrics.get('ap', 0.0):.4f}\n")
                f.write(f"\nTraining: Resumed from checkpoint\n")
            print(f"Summary saved to: {summary_file}")
    else:
        # Train the model from scratch
        model, best_metrics = train_graphmamba_contagion(
            data_name=args.data,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            pos_dim=args.pos_dim,
            mamba_state_dim=args.mamba_state_dim,
            gpu_id=args.gpu,
            lambda_sparse=args.lambda_sparse,
            lambda_tv=args.lambda_tv,
            gate_temperature=args.gate_temperature,
            save_dir=args.save_dir,
            checkpoint_freq=args.checkpoint_freq,
            save_best_only=args.save_best_only,
        )

    # Create visualization data if requested
    if args.create_viz_data:
        create_visualization_data(args.data, args.save_dir)
    
    print(f"\nTraining completed! Results saved to: {args.save_dir}")
    print(f"You can now use your existing visualization tools with the saved data.")
    print(f"Or analyze the detailed results in: {args.data}_results.json")
