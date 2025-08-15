#!/usr/bin/env python3
"""
Test Self-Explaining GraphMamba on Contagion Data
-------------------------------------------------
- Uses edge-gated spatial layer and adds sparsity + temporal smoothness losses
- Keeps the original 2:1 / 1:1 sampling scheme from your script
- Includes configurable save_dir parameter for results storage
- Separates training/evaluation from visualization
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
import matplotlib.pyplot as plt
import seaborn as sns
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


def visualize_contagion_interpretation(results_file, save_dir=None, figsize=(15, 10)):
    """
    Visualize the interpretation results from saved model predictions and gates
    
    Args:
        results_file: Path to the saved results file
        save_dir: Directory to save visualization plots (if None, uses same dir as results_file)
        figsize: Figure size for plots
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if save_dir is None:
        save_dir = os.path.dirname(results_file)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    details = results.get('details', {})
    if not details:
        print("No detailed results found for visualization")
        return
    
    predictions = np.array(details['predictions'])
    labels = np.array(details['labels'])
    pairs = details['pairs']
    timestamps = details['timestamps']
    gates = details['gates']
    
    # Create visualization plots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'GraphMamba Contagion Interpretation Results', fontsize=16)
    
    # 1. Prediction distribution
    axes[0, 0].hist(predictions[labels == 1], alpha=0.7, label='Positive', bins=30, color='green')
    axes[0, 0].hist(predictions[labels == 0], alpha=0.7, label='Negative', bins=30, color='red')
    axes[0, 0].set_xlabel('Prediction Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Prediction Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, predictions)
    axes[0, 1].plot(fpr, tpr, linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(labels, predictions)
    axes[0, 2].plot(recall, precision, linewidth=2)
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall Curve')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Gate importance over time
    unique_timestamps = sorted(list(set(timestamps)))
    gate_importance = []
    for ts in unique_timestamps:
        ts_indices = [i for i, t in enumerate(timestamps) if t == ts]
        if ts_indices:
            avg_gate = np.mean([gates[i] for i in ts_indices], axis=0)
            gate_importance.append(np.mean(avg_gate))
    
    axes[1, 0].plot(unique_timestamps, gate_importance, marker='o', linewidth=2)
    axes[1, 0].set_xlabel('Timestamp')
    axes[1, 0].set_ylabel('Average Gate Value')
    axes[1, 0].set_title('Gate Importance Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Gate sparsity analysis
    gate_values = np.concatenate(gates, axis=0)
    axes[1, 1].hist(gate_values.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 1].set_xlabel('Gate Value')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Gate Value Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Prediction vs Gate correlation
    if len(predictions) == len(gates):
        # Calculate average gate value per prediction
        avg_gates_per_pred = []
        for i, gate in enumerate(gates):
            avg_gates_per_pred.append(np.mean(gate))
        
        axes[1, 2].scatter(predictions, avg_gates_per_pred, alpha=0.6, s=20)
        axes[1, 2].set_xlabel('Prediction Score')
        axes[1, 2].set_ylabel('Average Gate Value')
        axes[1, 2].set_title('Prediction vs Gate Correlation')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(save_dir, 'contagion_interpretation_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    # Create detailed analysis report
    report_file = os.path.join(save_dir, 'contagion_interpretation_report.txt')
    with open(report_file, 'w') as f:
        f.write("GraphMamba Contagion Interpretation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {results.get('data_name', 'Unknown')}\n")
        f.write(f"Total samples: {len(predictions)}\n")
        f.write(f"Positive samples: {np.sum(labels)}\n")
        f.write(f"Negative samples: {np.sum(labels == 0)}\n")
        f.write(f"Average prediction (positive): {np.mean(predictions[labels == 1]):.4f}\n")
        f.write(f"Average prediction (negative): {np.mean(predictions[labels == 0]):.4f}\n")
        f.write(f"Gate sparsity: {np.mean(gate_values < 0.1):.4f}\n")
        f.write(f"Gate variance: {np.var(gate_values):.4f}\n")
        
        # Temporal analysis
        f.write(f"\nTemporal Analysis:\n")
        f.write(f"Number of timestamps: {len(unique_timestamps)}\n")
        f.write(f"Gate importance trend: {'Increasing' if gate_importance[-1] > gate_importance[0] else 'Decreasing'}\n")
    
    print(f"Analysis report saved to: {report_file}")
    plt.show()


def train_graphmamba_contagion(data_name='synthetic_icm_ba', epochs=100, lr=0.0001,
                               hidden_dim=64, pos_dim=128, mamba_state_dim=16, gpu_id=0,
                               lambda_sparse: float = 1e-4, lambda_tv: float = 1e-3,
                               gate_temperature: float = 1.0, save_dir: str = './results'):
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
            results_data['details'] = best_metrics['details']
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Results saved to: {results_file}")

    return model, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Self-Explaining GraphMamba on Contagion Data')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'visualize'], 
                       help='Mode: train model or visualize existing results')
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
    parser.add_argument('--results_file', type=str, help='Path to results file for visualization (required in visualize mode)')

    args = parser.parse_args()

    if args.mode == 'train':
        train_graphmamba_contagion(
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
        )
    elif args.mode == 'visualize':
        if not args.results_file:
            print("Error: --results_file is required for visualization mode")
            exit(1)
        visualize_contagion_interpretation(args.results_file, args.save_dir)
