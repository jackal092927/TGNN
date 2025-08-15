"""
Flexible Training Script for TGAM: Autoregressive vs Hybrid with Optional Teacher Forcing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from itertools import combinations
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from tgam_fixed import TGAM_LinkPrediction
from utils import RandEdgeSampler
from collections import defaultdict

class TGAMTrainer:
    """Flexible trainer supporting multiple paradigms"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # Training mode configurations
        self.training_mode = config.get('training_mode', 'individual')  # 'autoregressive', 'individual', 'hybrid'
        self.use_teacher_forcing = config.get('teacher_forcing', True)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 1.0)  # Prob of using ground truth
        
        # Training parameters
        self.lr = config.get('lr', 0.001)
        self.num_epochs = config.get('epochs', 20)
        self.steps_per_epoch = config.get('steps_per_epoch', 50)
        
        print(f"=== Training Configuration ===")
        print(f"Training Mode: {self.training_mode}")
        print(f"Teacher Forcing: {self.use_teacher_forcing}")
        print(f"Teacher Forcing Ratio: {self.teacher_forcing_ratio}")
        print(f"Learning Rate: {self.lr}")
        print(f"Epochs: {self.num_epochs}")
        
    def get_edge_candidates_at_timestamp(self, graph_state, timestamp, max_nodes):
        """Get all possible edge candidates at a given timestamp"""
        # For autoregressive: predict ALL possible edges at this timestamp
        existing_edges = set()
        for i, (src, dst, ts) in enumerate(graph_state):
            if ts < timestamp:
                existing_edges.add((min(src, dst), max(src, dst)))
        
        # Generate all possible new edges
        all_candidates = []
        nodes_at_ts = set()
        for src, dst, ts in graph_state:
            if ts < timestamp:
                nodes_at_ts.add(src)
                nodes_at_ts.add(dst)
        
        if len(nodes_at_ts) < 2:
            return []
            
        for src in nodes_at_ts:
            for dst in nodes_at_ts:
                if src != dst:
                    edge_key = (min(src, dst), max(src, dst))
                    if edge_key not in existing_edges:
                        all_candidates.append((src, dst))
        
        return all_candidates
    
    def autoregressive_loss(self, train_src_l, train_dst_l, train_ts_l, train_e_idx_l, 
                           n_feat, e_feat, step_range):
        """
        Compute autoregressive loss: predict ALL edges at each timestamp
        """
        total_loss = 0.0
        num_predictions = 0
        correct_predictions = 0
        
        # Group edges by timestamp
        timestamp_groups = defaultdict(list)
        for i, ts in enumerate(train_ts_l):
            if i in step_range:
                timestamp_groups[ts].append(i)
        
        unique_timestamps = sorted([ts for ts in timestamp_groups.keys() if ts > 0])
        
        for ts_idx, current_ts in enumerate(unique_timestamps):
            if current_ts == 0:
                continue  # Skip initial state
                
            # Get history up to (but not including) current timestamp
            history_mask = train_ts_l < current_ts
            if not np.any(history_mask):
                continue
                
            history_indices = np.where(history_mask)[0]
            if len(history_indices) == 0:
                continue
                
            # Ground truth edges at current timestamp
            gt_edges_indices = timestamp_groups[current_ts]
            gt_edges = [(train_src_l[i], train_dst_l[i]) for i in gt_edges_indices]
            gt_edge_set = set(gt_edges)
            
            # Get all possible edge candidates
            max_nodes = max(max(train_src_l), max(train_dst_l)) + 1
            graph_state = [(train_src_l[i], train_dst_l[i], train_ts_l[i]) for i in history_indices]
            candidates = self.get_edge_candidates_at_timestamp(graph_state, current_ts, max_nodes)
            
            if len(candidates) == 0:
                continue
            
            # Limit candidates for computational efficiency
            max_candidates = min(50, len(candidates))
            if len(candidates) > max_candidates:
                # Include all ground truth edges + random sample of others
                gt_candidates = [c for c in candidates if c in gt_edge_set or (c[1], c[0]) in gt_edge_set]
                other_candidates = [c for c in candidates if c not in gt_edge_set and (c[1], c[0]) not in gt_edge_set]
                remaining_slots = max_candidates - len(gt_candidates)
                if remaining_slots > 0 and other_candidates:
                    other_candidates = np.random.choice(len(other_candidates), 
                                                      min(remaining_slots, len(other_candidates)), 
                                                      replace=False)
                    other_candidates = [candidates[i] for i in other_candidates]
                else:
                    other_candidates = []
                candidates = gt_candidates + other_candidates
            
            # Predict each candidate edge
            candidate_probs = []
            candidate_labels = []
            
            for src, dst in candidates:
                # Determine if this edge should exist (ground truth)
                label = 1 if (src, dst) in gt_edge_set or (dst, src) in gt_edge_set else 0
                
                # Get model prediction
                if self.use_teacher_forcing and np.random.random() < self.teacher_forcing_ratio:
                    # Teacher forcing: use ground truth history
                    hist_src = train_src_l[history_indices]
                    hist_dst = train_dst_l[history_indices]  
                    hist_ts = train_ts_l[history_indices]
                    hist_idx = train_e_idx_l[history_indices]
                else:
                    # Use model's own predictions (not implemented for now)
                    hist_src = train_src_l[history_indices]
                    hist_dst = train_dst_l[history_indices]
                    hist_ts = train_ts_l[history_indices]
                    hist_idx = train_e_idx_l[history_indices]
                
                try:
                    prob = self.model(
                        hist_src, hist_dst, dst,
                        hist_ts, hist_idx,
                        n_feat, e_feat
                    )
                    candidate_probs.append(prob)
                    candidate_labels.append(label)
                except Exception as e:
                    continue  # Skip problematic candidates
            
            if len(candidate_probs) == 0:
                continue
                
            # Compute loss for all candidates at this timestamp
            probs_tensor = torch.stack(candidate_probs).squeeze(-1)  # Remove extra dimension
            labels_tensor = torch.tensor(candidate_labels, dtype=torch.float, device=self.device)
            
            timestamp_loss = nn.BCELoss()(probs_tensor, labels_tensor)
            total_loss += timestamp_loss
            
            # Accuracy computation
            pred_labels = (probs_tensor > 0.5).cpu().numpy()
            correct_predictions += (pred_labels == np.array(candidate_labels)).sum()
            num_predictions += len(candidate_labels)
        
        avg_accuracy = correct_predictions / max(num_predictions, 1)
        return total_loss, avg_accuracy
    
    def individual_loss(self, train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
                       n_feat, e_feat, sampler, step_range):
        """
        Compute individual edge prediction loss (TGIB-style)
        """
        total_loss = 0.0
        correct_predictions = 0
        num_predictions = 0
        
        criterion = nn.BCELoss()
        
        for step, k in enumerate(step_range):
            if k >= len(train_src_l):
                break
                
            # Positive example (ground truth edge)
            pos_prob = self.model(
                train_src_l[:k+1], train_dst_l[:k+1], train_dst_l[k],
                train_ts_l[:k+1], train_e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Negative example (random edge)
            u_fake, i_fake = sampler.sample(1)
            fake_dst = i_fake[0]
            
            neg_prob = self.model(
                train_src_l[:k+1], train_dst_l[:k+1], fake_dst,
                train_ts_l[:k+1], train_e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Loss computation
            pos_label = torch.ones(1, device=self.device)
            neg_label = torch.zeros(1, device=self.device)
            
            loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
            total_loss += loss
            
            # Accuracy
            pos_pred = (pos_prob > 0.5).item()
            neg_pred = (neg_prob > 0.5).item()
            correct_predictions += int(pos_pred == 1) + int(neg_pred == 0)
            num_predictions += 2
        
        avg_accuracy = correct_predictions / max(num_predictions, 1)
        return total_loss, avg_accuracy
    
    def hybrid_loss(self, train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
                   n_feat, e_feat, sampler, step_range):
        """
        Compute hybrid loss: combination of autoregressive and individual
        """
        # Get both losses
        auto_loss, auto_acc = self.autoregressive_loss(
            train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
            n_feat, e_feat, step_range
        )
        
        indiv_loss, indiv_acc = self.individual_loss(
            train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
            n_feat, e_feat, sampler, step_range
        )
        
        # Weighted combination
        alpha = 0.7  # Weight for autoregressive loss
        hybrid_loss = alpha * auto_loss + (1 - alpha) * indiv_loss
        hybrid_acc = alpha * auto_acc + (1 - alpha) * indiv_acc
        
        return hybrid_loss, hybrid_acc
    
    def train_epoch(self, train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
                   n_feat, e_feat, sampler, optimizer, first_prediction_idx):
        """Train one epoch with the configured approach"""
        
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        # Generate random steps for this epoch
        available_steps = list(range(first_prediction_idx, len(train_src_l)))
        if len(available_steps) == 0:
            return 0.0, 0.0
            
        selected_steps = np.random.choice(
            available_steps, 
            min(self.steps_per_epoch, len(available_steps)), 
            replace=len(available_steps) < self.steps_per_epoch
        )
        
        # Compute loss based on training mode
        if self.training_mode == 'autoregressive':
            loss, acc = self.autoregressive_loss(
                train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
                n_feat, e_feat, selected_steps
            )
        elif self.training_mode == 'individual':
            loss, acc = self.individual_loss(
                train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
                n_feat, e_feat, sampler, selected_steps
            )
        elif self.training_mode == 'hybrid':
            loss, acc = self.hybrid_loss(
                train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
                n_feat, e_feat, sampler, selected_steps
            )
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
        
        # Backward pass
        if loss.item() > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return loss.item(), acc
    
    def evaluate(self, train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
                n_feat, e_feat, sampler, first_prediction_idx):
        """Evaluate model performance"""
        
        self.model.eval()
        eval_accuracies = []
        eval_aps = []
        eval_aucs = []
        pos_probs = []
        neg_probs = []
        
        test_steps = min(20, len(train_src_l) - first_prediction_idx)
        
        with torch.no_grad():
            for step in range(test_steps):
                k = first_prediction_idx + step
                if k >= len(train_src_l):
                    break
                
                # Positive prediction
                pos_prob = self.model(
                    train_src_l[:k+1], train_dst_l[:k+1], train_dst_l[k],
                    train_ts_l[:k+1], train_e_idx_l[:k+1],
                    n_feat, e_feat
                )
                
                # Negative prediction
                u_fake, i_fake = sampler.sample(1)
                neg_prob = self.model(
                    train_src_l[:k+1], train_dst_l[:k+1], i_fake[0],
                    train_ts_l[:k+1], train_e_idx_l[:k+1],
                    n_feat, e_feat
                )
                
                # Metrics
                pred_score = np.array([pos_prob.item(), neg_prob.item()])
                pred_label = pred_score > 0.5
                true_label = np.array([1, 0])
                
                acc = (pred_label == true_label).mean()
                ap = average_precision_score(true_label, pred_score)
                auc = roc_auc_score(true_label, pred_score)
                
                eval_accuracies.append(acc)
                eval_aps.append(ap)
                eval_aucs.append(auc)
                pos_probs.append(pos_prob.item())
                neg_probs.append(neg_prob.item())
        
        return {
            'accuracy': np.mean(eval_accuracies) if eval_accuracies else 0.0,
            'ap': np.mean(eval_aps) if eval_aps else 0.0,
            'auc': np.mean(eval_aucs) if eval_aucs else 0.0,
            'pos_probs': pos_probs,
            'neg_probs': neg_probs
        }


def train_tgam_flexible(config):
    """Main training function with flexible configuration"""
    
    # Load dataset
    data = config.get('dataset', 'triadic_fixed')
    g_df = pd.read_csv(f'./processed/{data}/ml_{data}.csv')
    e_feat = np.load(f'./processed/{data}/ml_{data}.npy')
    n_feat = np.load(f'./processed/{data}/ml_{data}_node.npy')
    
    print(f"=== Training TGAM on {data} ===")
    print(f"Dataset: {len(g_df)} edges, {len(n_feat)} nodes")
    print(f"Training Mode: {config.get('training_mode', 'individual')}")
    
    # Data preparation
    val_time = np.quantile(g_df.ts, 0.7)
    train_mask = g_df.ts <= val_time
    train_data = g_df[train_mask]
    
    train_src_l = train_data.u.values
    train_dst_l = train_data.i.values
    train_ts_l = train_data.ts.values
    train_e_idx_l = train_data.idx.values
    
    # Find first prediction index
    first_prediction_idx = np.where(train_ts_l > 0)[0][0] if np.any(train_ts_l > 0) else 1
    
    print(f"Temporal structure:")
    print(f"  Timestamp 0 edges: {np.sum(train_ts_l == 0)} (initial state)")
    print(f"  Timestamp 1+ edges: {np.sum(train_ts_l > 0)} (training targets)")
    print(f"  First prediction at index: {first_prediction_idx}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TGAM_LinkPrediction(
        node_feat_dim=n_feat.shape[1],
        edge_feat_dim=e_feat.shape[1],
        hidden_dim=config.get('hidden_dim', 128),
        max_nodes=config.get('max_nodes', 1000),
        num_graph_layers=config.get('num_graph_layers', 2),
        num_temporal_layers=config.get('num_temporal_layers', 4)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize trainer
    trainer = TGAMTrainer(model, device, config)
    optimizer = optim.Adam(model.parameters(), lr=trainer.lr)
    sampler = RandEdgeSampler(train_src_l, train_dst_l)
    
    # Training loop
    print(f"\n=== Training Progress ({config.get('training_mode', 'individual')}) ===")
    
    epoch_losses = []
    epoch_accuracies = []
    
    for epoch in range(trainer.num_epochs):
        start_time = time.time()
        
        # Train one epoch
        avg_loss, avg_acc = trainer.train_epoch(
            train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
            n_feat, e_feat, sampler, optimizer, first_prediction_idx
        )
        
        epoch_time = time.time() - start_time
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(avg_acc)
        
        print(f"Epoch {epoch+1:2d}/{trainer.num_epochs}: "
              f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Time={epoch_time:.2f}s")
        
        # Early stopping check
        if epoch >= 5 and np.mean(epoch_accuracies[-3:]) > 0.85:
            print(f"Early stopping: High accuracy achieved!")
            break
    
    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    eval_results = trainer.evaluate(
        train_src_l, train_dst_l, train_ts_l, train_e_idx_l,
        n_feat, e_feat, sampler, first_prediction_idx
    )
    
    print(f"\nFinal Results ({config.get('training_mode', 'individual')}):")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  AP Score: {eval_results['ap']:.4f}")
    print(f"  AUC Score: {eval_results['auc']:.4f}")
    
    # Training analysis
    improvement = epoch_accuracies[-1] - epoch_accuracies[0] if len(epoch_accuracies) > 1 else 0
    print(f"Training improvement: {improvement:.4f}")
    
    # Save model
    model_name = f"tgam_{data}_{config.get('training_mode', 'individual')}"
    if config.get('teacher_forcing', True):
        model_name += "_tf"
    model_name += ".pth"
    
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to {model_name}")
    
    return {
        'final_accuracy': eval_results['accuracy'],
        'final_ap': eval_results['ap'],
        'final_auc': eval_results['auc'],
        'training_losses': epoch_losses,
        'training_accuracies': epoch_accuracies,
        'improvement': improvement,
        'config': config
    }


if __name__ == '__main__':
    # Configuration examples
    configs = [
        {
            'training_mode': 'individual',
            'teacher_forcing': True,
            'teacher_forcing_ratio': 1.0,
            'dataset': 'triadic_fixed',
            'epochs': 15,
            'lr': 0.001,
            'steps_per_epoch': 40,
            'hidden_dim': 128,
            'max_nodes': 100,  # Smaller since we have 31 nodes
            'num_graph_layers': 2,
            'num_temporal_layers': 4
        },
        {
            'training_mode': 'autoregressive', 
            'teacher_forcing': True,
            'teacher_forcing_ratio': 1.0,
            'dataset': 'triadic_fixed',
            'epochs': 15,
            'lr': 0.001,
            'steps_per_epoch': 20,  # Fewer steps due to computational cost
            'hidden_dim': 128,
            'max_nodes': 100,
            'num_graph_layers': 2,
            'num_temporal_layers': 4
        },
        {
            'training_mode': 'hybrid',
            'teacher_forcing': True,
            'teacher_forcing_ratio': 0.8,  # 80% teacher forcing
            'dataset': 'triadic_fixed',
            'epochs': 15,
            'lr': 0.001,
            'steps_per_epoch': 30,
            'hidden_dim': 128,
            'max_nodes': 100,
            'num_graph_layers': 2,
            'num_temporal_layers': 4
        }
    ]
    
    results = {}
    
    print("=== Comparing Training Approaches ===\n")
    
    for i, config in enumerate(configs):
        print(f"\n--- Experiment {i+1}: {config['training_mode']} ---")
        results[config['training_mode']] = train_tgam_flexible(config)
        print(f"--- End Experiment {i+1} ---\n")
    
    # Comparison summary
    print("\n=== COMPARISON SUMMARY ===")
    for mode, result in results.items():
        print(f"{mode:>15}: Acc={result['final_accuracy']:.3f}, "
              f"AP={result['final_ap']:.3f}, AUC={result['final_auc']:.3f}, "
              f"Improve={result['improvement']:.3f}") 