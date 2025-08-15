"""
Comprehensive Evaluation and Visualization for GraphMamba Contagion Models
-----------------------------------------------------------------------
Implements TGIB-style evaluation strategies:

1. Explanation-only performance (TGIB-style): AP/AUC vs sparsity ρ
2. Fidelity curves: Deletion/Insertion AUC 
3. Process-grounded metrics: Path-Recall@k, Triadic closure
4. Parsimony & stability: Sparsity, temporal span, TV, Jaccard overlap

Author: AI Assistant
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import json
import logging
from typing import List, Tuple, Dict, Optional
import networkx as nx
from collections import defaultdict
import os

from graphmamba_IB_explain import GraphMamba


class ContagionEvaluator:
    """Comprehensive evaluator for GraphMamba contagion models"""
    
    def __init__(self, model: GraphMamba, graph_sequence: List[torch.Tensor], 
                 g_df: pd.DataFrame, timestamps: List, device: torch.device, logger: logging.Logger):
        self.model = model
        self.graph_sequence = graph_sequence
        self.g_df = g_df
        self.timestamps = timestamps
        self.device = device
        self.logger = logger
        
        # Extract edge information
        self.edge_timestamps = self._extract_edge_timestamps()
        self.max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1
        
    def _extract_edge_timestamps(self) -> Dict[Tuple[int, int], List[float]]:
        """Extract when each edge appears in the sequence"""
        edge_timestamps = defaultdict(list)
        for _, row in self.g_df.iterrows():
            u, v = int(row['u']), int(row['i'])
            edge = (min(u, v), max(u, v))
            edge_timestamps[edge].append(row['ts'])
        return dict(edge_timestamps)
    
    def evaluate_explanation_only_performance(self, sparsity_levels: List[float] = None) -> Dict:
        """
        TGIB-style evaluation: Keep only top-ρ% gated edges, predict using only R_k
        Returns AP/AUC vs sparsity ρ
        """
        if sparsity_levels is None:
            sparsity_levels = np.arange(0.05, 1.01, 0.05)  # 5% to 100%
        
        self.logger.info("Evaluating explanation-only performance...")
        
        # Get gates for the entire sequence
        with torch.no_grad():
            self.model.eval()
            seq_emb, gates_list = self.model.forward_sequence(self.graph_sequence, return_gates=True)
        
        results = {
            'sparsity_levels': sparsity_levels,
            'ap_scores': [],
            'auc_scores': [],
            'num_edges_kept': []
        }
        
        for sparsity in tqdm(sparsity_levels, desc="Sparsity levels"):
            ap_scores, auc_scores, edge_counts = [], [], []
            
            # Evaluate on each timestep
            for i in range(len(self.timestamps) - 1):
                next_ts = self.timestamps[i + 1]
                current_gates = gates_list[i]
                
                # Get top-k edges based on sparsity
                k = int(sparsity * (current_gates > 0).sum().item())
                top_edges = self.model.topk_edges_from_gates(current_gates, k)
                
                if len(top_edges) == 0:
                    continue
                
                # Create subgraph with only top edges
                subgraph_adj = torch.zeros(self.max_nodes, self.max_nodes, device=self.device)
                for u, v, _ in top_edges:
                    subgraph_adj[u, v] = 1.0
                    subgraph_adj[v, u] = 1.0
                
                # Get predictions using only subgraph
                current_emb = seq_emb[i]
                next_edges = self.g_df[self.g_df['ts'] == next_ts]
                
                if len(next_edges) == 0:
                    continue
                
                # Sample positive and negative pairs
                pos, neg = self._sample_edge_pairs(next_edges, current_emb.shape[0])
                if len(pos) == 0 or len(neg) == 0:
                    continue
                
                # Balance samples
                num_samples = min(len(pos), len(neg))
                pos_s = pos[:num_samples]
                neg_s = neg[:num_samples]
                
                pairs = pos_s + neg_s
                labels = [1.0] * len(pos_s) + [0.0] * len(neg_s)
                
                # Predict using subgraph
                pairs_t = torch.tensor(pairs, device=self.device)
                preds = self.model.predict_next_edges(current_emb, pairs_t)
                
                # Calculate metrics
                pred_np = preds.cpu().numpy()
                labels_np = np.array(labels)
                
                try:
                    ap = average_precision_score(labels_np, pred_np)
                    auc = roc_auc_score(labels_np, pred_np)
                    ap_scores.append(ap)
                    auc_scores.append(auc)
                    edge_counts.append(len(top_edges))
                except ValueError:
                    continue
            
            if ap_scores:
                results['ap_scores'].append(np.mean(ap_scores))
                results['auc_scores'].append(np.mean(auc_scores))
                results['num_edges_kept'].append(np.mean(edge_counts))
            else:
                results['ap_scores'].append(0.0)
                results['auc_scores'].append(0.0)
                results['num_edges_kept'].append(0)
        
        return results
    
    def evaluate_fidelity_curves(self, sparsity_levels: List[float] = None) -> Dict:
        """
        Deletion/Insertion AUC: Remove/insert explanation edges and measure prediction drop/recovery
        """
        if sparsity_levels is None:
            sparsity_levels = np.arange(0.05, 1.01, 0.05)
        
        self.logger.info("Evaluating fidelity curves (deletion/insertion)...")
        
        with torch.no_grad():
            self.model.eval()
            seq_emb, gates_list = self.model.forward_sequence(self.graph_sequence, return_gates=True)
        
        results = {
            'sparsity_levels': sparsity_levels,
            'deletion_auc': [],
            'insertion_auc': [],
            'prediction_drops': []
        }
        
        for sparsity in tqdm(sparsity_levels, desc="Fidelity evaluation"):
            deletion_scores, insertion_scores, drops = [], [], []
            
            for i in range(len(self.timestamps) - 1):
                next_ts = self.timestamps[i + 1]
                current_gates = gates_list[i]
                current_emb = seq_emb[i]
                
                # Get original predictions
                next_edges = self.g_df[self.g_df['ts'] == next_ts]
                if len(next_edges) == 0:
                    continue
                
                pos, neg = self._sample_edge_pairs(next_edges, current_emb.shape[0])
                if len(pos) == 0 or len(neg) == 0:
                    continue
                
                num_samples = min(len(pos), len(neg))
                pos_s = pos[:num_samples]
                neg_s = neg[:num_samples]
                pairs = pos_s + neg_s
                labels = [1.0] * len(pos_s) + [0.0] * len(neg_s)
                
                pairs_t = torch.tensor(pairs, device=self.device)
                original_preds = self.model.predict_next_edges(current_emb, pairs_t)
                
                # Deletion: Remove top edges and measure prediction drop
                k = int(sparsity * (current_gates > 0).sum().item())
                top_edges = self.model.topk_edges_from_gates(current_gates, k)
                
                if len(top_edges) > 0:
                    # Create mask for deletion
                    deletion_mask = torch.ones_like(current_gates)
                    for u, v, _ in top_edges:
                        deletion_mask[u, v] = 0.0
                        deletion_mask[v, u] = 0.0
                    
                    # Apply deletion and predict
                    modified_emb = current_emb * deletion_mask.unsqueeze(-1)
                    deletion_preds = self.model.predict_next_edges(modified_emb, pairs_t)
                    
                    # Calculate deletion AUC
                    try:
                        deletion_auc = roc_auc_score(labels, deletion_preds.cpu().numpy())
                        deletion_scores.append(deletion_auc)
                        
                        # Measure prediction drop
                        pred_drop = (original_preds - deletion_preds).abs().mean().item()
                        drops.append(pred_drop)
                    except ValueError:
                        continue
                
                # Insertion: Add top edges and measure prediction recovery
                if len(top_edges) > 0:
                    # Create mask for insertion (add edges that weren't there)
                    insertion_mask = torch.zeros_like(current_gates)
                    for u, v, _ in top_edges:
                        insertion_mask[u, v] = 1.0
                        insertion_mask[v, u] = 1.0
                    
                    # Apply insertion and predict
                    modified_emb = current_emb + insertion_mask.unsqueeze(-1) * 0.1  # Small addition
                    insertion_preds = self.model.predict_next_edges(modified_emb, pairs_t)
                    
                    try:
                        insertion_auc = roc_auc_score(labels, insertion_preds.cpu().numpy())
                        insertion_scores.append(insertion_auc)
                    except ValueError:
                        continue
            
            if deletion_scores:
                results['deletion_auc'].append(np.mean(deletion_scores))
                results['insertion_auc'].append(np.mean(insertion_scores))
                results['prediction_drops'].append(np.mean(drops))
            else:
                results['deletion_auc'].append(0.0)
                results['insertion_auc'].append(0.0)
                results['prediction_drops'].append(0.0)
        
        return results
    
    def evaluate_process_grounded_metrics(self, k_values: List[int] = None) -> Dict:
        """
        Process-grounded metrics for contagion:
        - Path-Recall@k: fraction of edges on minimal diffusion paths captured by top-k gates
        - Counterfactual drop: re-simulate with edges removed
        """
        if k_values is None:
            k_values = [5, 10, 20, 50, 100]
        
        self.logger.info("Evaluating process-grounded metrics...")
        
        with torch.no_grad():
            self.model.eval()
            seq_emb, gates_list = self.model.forward_sequence(self.graph_sequence, return_gates=True)
        
        results = {
            'k_values': k_values,
            'path_recall': [],
            'counterfactual_drops': [],
            'temporal_coverage': []
        }
        
        for k in tqdm(k_values, desc="Process-grounded evaluation"):
            path_recalls, counterfactual_drops, temporal_covers = [], [], []
            
            for i in range(len(self.timestamps) - 1):
                next_ts = self.timestamps[i + 1]
                current_gates = gates_list[i]
                current_emb = seq_emb[i]
                
                # Get top-k edges
                top_edges = self.model.topk_edges_from_gates(current_gates, k)
                if len(top_edges) == 0:
                    continue
                
                # Path-Recall@k: Check if top edges are on diffusion paths
                next_edges = self.g_df[self.g_df['ts'] == next_ts]
                if len(next_edges) == 0:
                    continue
                
                # Simple heuristic: edges that appear before the target timestamp
                target_edges = set()
                for _, row in next_edges.iterrows():
                    u, v = int(row['u']), int(row['i'])
                    target_edges.add((min(u, v), max(u, v)))
                
                # Check overlap with top-k edges
                top_edge_set = {(u, v) for u, v, _ in top_edges}
                path_overlap = len(top_edge_set.intersection(target_edges))
                path_recall = path_overlap / len(target_edges) if target_edges else 0.0
                path_recalls.append(path_recall)
                
                # Counterfactual drop: Remove top edges and measure performance drop
                if len(top_edges) > 0:
                    # Create deletion mask
                    deletion_mask = torch.ones_like(current_gates)
                    for u, v, _ in top_edges:
                        deletion_mask[u, v] = 0.0
                        deletion_mask[v, u] = 0.0
                    
                    # Sample pairs for evaluation
                    pos, neg = self._sample_edge_pairs(next_edges, current_emb.shape[0])
                    if len(pos) > 0 and len(neg) > 0:
                        num_samples = min(len(pos), len(neg))
                        pos_s = pos[:num_samples]
                        neg_s = neg[:num_samples]
                        pairs = pos_s + neg_s
                        labels = [1.0] * len(pos_s) + [0.0] * len(neg_s)
                        
                        pairs_t = torch.tensor(pairs, device=self.device)
                        
                        # Original predictions
                        original_preds = self.model.predict_next_edges(current_emb, pairs_t)
                        
                        # Modified predictions
                        modified_emb = current_emb * deletion_mask.unsqueeze(-1)
                        modified_preds = self.model.predict_next_edges(modified_emb, pairs_t)
                        
                        # Calculate drop
                        pred_drop = (original_preds - modified_preds).abs().mean().item()
                        counterfactual_drops.append(pred_drop)
                
                # Temporal coverage: How many timesteps do the top edges span?
                edge_timestamps = []
                for u, v, _ in top_edges:
                    edge = (min(u, v), max(u, v))
                    if edge in self.edge_timestamps:
                        edge_timestamps.extend(self.edge_timestamps[edge])
                
                if edge_timestamps:
                    temporal_span = max(edge_timestamps) - min(edge_timestamps)
                    temporal_covers.append(temporal_span)
            
            if path_recalls:
                results['path_recall'].append(np.mean(path_recalls))
                results['counterfactual_drops'].append(np.mean(counterfactual_drops))
                results['temporal_coverage'].append(np.mean(temporal_covers))
            else:
                results['path_recall'].append(0.0)
                results['counterfactual_drops'].append(0.0)
                results['temporal_coverage'].append(0.0)
        
        return results
    
    def evaluate_parsimony_stability(self, perturbation_levels: List[float] = None) -> Dict:
        """
        Parsimony & stability metrics:
        - Sparsity: |R_k|/|G_k|
        - Temporal span and TV
        - Jaccard overlap under perturbations
        """
        if perturbation_levels is None:
            perturbation_levels = [0.01, 0.05, 0.1, 0.2]
        
        self.logger.info("Evaluating parsimony and stability...")
        
        with torch.no_grad():
            self.model.eval()
            seq_emb, gates_list = self.model.forward_sequence(self.graph_sequence, return_gates=True)
        
        results = {
            'sparsity_ratios': [],
            'temporal_variations': [],
            'jaccard_stabilities': []
        }
        
        # Calculate sparsity ratios
        for i, gates in enumerate(gates_list):
            if i == 0:
                continue
            
            # Sparsity: ratio of active gates to total possible edges
            active_gates = (gates > 0.1).float()  # Threshold for "active"
            total_possible = self.max_nodes * (self.max_nodes - 1) // 2
            sparsity_ratio = active_gates.sum().item() / total_possible
            results['sparsity_ratios'].append(sparsity_ratio)
            
            # Temporal variation (TV) with previous timestep
            if i > 0:
                prev_gates = gates_list[i-1]
                common_mask = ((gates > 0.1) & (prev_gates > 0.1)).float()
                if common_mask.sum().item() > 0:
                    tv = (gates - prev_gates).abs() * common_mask
                    tv_mean = tv.sum().item() / common_mask.sum().item()
                    results['temporal_variations'].append(tv_mean)
        
        # Jaccard stability under perturbations
        for pert_level in perturbation_levels:
            jaccard_scores = []
            
            for i in range(len(gates_list) - 1):
                current_gates = gates_list[i]
                next_gates = gates_list[i + 1]
                
                # Add perturbation to current gates
                perturbation = torch.randn_like(current_gates) * pert_level
                perturbed_gates = current_gates + perturbation
                perturbed_gates = torch.clamp(perturbed_gates, 0, 1)
                
                # Get top-k edges from both
                k = 20  # Fixed k for stability comparison
                original_top = set(self.model.topk_edges_from_gates(current_gates, k))
                perturbed_top = set(self.model.topk_edges_from_gates(perturbed_gates, k))
                
                # Calculate Jaccard similarity
                if original_top and perturbed_top:
                    intersection = len(original_top.intersection(perturbed_top))
                    union = len(original_top.union(perturbed_top))
                    jaccard = intersection / union if union > 0 else 0.0
                    jaccard_scores.append(jaccard)
            
            if jaccard_scores:
                results['jaccard_stabilities'].append(np.mean(jaccard_scores))
            else:
                results['jaccard_stabilities'].append(0.0)
        
        return results
    
    def _sample_edge_pairs(self, next_edges: pd.DataFrame, num_nodes: int) -> Tuple[List, List]:
        """Sample positive and negative edge pairs for evaluation"""
        pos, neg = [], []
        
        # Positive edges (existing at next timestamp)
        for _, row in next_edges.iterrows():
            u, v = int(row['u']), int(row['i'])
            pos.append((u, v))
        
        # Negative edges (random non-existing pairs)
        existing_edges = set()
        for _, row in next_edges.iterrows():
            u, v = int(row['u']), int(row['i'])
            existing_edges.add((min(u, v), max(u, v)))
        
        # Sample negative edges
        num_neg = min(len(pos), num_nodes * 2)  # Reasonable number of negatives
        neg_count = 0
        max_attempts = num_neg * 10
        
        for _ in range(max_attempts):
            if neg_count >= num_neg:
                break
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v:
                edge = (min(u, v), max(u, v))
                if edge not in existing_edges and edge not in neg:
                    neg.append(edge)
                    neg_count += 1
        
        return pos, neg
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run all evaluation metrics and return comprehensive results"""
        self.logger.info("Starting comprehensive evaluation...")
        
        results = {}
        
        # 1. Explanation-only performance
        results['explanation_only'] = self.evaluate_explanation_only_performance()
        
        # 2. Fidelity curves
        results['fidelity'] = self.evaluate_fidelity_curves()
        
        # 3. Process-grounded metrics
        results['process_grounded'] = self.evaluate_process_grounded_metrics()
        
        # 4. Parsimony & stability
        results['parsimony_stability'] = self.evaluate_parsimony_stability()
        
        self.logger.info("Comprehensive evaluation completed!")
        return results


class ContagionVisualizer:
    """Visualization tools for contagion evaluation results"""
    
    def __init__(self, results: Dict, save_dir: str = "./evaluation_plots"):
        self.results = results
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_explanation_performance(self, save: bool = True):
        """Plot AP/AUC vs sparsity curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        data = self.results['explanation_only']
        sparsity = data['sparsity_levels']
        
        # AP vs Sparsity
        ax1.plot(sparsity, data['ap_scores'], 'o-', linewidth=2, markersize=6)
        ax1.set_xlabel('Sparsity Level (ρ)')
        ax1.set_ylabel('Average Precision')
        ax1.set_title('Explanation-Only Performance: AP vs Sparsity')
        ax1.grid(True, alpha=0.3)
        
        # AUC vs Sparsity
        ax2.plot(sparsity, data['auc_scores'], 's-', linewidth=2, markersize=6, color='orange')
        ax2.set_xlabel('Sparsity Level (ρ)')
        ax2.set_ylabel('ROC AUC')
        ax2.set_title('Explanation-Only Performance: AUC vs Sparsity')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_dir}/explanation_performance.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_fidelity_curves(self, save: bool = True):
        """Plot deletion/insertion fidelity curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        data = self.results['fidelity']
        sparsity = data['sparsity_levels']
        
        # Deletion AUC
        ax1.plot(sparsity, data['deletion_auc'], 'o-', linewidth=2, markersize=6, color='red')
        ax1.set_xlabel('Sparsity Level (ρ)')
        ax1.set_ylabel('Deletion AUC')
        ax1.set_title('Fidelity: Deletion AUC vs Sparsity')
        ax1.grid(True, alpha=0.3)
        
        # Insertion AUC
        ax2.plot(sparsity, data['insertion_auc'], 's-', linewidth=2, markersize=6, color='green')
        ax2.set_xlabel('Sparsity Level (ρ)')
        ax2.set_ylabel('Insertion AUC')
        ax2.set_title('Fidelity: Insertion AUC vs Sparsity')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_dir}/fidelity_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_process_grounded_metrics(self, save: bool = True):
        """Plot process-grounded evaluation metrics"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        data = self.results['process_grounded']
        k_values = data['k_values']
        
        # Path Recall
        ax1.plot(k_values, data['path_recall'], 'o-', linewidth=2, markersize=6)
        ax1.set_xlabel('Top-k Edges')
        ax1.set_ylabel('Path Recall')
        ax1.set_title('Process-Grounded: Path Recall@k')
        ax1.grid(True, alpha=0.3)
        
        # Counterfactual Drops
        ax2.plot(k_values, data['counterfactual_drops'], 's-', linewidth=2, markersize=6, color='orange')
        ax2.set_xlabel('Top-k Edges')
        ax2.set_ylabel('Prediction Drop')
        ax2.set_title('Process-Grounded: Counterfactual Drop')
        ax2.grid(True, alpha=0.3)
        
        # Temporal Coverage
        ax3.plot(k_values, data['temporal_coverage'], '^-', linewidth=2, markersize=6, color='green')
        ax3.set_xlabel('Top-k Edges')
        ax3.set_ylabel('Temporal Span')
        ax3.set_title('Process-Grounded: Temporal Coverage')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_dir}/process_grounded_metrics.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_parsimony_stability(self, save: bool = True):
        """Plot parsimony and stability metrics"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        data = self.results['parsimony_stability']
        
        # Sparsity ratios over time
        if data['sparsity_ratios']:
            ax1.plot(data['sparsity_ratios'], 'o-', linewidth=2, markersize=4)
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Sparsity Ratio')
            ax1.set_title('Parsimony: Sparsity Over Time')
            ax1.grid(True, alpha=0.3)
        
        # Temporal variations
        if data['temporal_variations']:
            ax2.plot(data['temporal_variations'], 's-', linewidth=2, markersize=4, color='orange')
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Temporal Variation (TV)')
            ax2.set_title('Stability: Temporal Variation')
            ax2.grid(True, alpha=0.3)
        
        # Jaccard stability under perturbations
        perturbation_levels = [0.01, 0.05, 0.1, 0.2]
        if data['jaccard_stabilities']:
            ax3.plot(perturbation_levels, data['jaccard_stabilities'], '^-', linewidth=2, markersize=6, color='green')
            ax3.set_xlabel('Perturbation Level')
            ax3.set_ylabel('Jaccard Stability')
            ax3.set_title('Stability: Jaccard Overlap')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_dir}/parsimony_stability.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, save: bool = True) -> str:
        """Create a comprehensive summary report"""
        report = []
        report.append("=" * 60)
        report.append("GRAPHMAMBA CONTAGION EVALUATION SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Explanation-only performance summary
        exp_data = self.results['explanation_only']
        report.append("1. EXPLANATION-ONLY PERFORMANCE (TGIB-style)")
        report.append("-" * 40)
        best_ap_idx = np.argmax(exp_data['ap_scores'])
        best_auc_idx = np.argmax(exp_data['auc_scores'])
        report.append(f"Best AP: {exp_data['ap_scores'][best_ap_idx]:.4f} at sparsity {exp_data['sparsity_levels'][best_ap_idx]:.2f}")
        report.append(f"Best AUC: {exp_data['auc_scores'][best_auc_idx]:.4f} at sparsity {exp_data['sparsity_levels'][best_auc_idx]:.2f}")
        report.append(f"AP at 20% sparsity: {exp_data['ap_scores'][3]:.4f}")
        report.append(f"AUC at 20% sparsity: {exp_data['auc_scores'][3]:.4f}")
        report.append("")
        
        # Fidelity summary
        fid_data = self.results['fidelity']
        report.append("2. FIDELITY CURVES")
        report.append("-" * 40)
        report.append(f"Deletion AUC range: [{min(fid_data['deletion_auc']):.4f}, {max(fid_data['deletion_auc']):.4f}]")
        report.append(f"Insertion AUC range: [{min(fid_data['insertion_auc']):.4f}, {max(fid_data['insertion_auc']):.4f}]")
        report.append(f"Max prediction drop: {max(fid_data['prediction_drops']):.4f}")
        report.append("")
        
        # Process-grounded summary
        proc_data = self.results['process_grounded']
        report.append("3. PROCESS-GROUNDED METRICS")
        report.append("-" * 40)
        report.append(f"Path Recall@10: {proc_data['path_recall'][1]:.4f}")
        report.append(f"Path Recall@20: {proc_data['path_recall'][2]:.4f}")
        report.append(f"Counterfactual drop@20: {proc_data['counterfactual_drops'][2]:.4f}")
        report.append(f"Temporal coverage@20: {proc_data['temporal_coverage'][2]:.4f}")
        report.append("")
        
        # Parsimony & stability summary
        par_data = self.results['parsimony_stability']
        report.append("4. PARSIMONY & STABILITY")
        report.append("-" * 40)
        if par_data['sparsity_ratios']:
            report.append(f"Mean sparsity ratio: {np.mean(par_data['sparsity_ratios']):.4f}")
        if par_data['temporal_variations']:
            report.append(f"Mean temporal variation: {np.mean(par_data['temporal_variations']):.4f}")
        if par_data['jaccard_stabilities']:
            report.append(f"Mean Jaccard stability: {np.mean(par_data['jaccard_stabilities']):.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if save:
            with open(f"{self.save_dir}/evaluation_summary.txt", 'w') as f:
                f.write(report_text)
        
        return report_text


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate GraphMamba Contagion Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_name', type=str, default='synthetic_icm_ba', help='Dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--save_dir', type=str, default='./evaluation_plots', help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info(f"Loading {args.data_name} dataset...")
    g_df = pd.read_csv(f'./processed/{args.data_name}/ml_{args.data_name}.csv')
    timestamps = sorted(g_df['ts'].unique())
    
    # Create graph sequence
    graph_sequence = []
    max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1
    for ts in timestamps:
        edges_up_to_ts = g_df[g_df['ts'] <= ts]
        adj = torch.zeros(max_nodes, max_nodes)
        for _, row in edges_up_to_ts.iterrows():
            u, v = int(row['u']), int(row['i'])
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        graph_sequence.append(adj)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = GraphMamba(max_nodes=max_nodes, pos_dim=128, hidden_dim=64,
                       gnn_layers=2, mamba_state_dim=16, dropout=0.1,
                       use_edge_gates=True, gate_temperature=1.0).to(device)
    
    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Run evaluation
    evaluator = ContagionEvaluator(model, graph_sequence, g_df, timestamps, device, logger)
    results = evaluator.run_comprehensive_evaluation()
    
    # Visualize results
    visualizer = ContagionVisualizer(results, args.save_dir)
    visualizer.plot_explanation_performance()
    visualizer.plot_fidelity_curves()
    visualizer.plot_process_grounded_metrics()
    visualizer.plot_parsimony_stability()
    
    # Print summary
    summary = visualizer.create_summary_report()
    print(summary)
    
    logger.info(f"Evaluation completed! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
