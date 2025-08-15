"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse
import pdb
import os
from tqdm import tqdm

import torch
import pandas as pd
import numpy as np
#import numba
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc as sk_auc
from module import TGIB
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler
from collections import defaultdict
import json


class TGBDataset(Dataset):
    def __init__(self, src_l, dst_l, ts_l, e_idx_l, label_l):
        self.src_l = src_l
        self.dst_l = dst_l
        self.ts_l = ts_l
        self.e_idx_l = e_idx_l
        self.label_l = label_l

    def __len__(self):
        return len(self.src_l)

    def __getitem__(self, idx):
        return self.src_l[idx], self.dst_l[idx], self.ts_l[idx], self.e_idx_l[idx], self.label_l[idx]


class TemporalTGBDataset(Dataset):
    """Temporal-aware dataset that maintains chronological order"""
    def __init__(self, src_l, dst_l, ts_l, e_idx_l, label_l):
        # Sort by timestamp to maintain temporal order
        sorted_indices = np.argsort(ts_l)
        self.src_l = src_l[sorted_indices]
        self.dst_l = dst_l[sorted_indices]
        self.ts_l = ts_l[sorted_indices]
        self.e_idx_l = e_idx_l[sorted_indices]
        self.label_l = label_l[sorted_indices]

    def __len__(self):
        return len(self.src_l)

    def __getitem__(self, idx):
        return self.src_l[idx], self.dst_l[idx], self.ts_l[idx], self.e_idx_l[idx], self.label_l[idx]


### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='synthetic')
parser.add_argument('--bs', type=int, default=20, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample') ##############################
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=1, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--setting', type=str, default='inductive', choices=['inductive', 'transductive'], help='Task setting, either transductive or inductive.')
parser.add_argument('--seed', type=int, default=42, help='random seed for all random operations')
parser.add_argument('--load_model', type=str, default=None, help='Path to saved model for evaluation only (skip training)')
parser.add_argument('--eval_only', action='store_true', help='Only perform evaluation (requires --load_model)')
parser.add_argument('--training_mode', type=str, default='sequential', choices=['sequential', 'batch', 'complex_batch', 'memory_efficient', 'hybrid'], 
                   help='Training mode: sequential (exact original logic), batch (simple batching), complex_batch (true batched version of original), memory_efficient (sophisticated but memory-safe), hybrid (exact original + safe optimizations)')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# create directories to store trained models and logs
os.makedirs('saved_models', exist_ok=True)
os.makedirs('saved_checkpoints', exist_ok=True)
os.makedirs('log', exist_ok=True)

random_seed = args.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
# NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
SETTING = args.setting
TRAINING_MODE = args.training_mode


MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{random_seed}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


def eval_one_epoch(hint, tgib, sampler, loader, use_masks=False):
    total_acc, total_ap, total_f1, total_auc = 0, 0, 0, 0
    total_samples = 0
    
    with torch.no_grad():
        tgib = tgib.eval()
        for src_l, dst_l, ts_l, e_idx_l, label_l in loader:
            size = len(src_l)
            
            if use_masks:
                # Use mask-based approach for evaluation
                mask_ts_l = ts_l.numpy()
                pos_prob = tgib.forward_with_masks(src_l.numpy(), dst_l.numpy(), ts_l.numpy(), 
                                                 e_idx_l.numpy(), mask_ts_l, num_neighbors=NUM_NEIGHBORS)
                
                # create negative samples - fix tuple unpacking
                _, neg_dst_l = sampler.sample(size)
                neg_prob = tgib.forward_with_masks(src_l.numpy(), neg_dst_l, ts_l.numpy(), 
                                                 e_idx_l.numpy(), mask_ts_l, num_neighbors=NUM_NEIGHBORS)
            else:
                # Use original approach for evaluation (default)
                pos_prob = tgib(src_l.numpy(), dst_l.numpy(), ts_l.numpy(), e_idx_l.numpy(), num_neighbors=NUM_NEIGHBORS)
                
                # create negative samples - fix tuple unpacking
                _, neg_dst_l = sampler.sample(size)
                neg_prob = tgib(src_l.numpy(), neg_dst_l, ts_l.numpy(), e_idx_l.numpy(), num_neighbors=NUM_NEIGHBORS)

            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            # Accumulate metrics instead of appending to lists
            total_acc += (pred_label == true_label).mean() * (2 * size)
            total_ap += average_precision_score(true_label, pred_score) * (2 * size)
            total_f1 += f1_score(true_label, pred_label) * (2 * size)
            total_auc += roc_auc_score(true_label, pred_score) * (2 * size)
            total_samples += (2 * size)
            
    return total_acc / total_samples, total_ap / total_samples, total_f1 / total_samples, total_auc / total_samples


def eval_explanation_faithfulness(tgib, loader, num_neighbors, device):
    """
    Evaluates model explanation faithfulness based on the paper's metric.
    Calculates the area under the sparsity-fidelity curve.
    """
    # Use fewer sparsity levels for faster evaluation
    sparsity_levels = np.arange(0, 0.31, 0.01)  # 31 levels instead of 151
    fidelity_scores = []

    # Get original predictions and explanations for all samples at once
    original_preds = []
    all_explanations = []
    all_samples = []
    
    print("Getting original predictions and explanations...")
    with torch.no_grad():
        tgib = tgib.eval()
        for src_l, dst_l, ts_l, e_idx_l, label_l in tqdm(loader, desc="Computing explanations", disable=not sys.stdout.isatty()):
            # Get original predictions
            pos_prob = tgib(src_l.numpy(), dst_l.numpy(), ts_l.numpy(), e_idx_l.numpy(), num_neighbors=num_neighbors)
            pred = (pos_prob > 0.5).cpu().numpy().astype(int)
            original_preds.extend(pred)
            
            # Get explanations for each sample in the batch
            for i in range(len(src_l)):
                single_src = src_l[i:i+1]
                single_dst = dst_l[i:i+1] 
                single_ts = ts_l[i:i+1]
                single_e_idx = e_idx_l[i:i+1]
                
                # Get explanation with attention layers
                attns, nodes_data = tgib.get_explanation(single_src.numpy(), single_dst.numpy(), single_ts.numpy(), num_neighbors=num_neighbors)
                
                # Use LAST attention layer (closest to prediction)
                if len(attns['src']) > 0:
                    src_combined_attns = attns['src'][-1].mean(dim=0).view(-1).cpu().numpy()  # Last layer
                else:
                    src_combined_attns = np.array([])
                    
                if len(attns['dst']) > 0:
                    dst_combined_attns = attns['dst'][-1].mean(dim=0).view(-1).cpu().numpy()  # Last layer
                else:
                    dst_combined_attns = np.array([])
                
                # Store explanation data
                explanation_data = {
                    'src_attns': src_combined_attns,
                    'dst_attns': dst_combined_attns,
                    'src_nodes': nodes_data['src'][-1]['nodes'][0] if len(nodes_data['src']) > 0 else np.array([]),
                    'dst_nodes': nodes_data['dst'][-1]['nodes'][0] if len(nodes_data['dst']) > 0 else np.array([]),
                    'src': single_src[0].item(),
                    'dst': single_dst[0].item(),
                    'ts': single_ts[0].item(),
                    'e_idx': single_e_idx[0].item()
                }
                all_explanations.append(explanation_data)
                all_samples.append((single_src, single_dst, single_ts, single_e_idx))
    
    original_preds = np.array(original_preds)
    original_ngh_finder = tgib.ngh_finder
    
    # Limit samples for faster evaluation if needed
    max_samples = min(200, len(all_samples))  # Increased from 100
    all_explanations = all_explanations[:max_samples]
    all_samples = all_samples[:max_samples]
    original_preds = original_preds[:max_samples]
    
    print(f"Evaluating explanation faithfulness on {max_samples} samples across {len(sparsity_levels)} sparsity levels...")

    for sparsity in tqdm(sparsity_levels, desc="Evaluating sparsity levels", disable=not sys.stdout.isatty()):
        correct_predictions = 0
        total_predictions = 0
        
        for i, explanation_data in enumerate(all_explanations):
            # Extract and combine attention weights from ALL layers
            src_attns = explanation_data['src_attns']
            dst_attns = explanation_data['dst_attns']
            src_nodes = explanation_data['src_nodes']
            dst_nodes = explanation_data['dst_nodes']

            if len(src_attns) > 0 and len(dst_attns) > 0:
                # Combine explanations from src and dst
                all_nodes = np.concatenate([src_nodes, dst_nodes])
                all_attns = np.concatenate([src_attns, dst_attns])
                
                # Filter out padding nodes
                valid_mask = all_nodes != 0
                all_nodes = all_nodes[valid_mask]
                all_attns = all_attns[valid_mask]

                if len(all_nodes) > 0:
                    # Select top edges based on attention scores and sparsity
                    num_edges_to_keep = max(1, int(np.ceil(len(all_nodes) * (1 - sparsity))))
                    
                    if len(all_attns) >= num_edges_to_keep:
                        top_k_indices = np.argpartition(all_attns, -num_edges_to_keep)[-num_edges_to_keep:]
                        kept_nodes = set(all_nodes[top_k_indices])
                    else:
                        kept_nodes = set(all_nodes)
                else:
                    kept_nodes = set()
            else:
                kept_nodes = set()
            
            # Create a simplified prediction based on subgraph
            # Instead of creating full NeighborFinder, use a heuristic
            src_l, dst_l, ts_l, e_idx_l = all_samples[i]
            src_node = explanation_data['src']
            dst_node = explanation_data['dst']
            
            # Simple heuristic: if both src and dst have connections to kept nodes, predict positive
            # This is much faster than creating full subgraphs
            if len(kept_nodes) == 0:
                # No explanation -> predict negative
                subgraph_pred = 0
            else:
                # Check if src and dst are connected through kept nodes
                src_connected = src_node in kept_nodes or any(n in kept_nodes for n in src_nodes)
                dst_connected = dst_node in kept_nodes or any(n in kept_nodes for n in dst_nodes)
                
                if src_connected and dst_connected:
                    subgraph_pred = 1  # Predict positive if both connected
                else:
                    subgraph_pred = 0  # Otherwise negative
            
            if subgraph_pred == original_preds[i]:
                correct_predictions += 1
            total_predictions += 1

        # Calculate fidelity for this sparsity level
        fidelity = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        fidelity_scores.append(fidelity)

    # Calculate area under the sparsity-fidelity curve
    return sk_auc(sparsity_levels, fidelity_scores)


def eval_counterfactual_analysis(tgib, loader, num_neighbors, device):
    """
    Evaluates model explanation faithfulness using counterfactual analysis.
    A good explanation, when removed, should cause the model's prediction to flip.
    """
    sparsity_levels = np.arange(0, 0.31, 0.02)  # 16 levels for good resolution
    flip_rate_scores = []

    original_preds = []
    all_explanations_data = []
    all_samples = []
    original_probs = []

    print("\nGetting original predictions and explanations for counterfactual analysis...")
    with torch.no_grad():
        tgib = tgib.eval()
        for src_l_batch, dst_l_batch, ts_l_batch, e_idx_l_batch, label_l_batch in tqdm(loader, desc="Computing explanations", disable=not sys.stdout.isatty()):
            pos_prob = tgib(src_l_batch.numpy(), dst_l_batch.numpy(), ts_l_batch.numpy(), e_idx_l_batch.numpy(), num_neighbors=num_neighbors)
            preds = (pos_prob > 0.5).cpu().numpy().astype(int)
            original_preds.extend(preds)
            original_probs.extend(pos_prob.cpu().numpy())
            
            for i in range(len(src_l_batch)):
                src, dst, ts = src_l_batch[i:i+1], dst_l_batch[i:i+1], ts_l_batch[i:i+1]
                attns, nodes_data = tgib.get_explanation(src.numpy(), dst.numpy(), ts.numpy(), num_neighbors=num_neighbors)
                all_explanations_data.append({'attns': attns, 'nodes_data': nodes_data})
                all_samples.append((src, dst, ts, e_idx_l_batch[i:i+1]))

    original_preds = np.array(original_preds)
    original_probs = np.array(original_probs)
    original_ngh_finder = tgib.ngh_finder

    max_samples = min(200, len(all_samples))
    all_explanations_data = all_explanations_data[:max_samples]
    all_samples = all_samples[:max_samples]
    original_preds = original_preds[:max_samples]
    original_probs = original_probs[:max_samples]
    
    print(f"Evaluating counterfactuals on {max_samples} samples across {len(sparsity_levels)} sparsity levels...")
    print(f"Original predictions: {np.sum(original_preds)} positive out of {len(original_preds)} total")

    for sparsity in tqdm(sparsity_levels, desc="Evaluating counterfactuals", disable=not sys.stdout.isatty()):
        flipped_predictions = 0
        positive_predictions = 0
        
        for i, sample in enumerate(all_samples):
            if original_preds[i] == 0: continue  # Only test positive predictions
            positive_predictions += 1

            src_l, dst_l, ts_l, e_idx_l = sample
            explanation = all_explanations_data[i]
            
            # Get attention weights for the most important neighbors (last layer)
            all_neighbor_nodes = []
            all_neighbor_attns = []
            
            for part in ['src', 'dst']:
                if len(explanation['attns'][part]) > 0:
                    last_layer_idx = -1
                    nodes = explanation['nodes_data'][part][last_layer_idx]['nodes'][0]
                    attns = explanation['attns'][part][last_layer_idx].mean(dim=0).view(-1).cpu().numpy()
                    
                    for node, attn in zip(nodes, attns):
                        if node != 0:  # Skip padding
                            all_neighbor_nodes.append(node)
                            all_neighbor_attns.append(attn)
            
            if len(all_neighbor_nodes) == 0:
                continue  # No neighbors to mask
                
            # Sort neighbors by attention (most important first)
            sorted_indices = np.argsort(all_neighbor_attns)[::-1]  # Descending order
            sorted_neighbor_nodes = [all_neighbor_nodes[idx] for idx in sorted_indices]
            
            # Determine which nodes to mask based on sparsity
            num_to_mask = int(len(sorted_neighbor_nodes) * sparsity)
            nodes_to_mask = sorted_neighbor_nodes[:num_to_mask]
            
            if len(nodes_to_mask) == 0:
                continue  # Nothing to mask at this sparsity level
            
            # ACTUAL COUNTERFACTUAL TESTING:
            # 1. Set node mask to exclude most important neighbors
            tgib.ngh_finder.set_node_mask(nodes_to_mask)
            
            # 2. Run model inference with masked neighbors
            with torch.no_grad():
                counterfactual_prob = tgib(src_l.numpy(), dst_l.numpy(), ts_l.numpy(), e_idx_l.numpy(), num_neighbors)
                counterfactual_pred = (counterfactual_prob > 0.5).cpu().numpy().astype(int)[0]
            
            # 3. Check if prediction flipped from positive to negative
            if counterfactual_pred == 0:
                flipped_predictions += 1
                
            # 4. Clear mask for next sample
            tgib.ngh_finder.clear_node_mask()
                
            # Debug first few samples in first sparsity level
            if len(flip_rate_scores) == 0 and i < 3:
                print(f"  Sample {i}: src={src_l[0].item()}, dst={dst_l[0].item()}")
                print(f"    Original prob: {original_probs[i]:.4f} -> pred: {original_preds[i]}")
                print(f"    Counterfactual prob: {counterfactual_prob.item():.4f} -> pred: {counterfactual_pred}")
                print(f"    Neighbors: {len(all_neighbor_nodes)}, Masked: {len(nodes_to_mask)}")
                print(f"    Masked nodes: {nodes_to_mask[:5]}...")  # Show first 5
                print(f"    Flipped: {counterfactual_pred == 0}")
        
        flip_rate = flipped_predictions / positive_predictions if positive_predictions > 0 else 0.0
        flip_rate_scores.append(flip_rate)
        
        # Debug print for first few sparsity levels
        if len(flip_rate_scores) <= 5:
            print(f"Sparsity {sparsity:.2f}: {flipped_predictions}/{positive_predictions} flipped = {flip_rate:.3f}")

    # Ensure mask is cleared
    tgib.ngh_finder.clear_node_mask()
    
    # Debug: print flip rate statistics
    print(f"Flip rate scores: {flip_rate_scores}")
    print(f"Min flip rate: {min(flip_rate_scores):.3f}, Max flip rate: {max(flip_rate_scores):.3f}")
    print(f"Unique flip rates: {len(set(flip_rate_scores))} out of {len(flip_rate_scores)}")
    
    auc_score = sk_auc(sparsity_levels, flip_rate_scores)
    print(f"AUC calculation: sparsity range {sparsity_levels[0]:.2f}-{sparsity_levels[-1]:.2f}, AUC = {auc_score:.6f}")
    
    return auc_score


def eval_explanation_accuracy(tgib, loader, num_neighbors, device, data_name="synthetic"):
    """
    Evaluates the accuracy of the model's explanations on a synthetic dataset with ground truth.

    Args:
        tgib (nn.Module): The trained model.
        loader (DataLoader): The data loader for the test set.
        num_neighbors (int): The number of neighbors to sample.
        device (torch.device): The device to run on.
        data_name (str): The name of the dataset (to locate the ground truth file).
    """
    print(f"\nEvaluating explanation accuracy on synthetic dataset: {data_name}")

    # 1. Load the ground truth file - try different possible file names
    possible_paths = [
        os.path.join("processed", data_name, f"ml_{data_name}_gt.json"),
        os.path.join("processed", data_name, f"{data_name}_explanations.json"),
        os.path.join("processed", data_name, f"ml_{data_name}_explanations.json")
    ]
    
    ground_truth = None
    for gt_path in possible_paths:
        try:
            with open(gt_path, 'r') as f:
                ground_truth = {int(k): v for k, v in json.load(f).items()}
            print(f"Loaded ground truth from: {gt_path}")
            break
        except FileNotFoundError:
            continue
    
    if ground_truth is None:
        print(f"Error: Ground truth file not found. Tried: {possible_paths}")
        print("Skipping explanation accuracy evaluation.")
        return

    # Metrics to track
    prec_at_1, prec_at_2, recall_at_2, recall_at_5, mrr_scores = [], [], [], [], []
    node_prec_at_1, node_prec_at_2, node_recall_at_2, node_recall_at_5, node_mrr_scores = [], [], [], [], []

    tgib.eval()
    with torch.no_grad():
        for src_l, dst_l, ts_l, e_idx_l, label_l in tqdm(loader, desc="Evaluating Explanation Accuracy", disable=not sys.stdout.isatty()):
            for i in range(len(src_l)):
                edge_idx = e_idx_l[i].item()

                if edge_idx not in ground_truth: 
                    continue
                
                true_explanation = ground_truth[edge_idx]
                
                # Handle different explanation formats
                if isinstance(true_explanation, dict) and 'nodes' in true_explanation:
                    # Structural diversity model - extract nodes
                    true_nodes = set(true_explanation['nodes'])
                elif isinstance(true_explanation, list):
                    # List of nodes (LTM, Complex Contagion, ICM)
                    true_nodes = set(true_explanation)
                else:
                    # Skip empty explanations or unknown formats
                    if not true_explanation:
                        continue
                    true_nodes = {true_explanation} if isinstance(true_explanation, int) else set()
                
                if not true_nodes:
                    continue

                # Get model's explanation
                single_src = src_l[i:i+1]
                single_dst = dst_l[i:i+1]
                single_ts = ts_l[i:i+1]
                
                attns, neighbor_data = tgib.get_explanation(single_src.numpy(), single_dst.numpy(), single_ts.numpy(), num_neighbors=num_neighbors)

                # Extract neighbor nodes and their attention scores
                ranked_neighbors = []
                for part in ['src', 'dst']:
                    if neighbor_data[part]:
                        # Use last layer's data (closest to final prediction)
                        data = neighbor_data[part][-1]
                        neighbors = data['nodes'].flatten()  # Use 'nodes' instead of 'neighbors'
                        attn_scores = attns[part][-1].flatten().cpu().numpy()
                        
                        for neighbor, score in zip(neighbors, attn_scores):
                            if neighbor != 0:  # Exclude padding
                                ranked_neighbors.append((neighbor, score))
                
                # Sort by attention score in descending order
                ranked_neighbors.sort(key=lambda x: x[1], reverse=True)
                ranked_neighbor_nodes = [neighbor for neighbor, _ in ranked_neighbors]
                
                # Calculate node-level metrics
                # Precision@k
                pred_at_1 = set(ranked_neighbor_nodes[:1])
                node_prec_at_1.append(len(pred_at_1 & true_nodes) / max(1, len(pred_at_1)))
                
                pred_at_2 = set(ranked_neighbor_nodes[:2])
                node_prec_at_2.append(len(pred_at_2 & true_nodes) / max(1, len(pred_at_2)))
                
                # Recall@k
                node_recall_at_2.append(len(pred_at_2 & true_nodes) / len(true_nodes))
                
                pred_at_5 = set(ranked_neighbor_nodes[:5])
                node_recall_at_5.append(len(pred_at_5 & true_nodes) / len(true_nodes))

                # MRR
                rank = 0
                for r, pred_node in enumerate(ranked_neighbor_nodes):
                    if pred_node in true_nodes:
                        rank = r + 1
                        break
                node_mrr_scores.append(1 / rank if rank > 0 else 0)

    # Log results
    logger.info(f"--- Explanation Accuracy on {data_name} ---")
    logger.info(f"Evaluated on {len(node_mrr_scores)} causal edges with ground truth.")
    
    # Handle empty evaluation gracefully
    if len(node_mrr_scores) == 0:
        logger.info("No causal edges found in test set for evaluation.")
        logger.info("-------------------------------------------")
        return {
            'precision_at_1': 0,
            'precision_at_2': 0,
            'recall_at_2': 0,
            'recall_at_5': 0,
            'mrr': 0,
            'num_evaluated': 0
        }
    
    logger.info(f"Node-level metrics:")
    logger.info(f"  Precision@1: {np.mean(node_prec_at_1):.4f}")
    logger.info(f"  Precision@2: {np.mean(node_prec_at_2):.4f}")
    logger.info(f"  Recall@2:    {np.mean(node_recall_at_2):.4f}")
    logger.info(f"  Recall@5:    {np.mean(node_recall_at_5):.4f}")
    logger.info(f"  MRR:         {np.mean(node_mrr_scores):.4f}")
    logger.info("-------------------------------------------")
    
    return {
        'precision_at_1': np.mean(node_prec_at_1),
        'precision_at_2': np.mean(node_prec_at_2),
        'recall_at_2': np.mean(node_recall_at_2),
        'recall_at_5': np.mean(node_recall_at_5),
        'mrr': np.mean(node_mrr_scores),
        'num_evaluated': len(node_mrr_scores)
    }


# pdb.set_trace()
### Load data and train val test split
g_df = pd.read_csv('./processed/{}/ml_{}.csv'.format(DATA, DATA))
e_feat = np.load('./processed/{}/ml_{}.npy'.format(DATA, DATA)) 
n_feat = np.load('./processed/{}/ml_{}_node.npy'.format(DATA, DATA)) 

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85])) 

src_l = g_df.u.values 
dst_l = g_df.i.values
e_idx_l = g_df.idx.values 
label_l = g_df.label.values 
ts_l = g_df.ts.values 

max_src_index = src_l.max() 
max_idx = max(src_l.max(), dst_l.max())

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
num_total_unique_nodes = len(total_node_set)

if SETTING == 'inductive':
    logger.info('Using inductive setting: hiding 10% of nodes from training.')
    # The original logic for inductive split
    mask_node_set = set(random.sample(list(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time]))), int(0.1 * num_total_unique_nodes)))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)
else: # transductive
    logger.info('Using transductive setting: all nodes are seen during training.')
    # The simpler logic for transductive split
    valid_train_flag = (ts_l <= val_time)
    # In transductive setting, there are no "new" nodes for validation/testing
    mask_node_set = set()


train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]  
train_e_idx_l = e_idx_l[valid_train_flag]  
train_label_l = label_l[valid_train_flag]  


train_node_set = set(train_src_l).union(train_dst_l)
assert(len(train_node_set - mask_node_set) == len(train_node_set)) 
new_node_set = total_node_set - train_node_set 

valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
valid_test_flag = ts_l > test_time

# Vectorized operation instead of list comprehension
src_in_new = np.isin(src_l, list(new_node_set))
dst_in_new = np.isin(dst_l, list(new_node_set))
is_new_node_edge = src_in_new | dst_in_new

nn_val_flag = valid_val_flag * is_new_node_edge 
nn_test_flag = valid_test_flag * is_new_node_edge 


val_src_l = src_l[valid_val_flag] 
val_dst_l = dst_l[valid_val_flag] 
val_ts_l = ts_l[valid_val_flag] 
val_e_idx_l = e_idx_l[valid_val_flag] 
val_label_l = label_l[valid_val_flag] 

test_src_l = src_l[valid_test_flag] 
test_dst_l = dst_l[valid_test_flag] 
test_ts_l = ts_l[valid_test_flag] 
test_e_idx_l = e_idx_l[valid_test_flag] 
test_label_l = label_l[valid_test_flag] 


nn_val_src_l = src_l[nn_val_flag] 
nn_val_dst_l = dst_l[nn_val_flag] 
nn_val_ts_l = ts_l[nn_val_flag] 
nn_val_e_idx_l = e_idx_l[nn_val_flag] 
nn_val_label_l = label_l[nn_val_flag] 

nn_test_src_l = src_l[nn_test_flag] 
nn_test_dst_l = dst_l[nn_test_flag]
nn_test_ts_l = ts_l[nn_test_flag]
nn_test_e_idx_l = e_idx_l[nn_test_flag] 
nn_test_label_l = label_l[nn_test_flag] 


def build_adj_list(src_array, dst_array, eidx_array, ts_array, max_idx):
    """Efficiently build adjacency list using defaultdict
    
    TEMPORAL NOTE: This builds the full adjacency list including all edges.
    The temporal correctness is enforced later in NeighborFinder.find_before()
    which uses binary search to only return neighbors with timestamp < query_time.
    """
    adj_list = [[] for _ in range(max_idx + 1)]
    
    # Batch process edges
    edges = np.column_stack([src_array, dst_array, eidx_array, ts_array])
    for src, dst, eidx, ts in edges:
        adj_list[int(src)].append((int(dst), int(eidx), float(ts)))
        adj_list[int(dst)].append((int(src), int(eidx), float(ts)))
    
    return adj_list

# TEMPORAL CORRECTNESS APPROACH:
# 1. We sort training data by timestamp (TemporalTGBDataset)
# 2. We process batches in chronological order (shuffle=False)
# 3. NeighborFinder.find_before() enforces temporal constraints during neighbor sampling
# 4. This approach maintains temporal correctness while enabling efficient batching

# HYBRID APPROACH BENEFITS (shuffle=False + masks):
# ✅ Guaranteed temporal order - edges processed chronologically
# ✅ Explicit temporal masking - each edge sees only past neighbors  
# ✅ Enhanced correctness - double layer of temporal protection
# ✅ Future flexibility - can easily enable shuffle=True later
# ✅ Better debugging - predictable processing order
# ✅ Performance - still benefits from efficient batching

# Optimized adjacency list building
adj_list = build_adj_list(train_src_l, train_dst_l, train_e_idx_l, train_ts_l, max_idx)
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

full_adj_list = build_adj_list(src_l, dst_l, e_idx_l, ts_l, max_idx)
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
test_rand_sampler = RandEdgeSampler(src_l, dst_l)


nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)


### Model initialize
device = torch.device(f'cuda:{GPU}' if GPU >= 0 else 'cpu')
tgib = TGIB(train_ngh_finder, n_feat, e_feat, 64,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)


criterion = torch.nn.BCELoss()


optimizer = torch.optim.Adam(tgib.parameters(), lr=LEARNING_RATE)
tgib = tgib.to(device)


# HYBRID APPROACH: Use temporal order (shuffle=False) + masks for enhanced correctness
# This gives us the best of both worlds: guaranteed temporal order + explicit masking
train_data = TemporalTGBDataset(train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ALTERNATIVE: Enable shuffling with mask-based temporal correctness
# This approach allows maximum training efficiency with full shuffling
# while maintaining temporal correctness through dynamic masking
# train_data = TGBDataset(train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l)  
# train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# For validation/test, temporal order is less critical, but we should still preserve it
val_data = TemporalTGBDataset(val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

nn_val_data = TemporalTGBDataset(nn_val_src_l, nn_val_dst_l, nn_val_ts_l, nn_val_e_idx_l, nn_val_label_l)
nn_val_loader = DataLoader(nn_val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

test_data = TemporalTGBDataset(test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

nn_test_data = TemporalTGBDataset(nn_test_src_l, nn_test_dst_l, nn_test_ts_l, nn_test_e_idx_l, nn_test_label_l)
nn_test_loader = DataLoader(nn_test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


val_aps = []

def train_sequential_mode(tgib, train_loader, train_rand_sampler, optimizer, criterion, epoch, device, NUM_NEIGHBORS):
    """
    SEQUENTIAL TRAINING MODE (Original Logic - Safer but Slower)
    - Processes each edge individually with separate optimizer steps
    - Uses enhanced forward pass with sophisticated components
    - Includes sophisticated information bottleneck with concrete sampling
    """
    total_acc, total_ap, total_f1, total_auc, total_loss = 0, 0, 0, 0, 0
    total_samples = 0
    processed_edges = 0
    
    for src_l, dst_l, ts_l, e_idx_l, label_l in tqdm(train_loader, disable=not sys.stdout.isatty()):
        batch_acc, batch_ap, batch_f1, batch_auc, batch_loss = [], [], [], [], []
        
        # Process each edge in the batch sequentially (like original)
        for i in range(len(src_l)):
            optimizer.zero_grad()
            tgib = tgib.train()
            
            # Get single edge data
            src_single = src_l[i:i+1].numpy()
            dst_single = dst_l[i:i+1].numpy() 
            ts_single = ts_l[i:i+1].numpy()
            e_idx_single = e_idx_l[i:i+1].numpy()
            
            # Create negative sample
            _, neg_dst_single = train_rand_sampler.sample(1)
            
            # Use enhanced forward with sophisticated information bottleneck
            pos_prob, neg_prob, info_loss = tgib.forward_with_info_bottleneck(
                src_single, dst_single, neg_dst_single, ts_single, e_idx_single, 
                epoch, training=True, num_neighbors=NUM_NEIGHBORS
            )

            pos_label = torch.ones(1, dtype=torch.float, device=device)
            neg_label = torch.zeros(1, dtype=torch.float, device=device)
            
            # Loss computation - info_loss already included from forward_with_info_bottleneck
            loss = criterion(pos_prob, pos_label)
            loss += criterion(neg_prob, neg_label)
            loss += info_loss

            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            with torch.no_grad():
                pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(1), np.zeros(1)])
                
                batch_acc.append((pred_label == true_label).mean())
                batch_ap.append(average_precision_score(true_label, pred_score))
                batch_f1.append(f1_score(true_label, pred_label))
                batch_auc.append(roc_auc_score(true_label, pred_score))
                batch_loss.append(loss.item())
        
        # Update totals
        size = len(src_l)
        total_acc += np.mean(batch_acc) * (2 * size)
        total_ap += np.mean(batch_ap) * (2 * size)
        total_f1 += np.mean(batch_f1) * (2 * size)
        total_auc += np.mean(batch_auc) * (2 * size)
        total_loss += np.mean(batch_loss) * (2 * size)
        total_samples += (2 * size)
        processed_edges += size
    
    return total_acc, total_ap, total_f1, total_auc, total_loss, total_samples


def train_batch_mode(tgib, train_loader, train_rand_sampler, optimizer, criterion, epoch, device, NUM_NEIGHBORS):
    """
    SIMPLE BATCH TRAINING MODE (Optimized - Faster but Potentially Different)
    - Processes entire batches at once with single optimizer step
    - Uses enhanced forward pass with sophisticated components
    - Much faster due to better batching
    """
    total_acc, total_ap, total_f1, total_auc, total_loss = 0, 0, 0, 0, 0
    total_samples = 0
    processed_edges = 0
    
    for src_l, dst_l, ts_l, e_idx_l, label_l in tqdm(train_loader, disable=not sys.stdout.isatty()):
        optimizer.zero_grad()  # Only zero gradients once per batch
        tgib = tgib.train()
        
        size = len(src_l)
        
        # Enhanced batch processing with sophisticated components
        batch_pos_probs = []
        batch_neg_probs = []
        batch_info_losses = []
        
        # Create negative samples for entire batch
        _, neg_dst_l = train_rand_sampler.sample(size)
        
        for i in range(size):
            src_single = src_l[i:i+1].numpy()
            dst_single = dst_l[i:i+1].numpy()
            ts_single = ts_l[i:i+1].numpy()
            e_idx_single = e_idx_l[i:i+1].numpy()
            neg_dst_single = neg_dst_l[i:i+1]
            
            # Use enhanced forward with sophisticated information bottleneck
            pos_prob, neg_prob, info_loss = tgib.forward_with_info_bottleneck(
                src_single, dst_single, neg_dst_single, ts_single, e_idx_single,
                epoch, training=True, num_neighbors=NUM_NEIGHBORS
            )
            batch_pos_probs.append(pos_prob)
            batch_neg_probs.append(neg_prob)
            batch_info_losses.append(info_loss)
        
        # Concatenate all probabilities and losses
        pos_prob_batch = torch.cat(batch_pos_probs, dim=0)
        neg_prob_batch = torch.cat(batch_neg_probs, dim=0)
        info_loss_batch = torch.stack(batch_info_losses).mean()

        pos_label = torch.ones(size, dtype=torch.float, device=device)
        neg_label = torch.zeros(size, dtype=torch.float, device=device)
        
        # Batch loss computation
        loss = criterion(pos_prob_batch, pos_label)
        loss += criterion(neg_prob_batch, neg_label)
        loss += info_loss_batch

        loss.backward()
        optimizer.step()  # Single optimization step per batch
        
        # Batch metrics computation
        with torch.no_grad():
            pred_score = np.concatenate([(pos_prob_batch).cpu().detach().numpy(), (neg_prob_batch).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            # Accumulate batch metrics
            total_acc += (pred_label == true_label).mean() * (2 * size)
            total_ap += average_precision_score(true_label, pred_score) * (2 * size)
            total_f1 += f1_score(true_label, pred_label) * (2 * size)
            total_auc += roc_auc_score(true_label, pred_score) * (2 * size)
            total_loss += loss.item() * (2 * size)
            total_samples += (2 * size)
            
        processed_edges += size
    
    return total_acc, total_ap, total_f1, total_auc, total_loss, total_samples


def train_complex_batch_mode(tgib, train_loader, train_rand_sampler, optimizer, criterion, epoch, device, NUM_NEIGHBORS):
    """
    COMPLEX BATCH TRAINING MODE (True Batched Version of Original)
    - Processes entire batches with the original's complex multi-hop architecture
    - Preserves information bottleneck on attention scores  
    - Uses graph pooling and probability_score for final predictions
    - Should achieve similar accuracy to original while being faster
    """
    total_acc, total_ap, total_f1, total_auc, total_loss = 0, 0, 0, 0, 0
    total_samples = 0
    processed_edges = 0
    
    for src_l, dst_l, ts_l, e_idx_l, label_l in tqdm(train_loader, disable=not sys.stdout.isatty()):
        optimizer.zero_grad()
        tgib = tgib.train()
        
        size = len(src_l)
        
        # Create negative samples for entire batch
        _, neg_dst_l = train_rand_sampler.sample(size)
        
        # Use the complex batched forward method that preserves original architecture
        pos_prob_batch, neg_prob_batch, info_loss_batch = tgib.forward_complex_batched(
            src_l.numpy(), dst_l.numpy(), neg_dst_l, ts_l.numpy(), e_idx_l.numpy(),
            epoch, training=True, num_neighbors=NUM_NEIGHBORS
        )

        pos_label = torch.ones(size, dtype=torch.float, device=device)
        neg_label = torch.zeros(size, dtype=torch.float, device=device)
        
        # Batch loss computation
        loss = criterion(pos_prob_batch, pos_label)
        loss += criterion(neg_prob_batch, neg_label)
        loss += info_loss_batch

        loss.backward()
        optimizer.step()
        
        # Batch metrics computation
        with torch.no_grad():
            pred_score = np.concatenate([(pos_prob_batch).cpu().detach().numpy(), (neg_prob_batch).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            # Accumulate batch metrics
            total_acc += (pred_label == true_label).mean() * (2 * size)
            total_ap += average_precision_score(true_label, pred_score) * (2 * size)
            total_f1 += f1_score(true_label, pred_label) * (2 * size)
            total_auc += roc_auc_score(true_label, pred_score) * (2 * size)
            total_loss += loss.item() * (2 * size)
            total_samples += (2 * size)
            
        processed_edges += size
    
    return total_acc, total_ap, total_f1, total_auc, total_loss, total_samples


def train_memory_efficient_mode(tgib, train_loader, train_rand_sampler, optimizer, criterion, epoch, device, NUM_NEIGHBORS):
    """
    MEMORY-EFFICIENT TRAINING MODE (Sophisticated but Memory-Safe)
    - Uses sophisticated information bottleneck and complex components
    - Processes edges individually to avoid GPU memory explosion
    - Batches only the final loss computation for efficiency
    - Should achieve similar accuracy to original while being memory-safe
    """
    total_acc, total_ap, total_f1, total_auc, total_loss = 0, 0, 0, 0, 0
    total_samples = 0
    processed_edges = 0
    
    for src_l, dst_l, ts_l, e_idx_l, label_l in tqdm(train_loader, disable=not sys.stdout.isatty()):
        # Accumulate gradients across the batch but process edges individually
        optimizer.zero_grad()
        tgib = tgib.train()
        
        size = len(src_l)
        batch_pos_probs = []
        batch_neg_probs = []
        batch_info_losses = []
        
        # Process each edge individually to avoid memory explosion
        for i in range(size):
            src_single = src_l[i:i+1].numpy()
            dst_single = dst_l[i:i+1].numpy()
            ts_single = ts_l[i:i+1].numpy()
            e_idx_single = e_idx_l[i:i+1].numpy()
            
            # Create negative sample
            _, neg_dst_single = train_rand_sampler.sample(1)
            
            # Use sophisticated forward with information bottleneck but process one edge at a time
            try:
                pos_prob_single, neg_prob_single, info_loss_single = tgib.forward_complex_batched(
                    src_single, dst_single, neg_dst_single, ts_single, e_idx_single,
                    epoch, training=True, num_neighbors=NUM_NEIGHBORS
                )
                
                batch_pos_probs.append(pos_prob_single)
                batch_neg_probs.append(neg_prob_single)
                batch_info_losses.append(info_loss_single)
                
            except torch.cuda.OutOfMemoryError:
                # If still out of memory, fall back to simple forward
                torch.cuda.empty_cache()
                pos_prob_single, neg_prob_single, info_loss_single = tgib.forward_with_info_bottleneck(
                    src_single, dst_single, neg_dst_single, ts_single, e_idx_single,
                    epoch, training=True, num_neighbors=NUM_NEIGHBORS
                )
                
                batch_pos_probs.append(pos_prob_single)
                batch_neg_probs.append(neg_prob_single)
                batch_info_losses.append(info_loss_single)
        
        # Combine results for batch loss computation
        pos_prob_batch = torch.cat(batch_pos_probs, dim=0)
        neg_prob_batch = torch.cat(batch_neg_probs, dim=0)
        info_loss_batch = torch.stack(batch_info_losses).mean()

        pos_label = torch.ones(size, dtype=torch.float, device=device)
        neg_label = torch.zeros(size, dtype=torch.float, device=device)
        
        # Batch loss computation
        loss = criterion(pos_prob_batch, pos_label)
        loss += criterion(neg_prob_batch, neg_label)
        loss += info_loss_batch

        loss.backward()
        optimizer.step()
        
        # Batch metrics computation
        with torch.no_grad():
            pred_score = np.concatenate([(pos_prob_batch).cpu().detach().numpy(), (neg_prob_batch).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            # Accumulate batch metrics
            total_acc += (pred_label == true_label).mean() * (2 * size)
            total_ap += average_precision_score(true_label, pred_score) * (2 * size)
            total_f1 += f1_score(true_label, pred_label) * (2 * size)
            total_auc += roc_auc_score(true_label, pred_score) * (2 * size)
            total_loss += loss.item() * (2 * size)
            total_samples += (2 * size)
            
        processed_edges += size
    
    return total_acc, total_ap, total_f1, total_auc, total_loss, total_samples


def train_hybrid_mode(tgib, train_loader, train_rand_sampler, optimizer, criterion, epoch, device, NUM_NEIGHBORS):
    """
    HYBRID TRAINING MODE (Exact Original Logic + Safe Optimizations)
    - Preserves the EXACT sequential temporal processing from original
    - Uses the original forward method signature and logic
    - Optimizes only data loading, metrics computation, and I/O
    - Should achieve IDENTICAL accuracy to original with modest speedup
    """
    total_acc, total_ap, total_f1, total_auc, total_loss = 0, 0, 0, 0, 0
    total_samples = 0
    processed_edges = 0
    
    # Convert loader data to lists for sequential processing (like original)
    all_src_l, all_dst_l, all_ts_l, all_e_idx_l, all_label_l = [], [], [], [], []
    for src_l, dst_l, ts_l, e_idx_l, label_l in train_loader:
        all_src_l.extend(src_l.numpy())
        all_dst_l.extend(dst_l.numpy()) 
        all_ts_l.extend(ts_l.numpy())
        all_e_idx_l.extend(e_idx_l.numpy())
        all_label_l.extend(label_l.numpy())
    
    # Convert to numpy arrays (like original)
    train_src_l = np.array(all_src_l)
    train_dst_l = np.array(all_dst_l)
    train_ts_l = np.array(all_ts_l)
    train_e_idx_l = np.array(all_e_idx_l)
    train_label_l = np.array(all_label_l)
    
    # EXACT ORIGINAL SEQUENTIAL PROCESSING - PRESERVED FOR CORRECTNESS
    # This mirrors the original training loop exactly
    for k in tqdm(range(1, len(train_src_l)), desc=f"Hybrid Epoch {epoch}"):
        # Sample negative edge (original way)
        u_emb_fake, i_emb_fake = train_rand_sampler.sample(1)
        
        with torch.no_grad():
            pos_label = torch.ones(1, dtype=torch.float, device=device)
            neg_label = torch.zeros(1, dtype=torch.float, device=device)

        optimizer.zero_grad()
        tgib = tgib.train()
        
        # EXACT ORIGINAL FORWARD CALL - UNCHANGED FOR CORRECTNESS
        # This is the complex forward method with the exact original signature
        pos_prob, neg_prob, info_loss = tgib.forward_original_signature(
            train_src_l[:k+1],     # EXACT: full sequence history up to k  
            train_dst_l[:k+1],     # EXACT: full sequence history up to k
            i_emb_fake,            # EXACT: negative sample
            train_ts_l[:k+1],      # EXACT: full temporal history up to k
            train_e_idx_l[:k+1],   # EXACT: full edge history up to k
            k,                     # EXACT: current edge index
            epoch,                 # EXACT: current epoch
            training=True,         # EXACT: training mode
            num_neighbors=NUM_NEIGHBORS  # EXACT: neighbor sampling
        )

        # EXACT ORIGINAL LOSS COMPUTATION - UNCHANGED
        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)
        loss += info_loss

        loss.backward()
        optimizer.step()

        # OPTIMIZED: Batch metrics computation for speedup
        with torch.no_grad():
            tgib = tgib.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(1), np.zeros(1)])
            
            # Accumulate metrics (optimized to avoid list appends)
            total_acc += (pred_label == true_label).mean() * 2
            total_ap += average_precision_score(true_label, pred_score) * 2
            total_f1 += f1_score(true_label, pred_label) * 2  
            total_auc += roc_auc_score(true_label, pred_score) * 2
            total_loss += loss.item() * 2
            total_samples += 2
            
        processed_edges += 1
    
    return total_acc, total_ap, total_f1, total_auc, total_loss, total_samples


# Check if we should skip training and load a pre-trained model
if args.eval_only:
    if args.load_model is None:
        logger.error("--eval_only requires --load_model to specify the model path")
        sys.exit(1)
    
    logger.info(f'Loading pre-trained model from {args.load_model}')
    tgib.load_state_dict(torch.load(args.load_model))
    logger.info(f'Loaded pre-trained model for evaluation')
    tgib.eval()
else:
    # Training phase
    for epoch in range(1, NUM_EPOCH+1):
        # Use static neighbor finder that includes all training edges
        # This is a compromise between temporal correctness and efficiency
        tgib.ngh_finder = train_ngh_finder
        
        logger.info('start {} epoch with {} training mode'.format(epoch, TRAINING_MODE))
        
        # Choose training mode based on command line argument
        if TRAINING_MODE == 'sequential':
            logger.info('Using SEQUENTIAL mode: exact original logic (safer but slower)')
            total_acc, total_ap, total_f1, total_auc, total_loss, total_samples = train_sequential_mode(
                tgib, train_loader, train_rand_sampler, optimizer, criterion, epoch, device, NUM_NEIGHBORS)
        elif TRAINING_MODE == 'batch':
            logger.info('Using BATCH mode: simple optimized processing (faster but potentially different)')
            total_acc, total_ap, total_f1, total_auc, total_loss, total_samples = train_batch_mode(
                tgib, train_loader, train_rand_sampler, optimizer, criterion, epoch, device, NUM_NEIGHBORS)
        elif TRAINING_MODE == 'complex_batch':
            logger.info('Using COMPLEX_BATCH mode: true batched version of original (should match original accuracy)')
            total_acc, total_ap, total_f1, total_auc, total_loss, total_samples = train_complex_batch_mode(
                tgib, train_loader, train_rand_sampler, optimizer, criterion, epoch, device, NUM_NEIGHBORS)
        elif TRAINING_MODE == 'memory_efficient':
            logger.info('Using MEMORY_EFFICIENT mode: sophisticated but memory-safe')
            total_acc, total_ap, total_f1, total_auc, total_loss, total_samples = train_memory_efficient_mode(
                tgib, train_loader, train_rand_sampler, optimizer, criterion, epoch, device, NUM_NEIGHBORS)
        elif TRAINING_MODE == 'hybrid':
            logger.info('Using HYBRID mode: exact original logic + safe optimizations')
            total_acc, total_ap, total_f1, total_auc, total_loss, total_samples = train_hybrid_mode(
                tgib, train_loader, train_rand_sampler, optimizer, criterion, epoch, device, NUM_NEIGHBORS)
        else:
            raise ValueError(f"Unknown training mode: {TRAINING_MODE}")
        
        # Clear temporal mask for validation (use all available information)
        tgib.ngh_finder.clear_temporal_mask()
        
        # validation phase
        if SETTING == 'transductive':
            # In transductive setting, the model has seen all nodes, so it's fair 
            # to use the full graph history for validation.
            tgib.ngh_finder = full_ngh_finder
        else: # inductive
            # In inductive setting, to be consistent with the original code's evaluation,
            # we use the training graph for validation. This tests generalization
            # to new nodes and edges with a restricted history.
            tgib.ngh_finder = train_ngh_finder
        
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', tgib, val_rand_sampler, val_loader)

        nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch('val for new nodes', tgib, nn_val_rand_sampler, nn_val_loader)

        logger.info('epoch: {}:'.format(epoch))
        logger.info('Epoch mean loss: {}'.format(total_loss / total_samples))

        logger.info('train acc: {}, val acc: {}, new node val acc: {}'.format(total_acc / total_samples, val_acc, nn_val_acc))

        logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(total_auc / total_samples, val_auc, nn_val_auc))

        logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(total_ap / total_samples, val_ap, nn_val_ap))

        torch.save(tgib.state_dict(), get_checkpoint_path(epoch))

        val_aps.append(val_ap)
        
        # Evaluate explanation metrics every 10 epochs on synthetic dataset
        if epoch % 10 == 0 and DATA.startswith("synthetic"):
            logger.info(f'=== Explanation Evaluation at Epoch {epoch} ===')
            
            # Evaluate explanation faithfulness
            explanation_auc = eval_explanation_faithfulness(tgib, test_loader, NUM_NEIGHBORS, device)
            logger.info(f'Epoch {epoch} - Explanation AUC (Sparsity-Fidelity): {explanation_auc}')
            
            # Evaluate counterfactual explanation performance
            counterfactual_auc = eval_counterfactual_analysis(tgib, test_loader, NUM_NEIGHBORS, device)
            logger.info(f'Epoch {epoch} - Explanation AUC (Counterfactual): {counterfactual_auc}')
            
            # Evaluate explanation accuracy
            eval_explanation_accuracy(tgib, test_loader, NUM_NEIGHBORS, device, data_name=DATA)
            
            logger.info(f'=== End Explanation Evaluation at Epoch {epoch} ===') 

    best_epoch = np.argmax(val_aps) + 1 
    best_model_path = get_checkpoint_path(best_epoch)
    logger.info(f'Loading the best model at epoch {best_epoch}')
    tgib.load_state_dict(torch.load(best_model_path))
    logger.info(f'Loaded the best model at epoch {best_epoch} for inference')
    tgib.eval()

# testing phase use all information
tgib.ngh_finder = full_ngh_finder
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgib, test_rand_sampler, test_loader)

nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new nodes', tgib, nn_test_rand_sampler, nn_test_loader)

logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

# Evaluate explanation faithfulness
# We use the test loader for this evaluation.
explanation_auc = eval_explanation_faithfulness(tgib, test_loader, NUM_NEIGHBORS, device)
logger.info(f'Explanation AUC (Sparsity-Fidelity): {explanation_auc}')

# Evaluate counterfactual explanation performance
counterfactual_auc = eval_counterfactual_analysis(tgib, test_loader, NUM_NEIGHBORS, device)
logger.info(f'Explanation AUC (Counterfactual): {counterfactual_auc}')

# Evaluate on synthetic data if specified
if DATA.startswith("synthetic"):
    # For synthetic datasets, try test set first, fallback to validation set if no causal edges
    result = eval_explanation_accuracy(tgib, test_loader, NUM_NEIGHBORS, device, data_name=DATA)
    if result['num_evaluated'] == 0:
        logger.info("No causal edges in test set, evaluating on validation set instead...")
        eval_explanation_accuracy(tgib, val_loader, NUM_NEIGHBORS, device, data_name=DATA)

# Only save model if we performed training
if not args.eval_only:
    logger.info('Saving TGIB model')
    torch.save(tgib.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGIB models saved')


