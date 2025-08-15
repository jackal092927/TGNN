"""
Corrected Training script for TGAM that follows TGIB's exact approach:
- Same temporal data splits
- Same negative sampling strategy  
- Same inductive evaluation protocol
- Binary classification: real vs fake edges
"""

import math
import logging
import time
import random
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from tgam_fixed import TGAM_LinkPrediction
from utils import EarlyStopMonitor, RandEdgeSampler


def setup_logging(data_name):
    """Setup logging configuration"""
    Path("log").mkdir(parents=True, exist_ok=True)
    Path("saved_models").mkdir(parents=True, exist_ok=True)
    Path("saved_checkpoints").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'log/tgam_corrected_{data_name}_{int(time.time())}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def eval_one_epoch(hint, tgam, sampler, val_src_l, val_dst_l, val_ts_l, val_e_idx_l, 
                   node_features, edge_features, epoch):
    """Evaluation function that matches TGIB's evaluation"""
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    
    with torch.no_grad():
        tgam.eval()
        for k in range(1, len(val_src_l)):
            # Sample negative edge
            u_emb_fake, i_emb_fake = sampler.sample(1)
            fake_dst = i_emb_fake[0]
            
            # Positive sample: real edge
            pos_prob = tgam(
                val_src_l[:k+1], val_dst_l[:k+1], val_dst_l[k], 
                val_ts_l[:k+1], val_e_idx_l[:k+1], 
                node_features, edge_features
            )
            
            # Negative sample: fake edge  
            neg_prob = tgam(
                val_src_l[:k+1], val_dst_l[:k+1], fake_dst,
                val_ts_l[:k+1], val_e_idx_l[:k+1],
                node_features, edge_features
            )
            
            # Compute metrics
            pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(1), np.zeros(1)])
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)


def main():
    parser = argparse.ArgumentParser('TGAM Corrected Training')
    parser.add_argument('-d', '--data', type=str, help='dataset name', default='wikipedia')
    parser.add_argument('--bs', type=int, default=200, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--hidden_dim', type=int, default=100, help='hidden dimension')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--num_graph_layers', type=int, default=2, help='number of graph layers')
    parser.add_argument('--num_temporal_layers', type=int, default=2, help='number of temporal layers')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.data)
    logger.info(f"Training TGAM on {args.data}")
    logger.info(args)
    
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds (same as TGIB)
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Load data (exactly like TGIB)
    try:
        g_df = pd.read_csv(f'./processed/{args.data}/ml_{args.data}.csv')
        e_feat = np.load(f'./processed/{args.data}/ml_{args.data}.npy')
        n_feat = np.load(f'./processed/{args.data}/ml_{args.data}_node.npy')
        logger.info(f"Loaded data: {len(g_df)} edges, {len(n_feat)} nodes, {e_feat.shape[1]} edge features")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Data preprocessing (exactly like TGIB)
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
    
    # Inductive setting: mask 10% of nodes
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), 
                                     int(0.1 * num_total_unique_nodes)))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values 
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag) 
    
    valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)
    
    # Training data
    train_src_l = src_l[valid_train_flag] 
    train_dst_l = dst_l[valid_train_flag] 
    train_ts_l = ts_l[valid_train_flag]  
    train_e_idx_l = e_idx_l[valid_train_flag]  
    train_label_l = label_l[valid_train_flag]  
    
    train_node_set = set(train_src_l).union(train_dst_l) 
    assert(len(train_node_set - mask_node_set) == len(train_node_set)) 
    new_node_set = total_node_set - train_node_set 
    
    # Validation and test splits
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time
    
    is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
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
    
    # Negative samplers (exactly like TGIB)
    train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
    val_rand_sampler = RandEdgeSampler(src_l, dst_l)
    nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
    test_rand_sampler = RandEdgeSampler(src_l, dst_l)
    nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)
    
    logger.info(f"Train: {len(train_src_l)}, Val: {len(val_src_l)}, Test: {len(test_src_l)}")
    logger.info(f"New node Val: {len(nn_val_src_l)}, New node Test: {len(nn_test_src_l)}")
    
    # Initialize model
    model = TGAM_LinkPrediction(
        node_feat_dim=n_feat.shape[1],
        edge_feat_dim=e_feat.shape[1],
        hidden_dim=args.hidden_dim,
        max_nodes=max_idx + 1,
        num_graph_layers=args.num_graph_layers,
        num_temporal_layers=args.num_temporal_layers
    ).to(device)
    
    # Loss and optimizer (same as TGIB)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Training instances: {len(train_src_l)}")
    
    # Training loop (exactly like TGIB)
    val_aps = []
    
    for epoch in range(1, args.n_epoch + 1):
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        logger.info(f'Start epoch {epoch}')
        
        model.train()
        
        for k in tqdm(range(1, len(train_src_l)), desc=f"Epoch {epoch}/{args.n_epoch}"):
            # Sample negative edge
            u_emb_fake, i_emb_fake = train_rand_sampler.sample(1)
            fake_dst = i_emb_fake[0]
            
            optimizer.zero_grad()
            
            # Positive sample: real edge
            pos_prob = model(
                train_src_l[:k+1], train_dst_l[:k+1], train_dst_l[k],
                train_ts_l[:k+1], train_e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Negative sample: fake edge
            neg_prob = model(
                train_src_l[:k+1], train_dst_l[:k+1], fake_dst,
                train_ts_l[:k+1], train_e_idx_l[:k+1],
                n_feat, e_feat
            )
            
            # Loss computation
            pos_label = torch.ones(1, dtype=torch.float, device=device)
            neg_label = torch.zeros(1, dtype=torch.float, device=device)
            
            loss = criterion(pos_prob, pos_label)
            loss += criterion(neg_prob, neg_label)
            
            loss.backward()
            optimizer.step()
            
            # Training metrics
            with torch.no_grad():
                pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(1), np.zeros(1)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))
        
        # Validation phase
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch(
            'val for old nodes', model, val_rand_sampler, val_src_l, 
            val_dst_l, val_ts_l, val_e_idx_l, n_feat, e_feat, epoch
        )
        
        nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch(
            'val for new nodes', model, val_rand_sampler, nn_val_src_l, 
            nn_val_dst_l, nn_val_ts_l, nn_val_e_idx_l, n_feat, e_feat, epoch
        )
        
        logger.info(f'Epoch {epoch}:')
        logger.info(f'Epoch mean loss: {np.mean(m_loss):.4f}')
        logger.info(f'Train acc: {np.mean(acc):.4f}, val acc: {val_acc:.4f}, new node val acc: {nn_val_acc:.4f}')
        logger.info(f'Train auc: {np.mean(auc):.4f}, val auc: {val_auc:.4f}, new node val auc: {nn_val_auc:.4f}')
        logger.info(f'Train ap: {np.mean(ap):.4f}, val ap: {val_ap:.4f}, new node val ap: {nn_val_ap:.4f}')
        
        val_aps.append(val_ap)
        
        # Save checkpoint
        torch.save(model.state_dict(), f'saved_checkpoints/tgam_{args.data}_epoch_{epoch}.pth')
    
    # Testing phase
    best_epoch = np.argmax(val_aps) + 1
    best_model_path = f'saved_checkpoints/tgam_{args.data}_epoch_{best_epoch}.pth'
    logger.info(f'Loading the best model at epoch {best_epoch}')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch(
        'test for old nodes', model, test_rand_sampler, test_src_l, 
        test_dst_l, test_ts_l, test_e_idx_l, n_feat, e_feat, best_epoch
    )
    
    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch(
        'test for new nodes', model, nn_test_rand_sampler, nn_test_src_l, 
        nn_test_dst_l, nn_test_ts_l, nn_test_e_idx_l, n_feat, e_feat, best_epoch
    )
    
    logger.info('Test statistics: Old nodes -- acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format(test_acc, test_auc, test_ap))
    logger.info('Test statistics: New nodes -- acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format(nn_test_acc, nn_test_auc, nn_test_ap))
    
    # Save final model
    torch.save(model.state_dict(), f'saved_models/tgam_{args.data}_final.pth')
    logger.info('TGAM model saved')


if __name__ == '__main__':
    main() 