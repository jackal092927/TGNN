import os, json, math
import logging
import argparse
import torch, torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from graphmamba_IB_explain import GraphMamba, build_sequence_with_recency_and_features, visualize_topk_gates_time_series
from graphmamba_IB_explain import save_checkpoint, dump_predictions_csv, dump_gates_csv, explanation_only_eval

def load_contagion_data(data_name):
    path=f'./processed/{data_name}/ml_{data_name}.csv'
    if not os.path.exists(path): raise FileNotFoundError(path)
    g_df = pd.read_csv(path); 
    if 'label' not in g_df.columns: print("Warning: 'label' column missing")
    return g_df

def evaluate_contagion_prediction(model, graph_sequence, g_df, timestamps, device):
    model.eval(); all_p, all_l = [], []
    with torch.no_grad():
        seq = model.forward_sequence(graph_sequence)
        for i in range(len(timestamps)-1):
            next_ts=timestamps[i+1]; cur=seq[i]
            nxt=g_df[g_df['ts']==next_ts]
            if len(nxt)==0: continue
            N=cur.shape[0]; pairs=[(u,v) for u in range(N) for v in range(u+1,N)]
            pos,neg=[],[]
            for (u,v) in pairs:
                ex=len(nxt[(nxt['u']==u)&(nxt['i']==v)])>0 or len(nxt[(nxt['u']==v)&(nxt['i']==u)])>0
                (pos if ex else neg).append((u,v))
            if not pos or not neg: continue
            m=min(len(pos),len(neg))
            sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
            prs=pos+neg_s; labs=[1.0]*len(pos)+[0.0]*len(neg_s)
            prd=model.predict_next_edges(cur, torch.tensor(prs, device=device)).detach().cpu().numpy().tolist()
            all_p+=prd; all_l+=labs
    if not all_p: return {"accuracy":0.0, "auc":0.5, "ap":0.0}
    preds=np.array(all_p); labs=np.array(all_l)
    return {"accuracy": float((preds>0.5).mean()), "auc": float(roc_auc_score(labs,preds)), "ap": float(average_precision_score(labs,preds)), "num": int(len(preds))}

def train_graphmamba_contagion(data_name='synthetic_icm_ba', epochs=50, lr=1e-3, hidden_dim=64, pos_dim=128, mamba_state_dim=16, gpu_id=0,
                               lambda_sparse: float = 0.0, lambda_tv: float = 1e-3, gate_temperature: float = 1.0,
                               beta_ib: float = 1e-3, r0: float = 0.05, alpha_prior: float = 0.7, gumbel_tau: float = 0.5, stochastic_gates: bool = True,
                               save_dir: str = '/mnt/data/experiments/contagion'):
    logging.basicConfig(level=logging.INFO); logger=logging.getLogger(__name__)
    g_df = load_contagion_data(data_name); timestamps = sorted(g_df['ts'].unique())
    graph_sequence, recency_priors, edge_feature_seq = build_sequence_with_recency_and_features(g_df, timestamps, tau_prior=1.0, include_delta_feature=True)
    max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    model = GraphMamba(max_nodes=max_nodes, pos_dim=pos_dim, hidden_dim=hidden_dim, gnn_layers=2, mamba_state_dim=mamba_state_dim,
                       dropout=0.1, use_edge_gates=True, gate_temperature=gate_temperature, edge_feat_dim=1).to(device)

    graph_sequence=[A.to(device) for A in graph_sequence]; recency_priors=[R.to(device) for R in recency_priors]; edge_feature_seq=[E.to(device) for E in edge_feature_seq]
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5); bce=nn.BCELoss()

    tr=int(len(timestamps)*0.7); va=int(len(timestamps)*0.15); train_ts=timestamps[:tr]; val_ts=timestamps[tr:tr+va]
    best_val_ap=0.0; best=None

    for epoch in range(epochs):
        model.train(); running=0.0; steps=0
        for i in range(len(train_ts)-1):
            seq = graph_sequence[:i+2]; rp=recency_priors[:i+2]; ef=edge_feature_seq[:i+2]
            seq_emb, gates_list, prob_list = model.forward_sequence(seq, return_gates=True, recency_priors=rp, edge_feature_seq=ef,
                                                                    stochastic=stochastic_gates, gumbel_tau=gumbel_tau)
            cur=seq_emb[i]; probs_t=prob_list[i]; probs_prev=prob_list[i-1] if i>0 else None
            next_ts=train_ts[i+1]; nxt=g_df[g_df['ts']==next_ts]
            if len(nxt)==0: continue
            N=cur.shape[0]; pairs=[(u,v) for u in range(N) for v in range(u+1,N)]
            pos,neg=[],[]
            for (u,v) in pairs:
                ex=len(nxt[(nxt['u']==u)&(nxt['i']==v)])>0 or len(nxt[(nxt['u']==v)&(nxt['i']==u)])>0
                (pos if ex else neg).append((u,v))
            if not pos or not neg: continue
            m=min(len(pos),len(neg))
            sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
            prs=pos+neg_s; labs=torch.tensor([1.0]*len(pos)+[0.0]*len(neg_s), device=device)
            prd=model.predict_next_edges(cur, torch.tensor(prs, device=device))
            loss_pred=bce(prd,labs)
            r_t=(alpha_prior*recency_priors[i] + (1-alpha_prior)*r0)
            loss_ib = GraphMamba.bernoulli_kl_loss(probs_t, r_t)
            loss_tv = GraphMamba.temporal_tv_loss(probs_t, probs_prev) if (i>0 and lambda_tv>0) else (probs_t.sum()*0)
            loss_sparse = GraphMamba.sparsity_loss(probs_t) if lambda_sparse>0 else (probs_t.sum()*0)
            loss = loss_pred + beta_ib*loss_ib + lambda_tv*loss_tv + lambda_sparse*loss_sparse
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            running += float(loss.item()); steps += 1
        avg=running/max(steps,1)
        # Validate
        val_seq = graph_sequence[:tr+va+1]
        with torch.no_grad(): seq = model.forward_sequence(val_seq, recency_priors=recency_priors[:tr+va+1], edge_feature_seq=edge_feature_seq[:tr+va+1])
        all_p, all_l = [], []
        for i in range(len(val_ts)-1):
            next_ts=val_ts[i+1]; cur=seq[i]; nxt=g_df[g_df['ts']==next_ts]
            if len(nxt)==0: continue
            N=cur.shape[0]; pairs=[(u,v) for u in range(N) for v in range(u+1,N)]
            pos,neg=[],[]
            for (u,v) in pairs:
                ex=len(nxt[(nxt['u']==u)&(nxt['i']==v)])>0 or len(nxt[(nxt['u']==v)&(nxt['i']==u)])>0
                (pos if ex else neg).append((u,v))
            if not pos or not neg: continue
            m=min(len(pos),len(neg))
            sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
            prs=pos+neg_s; labs=[1.0]*len(pos)+[0.0]*len(neg_s)
            prd = model.predict_next_edges(cur, torch.tensor(prs, device=device)).detach().cpu().numpy().tolist()
            all_p+=prd; all_l+=labs
        ap = float(average_precision_score(all_l, all_p)) if all_p else 0.0
        print(f"Epoch {epoch:03d} loss={avg:.4f} | Val AP={ap:.4f}")
        if ap>best_val_ap: best_val_ap=ap

    # Save artifacts
    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint(model, os.path.join(save_dir, f'{data_name}_best.pt'))
    # Dump predictions and gates
    with torch.no_grad():
        seq = model.forward_sequence(graph_sequence, recency_priors=recency_priors, edge_feature_seq=edge_feature_seq)
        _, gates_full, _ = model.forward_sequence(graph_sequence, return_gates=True, recency_priors=recency_priors, edge_feature_seq=edge_feature_seq)
    all_pairs, all_preds, all_labels, all_times = [], [], [], []
    for i in range(len(timestamps)-1):
        next_ts=timestamps[i+1]; cur=seq[i]; N=cur.shape[0]; nxt=g_df[g_df['ts']==next_ts]
        if len(nxt)==0: continue
        pairs=[(u,v) for u in range(N) for v in range(u+1,N)]; pos,neg=[],[]
        for (u,v) in pairs:
            ex=len(nxt[(nxt['u']==u)&(nxt['i']==v)])>0 or len(nxt[(nxt['u']==v)&(nxt['i']==u)])>0
            (pos if ex else neg).append((u,v))
        if not pos or not neg: continue
        m=min(len(pos),len(neg))
        sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
        prs=pos+neg_s; labs=[1.0]*len(pos)+[0.0]*len(neg_s)
        prd = model.predict_next_edges(cur, torch.tensor(prs, device=device)).detach().cpu().numpy().tolist()
        all_pairs+=prs; all_preds+=prd; all_labels+=labs; all_times+=[next_ts]*len(prs)
    dump_predictions_csv(all_pairs, all_preds, all_labels, all_times, os.path.join(save_dir, f'{data_name}_predictions.csv'))
    dump_gates_csv(gates_full, timestamps, os.path.join(save_dir, f'{data_name}_gates_top100.csv'), topk=100)
    exp_only = explanation_only_eval(model, graph_sequence, recency_priors, edge_feature_seq, g_df, timestamps, device)
    with open(os.path.join(save_dir, f'{data_name}_exp_only_ap.json'),'w') as f: json.dump(exp_only, f, indent=2)
    # quick viz
    visualize_topk_gates_time_series(gates_full, k=8, timestamps=timestamps, savepath=os.path.join(save_dir,'topk_gates_timeseries.png'))
    return model

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='synthetic_icm_ba')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--pos_dim', type=int, default=128)
    p.add_argument('--mamba_state_dim', type=int, default=16)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--lambda_sparse', type=float, default=0.0)
    p.add_argument('--lambda_tv', type=float, default=1e-3)
    p.add_argument('--gate_temperature', type=float, default=1.0)
    p.add_argument('--beta_ib', type=float, default=1e-3)
    p.add_argument('--r0', type=float, default=0.05)
    p.add_argument('--alpha_prior', type=float, default=0.7)
    p.add_argument('--gumbel_tau', type=float, default=0.5)
    p.add_argument('--no_stochastic_gates', action='store_true')
    p.add_argument('--save_dir', type=str, default='/mnt/data/experiments/contagion')
    a=p.parse_args()
    train_graphmamba_contagion(a.data, a.epochs, a.lr, a.hidden_dim, a.pos_dim, a.mamba_state_dim, a.gpu,
                               a.lambda_sparse, a.lambda_tv, a.gate_temperature, a.beta_ib, a.r0, a.alpha_prior, a.gumbel_tau, not a.no_stochastic_gates,
                               a.save_dir)