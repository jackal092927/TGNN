import math, os, json
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------ Mamba Block ------------------------------

class MambaBlock(nn.Module):
    def __init__(self, hidden_dim, state_dim=16, expand_factor=2, dt_rank=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.expand_dim = int(hidden_dim*expand_factor)
        self.dt_rank = dt_rank or max(1, hidden_dim//16)
        self.input_proj = nn.Linear(hidden_dim, self.expand_dim*2)
        self.dt_proj = nn.Linear(self.dt_rank, self.expand_dim)
        self.A_log = nn.Parameter(torch.randn(self.expand_dim, state_dim))
        self.D = nn.Parameter(torch.randn(self.expand_dim))
        self.delta_proj = nn.Linear(self.expand_dim, self.dt_rank)
        self.B_proj = nn.Linear(self.expand_dim, state_dim)
        self.C_proj = nn.Linear(self.expand_dim, state_dim)
        self.output_proj = nn.Linear(self.expand_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self._init()

    def _init(self):
        nn.init.normal_(self.A_log, mean=0, std=0.1)
        with torch.no_grad(): self.A_log.data = -torch.exp(self.A_log.data)
        for m in [self.dt_proj, self.B_proj, self.C_proj, self.delta_proj]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.D)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x_proj = self.input_proj(x)
        x, gate = x_proj.chunk(2, dim=-1)
        x = self.act(x)
        delta = F.softplus(self.dt_proj(self.delta_proj(x)))
        B, C = self.B_proj(x), self.C_proj(x)
        A = -torch.exp(self.A_log)
        y = self._scan(x, delta, A, B, C)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        y = y * self.act(gate)
        return self.output_proj(y) + res

    def _scan(self, x, delta, A, B, C):
        B_, T, D = x.shape
        Sd = A.shape[1]
        h = torch.zeros(B_, D, Sd, device=x.device)
        outs = []
        for t in range(T):
            x_t, d_t = x[:, t], delta[:, t]
            B_t, C_t = B[:, t], C[:, t]
            A_disc = torch.exp(d_t.unsqueeze(-1) * A.unsqueeze(0))
            h = A_disc * h + (d_t.unsqueeze(-1) * B_t.unsqueeze(1)) * x_t.unsqueeze(-1)
            y_t = torch.sum(C_t.unsqueeze(1) * h, dim=-1)
            outs.append(y_t)
        return torch.stack(outs, dim=1)

# -------------------------- Gated Positional GNN --------------------------

class PositionalGNNLayer(nn.Module):
    """
    Aggregation with gates (and priors/features):
        m_i = sum_j (A_ij * prior_ij * g_ij) * msg(h_j)
    logits_ij = <Q m_i, K m_j> / (sqrt(H)*tau) + b + phi(edge_feat_ij)
    g_ij = sigmoid(logits_ij); optionally ST-Gumbel during training.
    Returns both gates used (possibly sampled) and probabilities for losses.
    """
    def __init__(self, input_dim, hidden_dim, use_edge_gates=True, gate_temperature=1.0, edge_feat_dim=0):
        super().__init__()
        self.use_edge_gates = use_edge_gates
        self.gate_temperature = gate_temperature
        self.edge_feat_dim = edge_feat_dim
        self.message_net = nn.Linear(input_dim, hidden_dim)
        self.update_net  = nn.Linear(input_dim + hidden_dim, hidden_dim)
        if use_edge_gates:
            self.gate_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.gate_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.gate_bias = nn.Parameter(torch.tensor(-1.0))
            self.gate_feat = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)) if edge_feat_dim>0 else None
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, adj_matrix, return_gates=False,
                recency_prior=None, edge_features=None,
                stochastic=False, gumbel_tau=1.0):
        messages = self.message_net(node_features)
        if self.use_edge_gates:
            q, k = self.gate_q(messages), self.gate_k(messages)
            scores = (q @ k.T) / (math.sqrt(q.shape[-1]) * max(self.gate_temperature, 1e-6))
            if self.gate_feat is not None and edge_features is not None:
                scores = scores + self.gate_feat(edge_features).squeeze(-1)
            logits = scores + self.gate_bias
            probs = torch.sigmoid(logits)
            if stochastic and self.training:
                g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-9)))
                y = torch.sigmoid((logits + g)/max(gumbel_tau,1e-6))
                gates = (y>0.5).float() + (y - y.detach())
            else:
                gates = probs
            gates_masked = gates * adj_matrix
            probs_masked = probs * adj_matrix
            if recency_prior is not None:
                gates_masked *= recency_prior
                probs_masked *= recency_prior
        else:
            gates_masked = adj_matrix
            probs_masked = None

        aggregated = (gates_masked @ messages)
        updated = self.update_net(torch.cat([node_features, aggregated], dim=1))
        updated = self.act(updated); updated = self.norm(updated)

        if return_gates:
            return updated, gates_masked, probs_masked
        return updated

# ------------------------- Positional Embeddings --------------------------

def create_sincos_positional_embeddings(max_nodes, pos_dim):
    position = torch.arange(max_nodes).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, pos_dim, 2).float() * -(math.log(10000.0)/pos_dim))
    pe = torch.zeros(max_nodes, pos_dim)
    pe[:,0::2] = torch.sin(position*div); pe[:,1::2] = torch.cos(position*div)
    return pe

# ------------------------------- GraphMamba -------------------------------

class GraphMamba(nn.Module):
    def __init__(self, max_nodes, pos_dim=256, hidden_dim=64, gnn_layers=2,
                 mamba_state_dim=16, dropout=0.1, use_edge_gates=True,
                 gate_temperature=1.0, edge_feat_dim=0):
        super().__init__()
        self.max_nodes = max_nodes
        self.use_edge_gates = use_edge_gates
        self.pos_embeddings = nn.Parameter(create_sincos_positional_embeddings(max_nodes, pos_dim), requires_grad=False)
        self.gnn_input = PositionalGNNLayer(pos_dim, hidden_dim, use_edge_gates=use_edge_gates,
                                            gate_temperature=gate_temperature, edge_feat_dim=edge_feat_dim)
        self.gnn_layers_list = nn.ModuleList([PositionalGNNLayer(hidden_dim, hidden_dim, use_edge_gates=False)
                                              for _ in range(gnn_layers-1)])
        self.mamba_encoder = MambaBlock(hidden_dim=hidden_dim, state_dim=mamba_state_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout*0.5),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout*0.5),
            nn.Linear(hidden_dim//2, 1), nn.Sigmoid()
        )
        self.drop = nn.Dropout(dropout)

    def encode_graph(self, adj_matrix, return_gates=False, recency_prior=None, edge_features=None,
                     stochastic=False, gumbel_tau=1.0):
        num_nodes = adj_matrix.shape[0]
        node_pos_emb = self.pos_embeddings[:num_nodes]
        if self.use_edge_gates:
            x, gates, probs = self.gnn_input(node_pos_emb, adj_matrix, return_gates=True,
                                             recency_prior=recency_prior, edge_features=edge_features,
                                             stochastic=stochastic, gumbel_tau=gumbel_tau)
        else:
            x = self.gnn_input(node_pos_emb, adj_matrix, return_gates=False,
                               recency_prior=recency_prior, edge_features=edge_features)
            gates, probs = None, None
        for gnn in self.gnn_layers_list:
            x = gnn(x, adj_matrix); x = self.drop(x)
        return (x, (gates, probs)) if return_gates else (x, (None, None))

    def forward_sequence(self, graph_sequence: List[torch.Tensor], return_gates=False,
                         recency_priors=None, edge_feature_seq=None,
                         stochastic=False, gumbel_tau=1.0):
        seq_embs, gates_list, prob_list = [], [], []
        for t, adj in enumerate(graph_sequence):
            rp = recency_priors[t] if recency_priors is not None else None
            ef = edge_feature_seq[t] if edge_feature_seq is not None else None
            x, (g, p) = self.encode_graph(adj, return_gates=return_gates, recency_prior=rp, edge_features=ef,
                                          stochastic=stochastic, gumbel_tau=gumbel_tau)
            seq_embs.append(x)
            if return_gates and self.use_edge_gates:
                gates_list.append(g); prob_list.append(p)
        seq_embs = torch.stack(seq_embs, dim=0).transpose(0,1)  # [N,T,H]
        N = seq_embs.shape[0]; outs = []
        for i in range(N):
            outs.append(self.mamba_encoder(seq_embs[i:i+1]).squeeze(0))  # [T,H]
        outs = torch.stack(outs, dim=0).transpose(0,1)  # [T,N,H]
        if return_gates and self.use_edge_gates:
            return outs, gates_list, prob_list
        return outs

    def predict_next_edges(self, current_embeddings, edge_pairs):
        src = current_embeddings[edge_pairs[:,0]]; dst = current_embeddings[edge_pairs[:,1]]
        edge_feat = torch.cat([src+dst, (src-dst).abs()], dim=1)
        return self.edge_predictor(edge_feat).squeeze(-1)

    @staticmethod
    def sparsity_loss(gates_masked):  # mean gate on edges
        eps = 1e-8
        num_edges = (gates_masked>0).float().sum()
        if num_edges.item()==0: return gates_masked.sum()*0.0
        return gates_masked.sum()/(num_edges+eps)

    @staticmethod
    def temporal_tv_loss(gates_t, gates_prev):
        common = ((gates_t>0) & (gates_prev>0)).float()
        if common.sum().item()==0: return (gates_t-gates_prev).abs().sum()*0.0
        return ((gates_t-gates_prev).abs()*common).sum()/(common.sum()+1e-8)

    @staticmethod
    def bernoulli_kl_loss(p, r):
        eps = 1e-8
        p = p.clamp(eps,1-eps); r = r.clamp(eps,1-eps)
        kl = p * torch.log(p/r) + (1-p)*torch.log((1-p)/(1-r))
        mask = (r>eps).float()
        if mask.sum().item()==0: return kl.sum()*0.0
        return (kl*mask).sum()/(mask.sum()+eps)

    @staticmethod
    def topk_edges_from_gates(gates_masked, k=20):
        N = gates_masked.shape[0]
        tri = torch.triu(torch.ones(N,N, device=gates_masked.device), diagonal=1)
        vals = (gates_masked*tri).flatten()
        k = min(k, int(N*(N-1)//2))
        if k<=0: return []
        idx = torch.topk(vals, k).indices
        return [ (int(i//N), int(i%N), float(vals[ind].cpu())) for ind,i in zip(idx, idx) ]

# ---------------- Sequence builders, viz, artifacts, eval ----------------

def build_sequence_with_recency_and_features(g_df, timestamps, tau_prior=1.0, include_delta_feature=True):
    max_node = max(g_df['u'].max(), g_df['i'].max()) + 1
    seq, rp_seq, ef_seq = [], [], []
    last_time = {}
    for ts in timestamps:
        A = torch.zeros(max_node, max_node)
        rp = torch.zeros(max_node, max_node)
        ef = torch.zeros(max_node, max_node, 1) if include_delta_feature else None
        cur = g_df[g_df['ts'] <= ts]
        for _,r in cur.iterrows():
            u,v,t = int(r['u']), int(r['i']), float(r['ts'])
            A[u,v]=1.0; A[v,u]=1.0
            key=(min(u,v),max(u,v))
            if key not in last_time or t>last_time[key]: last_time[key]=t
        for (u,v),t_last in last_time.items():
            if A[u,v]>0:
                delta=max(ts-t_last,0.0); prior=math.exp(-delta/max(tau_prior,1e-6))
                rp[u,v]=rp[v,u]=prior
                if ef is not None: ef[u,v,0]=ef[v,u,0]=delta
        seq.append(A); rp_seq.append(rp); 
        if ef is not None: ef_seq.append(ef)
    return seq, rp_seq, ef_seq

def visualize_topk_gates_time_series(gates_list, k=10, timestamps=None, savepath='/mnt/data/topk_gates_timeseries.png'):
    import matplotlib.pyplot as plt
    if timestamps is None: timestamps=list(range(len(gates_list)))
    edge_set=set()
    for G in gates_list:
        N=G.shape[0]
        tri=torch.triu(torch.ones(N,N,device=G.device), diagonal=1)
        vals=(G*tri).flatten()
        kk=min(k,int((N*(N-1))//2)); idx=torch.topk(vals,kk).indices
        for idv in idx.tolist(): edge_set.add((int(idv//N), int(idv%N)))
    edges=sorted(edge_set); T=len(gates_list); import numpy as np
    M=np.zeros((T,len(edges)))
    for t,G in enumerate(gates_list):
        for e_idx,(i,j) in enumerate(edges):
            M[t,e_idx]=float(G[i,j].cpu())
    plt.figure(figsize=(10,5))
    for e_idx,(i,j) in enumerate(edges):
        plt.plot(timestamps, M[:,e_idx], label=f'({i},{j})')
    plt.xlabel('time'); plt.ylabel('gate'); plt.title('Top-k gates over time')
    if len(edges)<=10: plt.legend(loc='best', fontsize='small')
    plt.tight_layout(); plt.savefig(savepath); return savepath

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True); torch.save(model.state_dict(), path)

def dump_predictions_csv(pairs, preds, labels, times, out_csv):
    import csv; os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['time','u','v','pred','label'])
        for t,(u,v),yh,y in zip(times,pairs,preds,labels): w.writerow([t,u,v,float(yh),int(y)])

def dump_gates_csv(gates_list, timestamps, out_csv, topk=None):
    import csv; os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['time','i','j','gate'])
        for G,ts in zip(gates_list,timestamps):
            N=G.shape[0]
            if topk is None:
                tri=torch.triu(torch.ones(N,N,device=G.device), diagonal=1)
                idx=tri.nonzero(as_tuple=False)
                for i,j in idx.tolist(): w.writerow([ts,i,j,float(G[i,j].cpu())])
            else:
                tri=torch.triu(torch.ones(N,N,device=G.device), diagonal=1)
                vals=(G*tri).flatten(); k=min(topk,int((N*(N-1))//2))
                idc=torch.topk(vals,k).indices
                for idv in idc.tolist():
                    i,j=int(idv//N), int(idv%N)
                    w.writerow([ts,i,j,float(G[i,j].cpu())])

def explanation_only_eval(model, graph_sequence, recency_priors, edge_feature_seq, g_df, timestamps, device, sparsity_list=[0.02,0.05,0.1]):
    from sklearn.metrics import average_precision_score
    model.eval()
    with torch.no_grad():
        _, gates_list, _ = model.forward_sequence(graph_sequence, return_gates=True, recency_priors=recency_priors, edge_feature_seq=edge_feature_seq)
    N=graph_sequence[0].shape[0]; results={}
    for rho in sparsity_list:
        masked_seq=[]
        for G,A in zip(gates_list, graph_sequence):
            tri=torch.triu(torch.ones(N,N,device=G.device), diagonal=1)
            vals=(G*tri).flatten(); k=max(1,int(rho*(N*(N-1)//2)))
            idc=torch.topk(vals,k).indices
            M=torch.zeros_like(A)
            for idv in idc.tolist():
                i,j=int(idv//N), int(idv%N); M[i,j]=M[j,i]=1.0
            masked_seq.append(A*M)
        seq_emb = model.forward_sequence(masked_seq, recency_priors=recency_priors, edge_feature_seq=edge_feature_seq)
        all_pred, all_lab = [], []
        for i in range(len(timestamps)-1):
            next_ts=timestamps[i+1]; cur=seq_emb[i]; N=cur.shape[0]
            nxt=g_df[g_df['ts']==next_ts]
            if len(nxt)==0: continue
            pairs=[(u,v) for u in range(N) for v in range(u+1,N)]
            pos=[p for p in pairs if len(nxt[(nxt['u']==p[0])&(nxt['i']==p[1])])>0 or len(nxt[(nxt['u']==p[1])&(nxt['i']==p[0])])>0]
            neg=[p for p in pairs if p not in pos]
            if not pos or not neg: continue
            import torch
            m=min(len(pos),len(neg))
            sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
            prs=pos+neg_s; labs=[1.0]*len(pos)+[0.0]*len(neg_s)
            prd = model.predict_next_edges(cur, torch.tensor(prs, device=device)).detach().cpu().numpy().tolist()
            all_pred+=prd; all_lab+=labs
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(all_lab, all_pred) if all_pred else 0.0
        results[rho]=ap
    return results

# ------------------------------ Triadic runner ------------------------------

def create_graph_sequence(g_df, timestamps):
    max_node = max(g_df['u'].max(), g_df['i'].max()) + 1
    seq=[]
    for ts in timestamps:
        edges = g_df[g_df['ts']<=ts]
        A=torch.zeros(max_node,max_node)
        for _,r in edges.iterrows():
            u,v=int(r['u']), int(r['i']); A[u,v]=A[v,u]=1.0
        seq.append(A)
    return seq

def load_triadic_data(data_name):
    import pandas as pd, json
    g_df = pd.read_csv(f'./processed/{data_name}/ml_{data_name}.csv')
    with open(f'./processed/{data_name}/ml_{data_name}_gt_fixed.json','r') as f:
        gt=json.load(f)
    gt_by_ts={}
    for ts_str, edges in gt.items():
        ts=float(ts_str); gt_by_ts[ts] = set( (min(u,v), max(u,v)) for (u,v) in edges )
    return g_df, gt_by_ts

def evaluate_graphmamba_sequence(model, graph_sequence, ground_truth_by_ts, timestamps, device, logger, eval_timestamps=None, recency_priors=None, edge_feature_seq=None):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        seq = model.forward_sequence(graph_sequence, recency_priors=recency_priors, edge_feature_seq=edge_feature_seq)
        for i in range(len(timestamps)-1):
            next_ts = timestamps[i+1]
            if eval_timestamps is not None and next_ts not in eval_timestamps: continue
            cur = seq[i]
            true_edges = ground_truth_by_ts.get(next_ts, set())
            if len(true_edges)==0: continue
            N=cur.shape[0]
            pairs=[(u,v) for u in range(N) for v in range(u+1,N)]
            pos=[p for p in pairs if p in true_edges]; neg=[p for p in pairs if p not in true_edges]
            if not pos: continue
            import torch
            m=min(len(pos),len(neg)); 
            if m==0: continue
            sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
            prs=pos+neg_s; labs=[1.0]*len(pos)+[0.0]*len(neg_s)
            prd=model.predict_next_edges(cur, torch.tensor(prs, device=device)).cpu().numpy().tolist()
            all_predictions+=prd; all_labels+=labs
    if not all_predictions: return {"accuracy":0.0,"auc":0.5,"ap":0.0}
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
    preds=np.array(all_predictions); labs=np.array(all_labels)
    return {"accuracy": float((preds>0.5).mean()), "auc": float(roc_auc_score(labs,preds)), "ap": float(average_precision_score(labs,preds)), "num_samples": int(len(preds))}

def train_graphmamba(data_name='triadic_perfect_long_dense', epochs=100, lr=0.001, hidden_dim=64, pos_dim=256, mamba_state_dim=16,
                     lambda_sparse: float = 0.0, lambda_tv: float = 1e-3, gate_temperature: float = 1.0,
                     beta_ib: float = 1e-3, r0: float = 0.05, alpha_prior: float = 0.7, gumbel_tau: float = 0.5, stochastic_gates: bool = True,
                     save_dir: str = '/mnt/data/experiments/triadic'):
    import logging, pandas as pd
    logging.basicConfig(level=logging.INFO); logger=logging.getLogger(__name__)
    logger.info(f"Loading {data_name}..."); g_df, gt = load_triadic_data(data_name)
    timestamps = sorted(g_df['ts'].unique())
    logger.info(f"{len(timestamps)} timestamps, {len(g_df)} edges")
    graph_sequence, recency_priors, edge_feature_seq = build_sequence_with_recency_and_features(g_df, timestamps, tau_prior=1.0, include_delta_feature=True)
    max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1

    tr=int(len(timestamps)*0.7); va=int(len(timestamps)*0.15); te=len(timestamps)-tr-va
    train_ts=timestamps[:tr]; val_ts=timestamps[tr:tr+va]; test_ts=timestamps[tr+va:]

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); logger.info(f"Device: {device}")
    model = GraphMamba(max_nodes=max_nodes, pos_dim=pos_dim, hidden_dim=hidden_dim, gnn_layers=2, mamba_state_dim=mamba_state_dim,
                       dropout=0.1, use_edge_gates=True, gate_temperature=gate_temperature, edge_feat_dim=1).to(device)
    graph_sequence=[A.to(device) for A in graph_sequence]; recency_priors=[R.to(device) for R in recency_priors]; edge_feature_seq=[E.to(device) for E in edge_feature_seq]
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5); bce=nn.BCELoss()

    best_val_ap=0.0; best=None

    for epoch in range(epochs):
        model.train(); total=0.0; steps=0
        for i in range(len(train_ts)-1):
            next_ts=train_ts[i+1]
            seq = graph_sequence[:i+2]
            rp  = recency_priors[:i+2]
            ef  = edge_feature_seq[:i+2]
            seq_emb, gates_list, prob_list = model.forward_sequence(seq, return_gates=True, recency_priors=rp, edge_feature_seq=ef,
                                                                    stochastic=stochastic_gates, gumbel_tau=gumbel_tau)
            cur = seq_emb[i]; probs_t = prob_list[i]; probs_prev = prob_list[i-1] if i>0 else None

            # build pairs against gt at next_ts
            true = gt.get(next_ts, set())
            if not true: continue
            N=cur.shape[0]
            pairs=[(u,v) for u in range(N) for v in range(u+1,N)]
            pos=[p for p in pairs if p in true]; neg=[p for p in pairs if p not in true]
            if not pos: continue
            m=min(len(pos), len(neg)); 
            if m==0: continue
            sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
            prs=pos+neg_s; labs=torch.tensor([1.0]*len(pos)+[0.0]*len(neg_s), device=device)
            prd = model.predict_next_edges(cur, torch.tensor(prs, device=device))
            loss_pred = bce(prd, labs)

            # TGIB-style KL with recency-informed prior
            r_t = (alpha_prior * recency_priors[i] + (1 - alpha_prior) * r0)
            loss_ib = GraphMamba.bernoulli_kl_loss(probs_t, r_t)
            loss_tv = GraphMamba.temporal_tv_loss(probs_t, probs_prev) if (i>0 and lambda_tv>0) else (probs_t.sum()*0)
            loss_sparse = GraphMamba.sparsity_loss(probs_t) if lambda_sparse>0 else (probs_t.sum()*0)

            loss = loss_pred + beta_ib*loss_ib + lambda_tv*loss_tv + lambda_sparse*loss_sparse
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.item()); steps += 1
        avg = total/max(steps,1)

        # validation every epoch
        val_seq = graph_sequence[:tr+va+1]; val_rp=recency_priors[:tr+va+1]; val_ef=edge_feature_seq[:tr+va+1]
        val_metrics = evaluate_graphmamba_sequence(model, val_seq, gt, timestamps[:tr+va], device, None, eval_timestamps=set(val_ts),
                                                   recency_priors=val_rp, edge_feature_seq=val_ef)
        print(f"Epoch {epoch:03d} loss={avg:.4f} | Val AP={val_metrics['ap']:.4f} AUC={val_metrics['auc']:.4f}")
        if val_metrics['ap']>best_val_ap:
            best_val_ap=val_metrics['ap']; best=val_metrics.copy()
            test_metrics = evaluate_graphmamba_sequence(model, graph_sequence, gt, timestamps, device, None, eval_timestamps=set(test_ts),
                                                        recency_priors=recency_priors, edge_feature_seq=edge_feature_seq)
            best.update({'test_accuracy':test_metrics['accuracy'], 'test_auc':test_metrics['auc'], 'test_ap':test_metrics['ap']})

    # Save artifacts
    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint(model, os.path.join(save_dir, f'{data_name}_best.pt'))
    # Dump predictions (val/test) and gates
    # Val
    model.eval()
    with torch.no_grad():
        seq_emb = model.forward_sequence(graph_sequence[:tr+va+1], recency_priors=recency_priors[:tr+va+1], edge_feature_seq=edge_feature_seq[:tr+va+1])
        _, gates_full, _ = model.forward_sequence(graph_sequence, return_gates=True, recency_priors=recency_priors, edge_feature_seq=edge_feature_seq)
    all_pairs, all_preds, all_labels, all_times = [], [], [], []
    for i in range(len(val_ts)-1):
        next_ts=val_ts[i+1]; cur=seq_emb[i]; N=cur.shape[0]
        true=gt.get(next_ts, set()); 
        if not true: continue
        pairs=[(u,v) for u in range(N) for v in range(u+1,N)]; pos=[p for p in pairs if p in true]; neg=[p for p in pairs if p not in true]
        if not pos or not neg: continue
        m=min(len(pos),len(neg)); import torch
        sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
        prs=pos+neg_s; labs=[1.0]*len(pos)+[0.0]*len(neg_s)
        prd = model.predict_next_edges(cur, torch.tensor(prs, device=cur.device)).cpu().numpy().tolist()
        all_pairs+=prs; all_preds+=prd; all_labels+=labs; all_times+=[next_ts]*len(prs)
    dump_predictions_csv(all_pairs, all_preds, all_labels, all_times, os.path.join(save_dir, f'{data_name}_val_predictions.csv'))
    # Test
    all_pairs, all_preds, all_labels, all_times = [], [], [], []
    with torch.no_grad(): seq_all = model.forward_sequence(graph_sequence, recency_priors=recency_priors, edge_feature_seq=edge_feature_seq)
    for i in range(len(timestamps)-1):
        next_ts=timestamps[i+1]; cur=seq_all[i]; N=cur.shape[0]; true=gt.get(next_ts,set()); 
        if not true: continue
        pairs=[(u,v) for u in range(N) for v in range(u+1,N)]; pos=[p for p in pairs if p in true]; neg=[p for p in pairs if p not in true]
        if not pos or not neg: continue
        m=min(len(pos),len(neg)); import torch
        sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
        prs=pos+neg_s; labs=[1.0]*len(pos)+[0.0]*len(neg_s)
        prd = model.predict_next_edges(cur, torch.tensor(prs, device=cur.device)).cpu().numpy().tolist()
        all_pairs+=prs; all_preds+=prd; all_labels+=labs; all_times+=[next_ts]*len(prs)
    dump_predictions_csv(all_pairs, all_preds, all_labels, all_times, os.path.join(save_dir, f'{data_name}_test_predictions.csv'))
    dump_gates_csv(gates_full, timestamps, os.path.join(save_dir, f'{data_name}_gates_top100.csv'), topk=100)

    # Explanation-only AP vs sparsity
    exp_only = explanation_only_eval(model, graph_sequence, recency_priors, edge_feature_seq, g_df, timestamps, device)
    with open(os.path.join(save_dir, f'{data_name}_exp_only_ap.json'),'w') as f: json.dump(exp_only, f, indent=2)
    return model, best

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Self-Explaining GraphMamba (Triadic) with TGIB-style IB and artifact saving")
    parser.add_argument('--data', type=str, default='triadic_perfect_long_dense', help='Dataset name under ./processed/<data>')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--pos_dim', type=int, default=256, help='Positional embedding dimension')
    parser.add_argument('--mamba_state_dim', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--gate_temperature', type=float, default=1.0, help='Gate temperature (higher = smoother gates)')
    parser.add_argument('--lambda_sparse', type=float, default=0.0, help='Optional L1 sparsity on gate probs (IB usually suffices)')
    parser.add_argument('--lambda_tv', type=float, default=1e-3, help='Temporal smoothness on gate probs')
    parser.add_argument('--beta_ib', type=float, default=1e-3, help='Weight of TGIB-style KL loss')
    parser.add_argument('--r0', type=float, default=0.05, help='Global sparsity prior for KL')
    parser.add_argument('--alpha_prior', type=float, default=0.7, help='Mixing weight between recency prior and r0')
    parser.add_argument('--gumbel_tau', type=float, default=0.5, help='Gumbel-Sigmoid temperature for stochastic gates')
    parser.add_argument('--no_stochastic_gates', action='store_true', help='Disable stochastic (ST-Gumbel) gates for training')
    parser.add_argument('--save_dir', type=str, default='/mnt/data/experiments/triadic', help='Directory to save artifacts')
    args = parser.parse_args()

    train_graphmamba(
        data_name=args.data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        pos_dim=args.pos_dim,
        mamba_state_dim=args.mamba_state_dim,
        lambda_sparse=args.lambda_sparse,
        lambda_tv=args.lambda_tv,
        gate_temperature=args.gate_temperature,
        beta_ib=args.beta_ib,
        r0=args.r0,
        alpha_prior=args.alpha_prior,
        gumbel_tau=args.gumbel_tau,
        stochastic_gates=(not args.no_stochastic_gates),
        save_dir=args.save_dir,
    )