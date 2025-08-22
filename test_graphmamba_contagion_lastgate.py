import os, json, math, csv
from typing import List, Dict, Tuple, Optional
import torch, torch.nn as nn
import pandas as pd
import numpy as np
from graphmamba_last_gate import GraphMambaLastGate, save_checkpoint

# ---------------------- Data utils (contagion format) ----------------------

def load_contagion_data(data_name: str):
    path=f'./processed/{data_name}/ml_{data_name}.csv'
    if not os.path.exists(path): raise FileNotFoundError(path)
    g_df = pd.read_csv(path)
    return g_df

def build_sequence_with_features(g_df: pd.DataFrame, timestamps: List[float], tau_prior: float=1.0,
                                 include_delta_feature: bool=True):
    max_node = max(g_df['u'].max(), g_df['i'].max()) + 1
    seq, ef_seq = [], []
    last_time = {}
    for ts in timestamps:
        A = torch.zeros(max_node, max_node)
        ef = torch.zeros(max_node, max_node, 1) if include_delta_feature else None
        cur = g_df[g_df['ts'] <= ts]
        for _,r in cur.iterrows():
            u,v,t = int(r['u']), int(r['i']), float(r['ts'])
            A[u,v]=1.0; A[v,u]=1.0
            key=(min(u,v),max(u,v))
            if key not in last_time or t>last_time[key]: last_time[key]=t
        if ef is not None:
            for (u,v),t_last in last_time.items():
                if A[u,v]>0:
                    delta=max(ts-t_last,0.0)
                    ef[u,v,0]=ef[v,u,0]=delta
        seq.append(A); 
        if ef is not None: ef_seq.append(ef)
    return seq, ef_seq

# ------------------------------ Evaluation ---------------------------------

def evaluate_prediction(model, graph_sequence, ef_seq, g_df, timestamps, device):
    model.eval()
    with torch.no_grad():
        Hs, Gs, _ = model.forward_sequence([A.to(device) for A in graph_sequence],
                                           [E.to(device) for E in ef_seq] if ef_seq else None)
    all_p, all_l = [], []
    for i in range(len(timestamps)-1):
        next_ts=timestamps[i+1]
        nxt=g_df[g_df['ts']==next_ts]
        if len(nxt)==0: continue
        H_t = Hs[i]; G_t = Gs[i]
        N=H_t.shape[0]
        pairs=[(u,v) for u in range(N) for v in range(u+1,N)]
        pos,neg=[],[]
        for (u,v) in pairs:
            ex=len(nxt[(nxt['u']==u)&(nxt['i']==v)])>0 or len(nxt[(nxt['u']==v)&(nxt['i']==u)])>0
            (pos if ex else neg).append((u,v))
        if not pos or not neg: continue
        m=min(len(pos),len(neg))
        if m==0: continue
        sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
        prs=pos+neg_s; labs=[1.0]*len(pos)+[0.0]*len(neg_s)
        prd=model.predict_next_edges(H_t, G_t, torch.tensor(prs, device=device)).detach().cpu().numpy().tolist()
        all_p+=prd; all_l+=labs
    if not all_p: return {"accuracy":0.0,"auc":0.5,"ap":0.0}
    from sklearn.metrics import roc_auc_score, average_precision_score
    preds=np.array(all_p); labs=np.array(all_l)
    return {"accuracy": float(((preds>0.5)==labs).mean()), "auc": float(roc_auc_score(labs,preds)), "ap": float(average_precision_score(labs,preds)), "num": int(len(preds))}

# ------------------------------ Explanations --------------------------------

def explain_events_gradient_x_gate(model, Hs, Gs, g_df, timestamps, device, topk_attr=15, topk_events=50,
                                   save_dir='/mnt/data/experiments/explain', plot_subgraphs: bool = False):
    os.makedirs(save_dir, exist_ok=True)
    rows=[]
    # helper to plot a per-event subgraph with top-k attributed edges
    def plot_event_subgraph(u:int, v:int, next_ts:float, top_edges:List[Tuple[int,int,float,float]], out_dir:str):
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except Exception:
            return None
        os.makedirs(out_dir, exist_ok=True)
        # build subgraph from attributed edges
        Gnx = nx.Graph()
        for i,j,attr_val,_g in top_edges:
            Gnx.add_edge(int(i), int(j), weight=float(attr_val))
        # ensure target nodes present
        Gnx.add_node(int(u)); Gnx.add_node(int(v))
        if Gnx.number_of_nodes()==0:
            return None
        # layout and styling
        pos = nx.spring_layout(Gnx, seed=42, k=None)
        edge_attrs = [Gnx.edges[e]['weight'] for e in Gnx.edges]
        if edge_attrs:
            vmax = max(edge_attrs)
        else:
            vmax = 1.0
        widths = [1.0 + 4.0*(w/(vmax+1e-8)) for w in edge_attrs]
        plt.figure(figsize=(6,5))
        # draw nodes
        node_colors = []
        for n in Gnx.nodes:
            if n==u or n==v:
                node_colors.append('#d62728')
            else:
                node_colors.append('#1f77b4')
        nx.draw_networkx_nodes(Gnx, pos, node_color=node_colors, node_size=220, alpha=0.9)
        # draw edges colored by attribution
        if edge_attrs:
            edges = list(Gnx.edges())
            lc = nx.draw_networkx_edges(Gnx, pos, edgelist=edges, width=widths, edge_color=edge_attrs, edge_cmap=plt.cm.viridis)
            plt.colorbar(lc, shrink=0.7, label='Grad×Gate')
        else:
            nx.draw_networkx_edges(Gnx, pos, width=1.5, alpha=0.6)
        # labels and title
        nx.draw_networkx_labels(Gnx, pos, font_size=8)
        plt.title(f"Top-{len(top_edges)} attributed edges for event ({u},{v}) at t={next_ts}")
        plt.axis('off')
        out_path = os.path.join(out_dir, f"event_t{str(next_ts).replace('.', '_')}_u{u}_v{v}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path
    for i in range(len(timestamps)-1):
        next_ts=timestamps[i+1]
        nxt=g_df[g_df['ts']==next_ts]
        if len(nxt)==0: continue
        H_t = Hs[i]; G_t = Gs[i]
        # build all positive events (u,v)
        pos = [(int(r['u']), int(r['i'])) for _,r in nxt.iterrows()]
        if not pos: continue
        # Limit number of events per timestamp (optional)
        if len(pos) > topk_events:
            pos = pos[:topk_events]
        for (u,v) in pos:
            # compute yhat(u,v)
            pair = torch.tensor([[u,v]], device=device)
            # make gates a leaf with gradients to attribute w.r.t gates
            G_t_var = G_t.detach().clone().requires_grad_(True)
            yhat = model.predict_next_edges(H_t, G_t_var, pair)
            # gradient wrt G_t probs
            yhat_scalar = yhat.squeeze()
            attr = GraphMambaLastGate.gradient_x_gate(yhat_scalar, G_t_var, retain_graph=True)  # [N,N]
            # take upper triangle top-k
            N = G_t.shape[0]
            tri = torch.triu(torch.ones(N,N, device=device), diagonal=1)
            vals = (attr*tri).flatten()
            k = min(topk_attr, int(N*(N-1)//2))
            idx = torch.topk(vals, k).indices
            # materialize top edges (i,j, attr, gate)
            top_edges = []
            for idv in idx.tolist():
                i_idx, j_idx = int(idv//N), int(idv%N)
                rows.append({
                    "time": float(next_ts),
                    "target_u": int(u),
                    "target_v": int(v),
                    "edge_i": i_idx,
                    "edge_j": j_idx,
                    "attr": float(attr[i_idx,j_idx].detach().cpu()),
                    "gate": float(G_t[i_idx,j_idx].detach().cpu())
                })
                top_edges.append((i_idx, j_idx, float(attr[i_idx,j_idx].detach().cpu()), float(G_t[i_idx,j_idx].detach().cpu())))
            if plot_subgraphs and top_edges:
                plot_event_subgraph(int(u), int(v), float(next_ts), top_edges, os.path.join(save_dir, 'subgraphs'))
    # save CSV
    out_csv = os.path.join(save_dir, "gradient_x_gate_explanations.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["time","target_u","target_v","edge_i","edge_j","attr","gate"])
        w.writeheader()
        for r in rows: w.writerow(r)
    return out_csv

def visualize_gate_heatmap(Gs, timestamps, out_png='/mnt/data/experiments/gate_heatmap_t0.png'):
    # Just plot the first timestamp's gate matrix as a heatmap for quick inspection.
    import matplotlib.pyplot as plt
    if not Gs: return None
    G0 = Gs[0].detach().cpu().numpy()
    plt.figure(figsize=(6,5))
    plt.imshow(G0, aspect='auto')
    plt.colorbar()
    plt.title(f"Late gate matrix at t={timestamps[0]}")
    plt.tight_layout()
    plt.savefig(out_png)
    return out_png

# ------------------------------ Training loop -------------------------------

def train_contagion_lastgate(data_name='synthetic_icm_ba', epochs=50, lr=1e-3, hidden_dim=64, pos_dim=128, mamba_state_dim=16, gpu_id=0,
                             lambda_sparse: float = 3e-4, lambda_tv: float = 1e-3, gate_temperature: float = 0.9,
                             save_dir: str = '/mnt/data/experiments/contagion_lastgate', resume_from: Optional[str] = None):
    # ---- experiment header/meta ----
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    print(f"Loading contagion dataset: {data_name}")
    g_df = load_contagion_data(data_name)
    timestamps = sorted(g_df['ts'].unique())
    print(f"Dataset: {len(g_df)} edges, {len(timestamps)} timestamps")
    graph_sequence, ef_seq = build_sequence_with_features(g_df, timestamps, tau_prior=1.0, include_delta_feature=True)
    max_nodes = max(g_df['u'].max(), g_df['i'].max()) + 1
    print(f"Graph sequence built: {len(graph_sequence)} graphs, max_nodes={max_nodes}, edge_features={'on' if ef_seq else 'off'}")

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        try:
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        except Exception:
            print(f"Using GPU {gpu_id}")
    else:
        print("Using CPU")
    model = GraphMambaLastGate(max_nodes=max_nodes, pos_dim=pos_dim, hidden_dim=hidden_dim, gnn_layers=2, mamba_state_dim=mamba_state_dim,
                               dropout=0.1, gate_temperature=gate_temperature, edge_feat_dim=(1 if ef_seq else 0)).to(device)
    print(f"Model params: pos_dim={pos_dim}, hidden_dim={hidden_dim}, mamba_state_dim={mamba_state_dim}, gnn_layers=2, dropout=0.1")
    print(f"Interpretation params: lambda_sparse={lambda_sparse}, lambda_tv={lambda_tv}, gate_temperature={gate_temperature}")

    # Optional: resume from checkpoint weights
    if resume_from is not None and os.path.exists(resume_from):
        state = torch.load(resume_from, map_location=device)
        if isinstance(state, dict) and all(k.startswith(('input','layers','mamba_encoder','late_head','edge_predictor','drop')) or True for k in state.keys()):
            model.load_state_dict(state)
            print(f"Loaded model weights from: {resume_from}")
        else:
            print(f"Warning: resume_from does not look like a state_dict: {resume_from}")

    graph_sequence=[A.to(device) for A in graph_sequence]
    ef_seq=[E.to(device) for E in ef_seq] if ef_seq else None

    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    bce=nn.BCELoss()
    print(f"Training params: epochs={epochs}, lr={lr}, optimizer=AdamW, weight_decay=1e-5")

    tr=int(len(timestamps)*0.7); va=int(len(timestamps)*0.15); train_ts=timestamps[:tr]; val_ts=timestamps[tr:tr+va]
    print(f"Split: train={len(train_ts)} ts, val={len(val_ts)} ts, test={len(timestamps)-len(train_ts)-len(val_ts)} ts")
    print("Sampling: train=2:1 (neg:pos; downsample negatives), eval=1:1 balanced")

    best_val_ap=0.0
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f'{data_name}_lastgate_best.pt')

    for epoch in range(epochs):
        model.train(); running=0.0; steps=0
        running_pred=0.0; running_sparse=0.0; running_tv=0.0
        for i in range(len(train_ts)-1):
            # forward states & late gates up to i
            Hs, Gs, _ = model.forward_sequence(graph_sequence[:i+1], ef_seq[:i+1] if ef_seq else None)
            H_t = Hs[-1]; G_t = Gs[-1]  # states & gates at time i
            next_ts=train_ts[i+1]
            nxt=g_df[g_df['ts']==next_ts]
            if len(nxt)==0: continue
            N=H_t.shape[0]; pairs=[(u,v) for u in range(N) for v in range(u+1,N)]
            pos,neg=[],[]
            for (u,v) in pairs:
                ex=len(nxt[(nxt['u']==u)&(nxt['i']==v)])>0 or len(nxt[(nxt['u']==v)&(nxt['i']==u)])>0
                (pos if ex else neg).append((u,v))
            if not pos or not neg: continue
            # 2:1 (neg:pos) sampling for training: keep all positives, downsample negatives
            pos_s = pos
            if len(pos) == 0 or len(neg) == 0:
                continue
            num_neg = min(len(neg), 2 * len(pos))
            sn = torch.randperm(len(neg))[:num_neg]
            neg_s = [neg[j] for j in sn]
            if not pos_s or not neg_s: continue
            prs=pos_s+neg_s; labs=torch.tensor([1.0]*len(pos_s)+[0.0]*len(neg_s), device=device)
            prd=model.predict_next_edges(H_t, G_t, torch.tensor(prs, device=device), adj_matrix_t=graph_sequence[i])
            loss_pred=bce(prd, labs)

            # reg on late gates
            # use adjacency at time i as mask so we mainly regularize existing edges
            A_t = graph_sequence[i]
            loss_sparse = GraphMambaLastGate.sparsity_loss(G_t, mask=A_t)
            if i>0:
                # need previous gates at time i-1
                with torch.no_grad():
                    _, Gs_prev, _ = model.forward_sequence(graph_sequence[:i], ef_seq[:i] if ef_seq else None)
                G_prev = Gs_prev[-1]
                mask_tv = (A_t + graph_sequence[i-1]).clamp(max=1.0)
                loss_tv = GraphMambaLastGate.temporal_tv_loss(G_t, G_prev, mask=mask_tv)
            else:
                loss_tv = torch.tensor(0.0, device=device)

            loss = loss_pred + lambda_sparse*loss_sparse + lambda_tv*loss_tv
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            running += float(loss.item()); steps += 1
            running_pred += float(loss_pred.item()); running_sparse += float(loss_sparse.item()); running_tv += float(loss_tv.item())

        # validation every 10 epochs (and last)
        if (epoch % 10 == 0) or (epoch == epochs - 1):
            model.eval()
            with torch.no_grad():
                Hs_all, Gs_all, _ = model.forward_sequence(graph_sequence[:tr+va], ef_seq[:tr+va] if ef_seq else None)
            from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
            all_p, all_l = [], []
            for i in range(len(val_ts)-1):
                next_ts=val_ts[i+1]
                nxt=g_df[g_df['ts']==next_ts]
                if len(nxt)==0: continue
                # use offset into combined train+val prefix
                offset = tr + i
                H_t = Hs_all[offset]; G_t = Gs_all[offset]
                N=H_t.shape[0]; pairs=[(u,v) for u in range(N) for v in range(u+1,N)]
                pos,neg=[],[]
                for (u,v) in pairs:
                    ex=len(nxt[(nxt['u']==u)&(nxt['i']==v)])>0 or len(nxt[(nxt['u']==v)&(nxt['i']==u)])>0
                    (pos if ex else neg).append((u,v))
                if not pos or not neg: continue
                m=min(len(pos),len(neg))
                if m==0: continue
                sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
                prs=pos+neg_s; labs=[1.0]*len(pos)+[0.0]*len(neg_s)
                prd = model.predict_next_edges(H_t, G_t, torch.tensor(prs, device=device), adj_matrix_t=graph_sequence[offset]).detach().cpu().numpy().tolist()
                all_p+=prd; all_l+=labs
            if all_p:
                preds_np = np.array(all_p); labs_np = np.array(all_l)
                ap = float(average_precision_score(labs_np, preds_np))
                auc = float(roc_auc_score(labs_np, preds_np)) if (len(np.unique(labs_np))>1) else 0.5
                acc = float(((preds_np>0.5)==labs_np).mean())
            else:
                ap = 0.0; auc = 0.5; acc = 0.0
            if steps>0:
                print(f"Epoch {epoch:03d} loss={running/steps:.4f} (pred={running_pred/steps:.4f}, sparse={running_sparse/steps:.4f}, tv={running_tv/steps:.4f}) | Val Acc={acc:.4f} AUC={auc:.4f} AP={ap:.4f} | gate stats: mean={float(torch.stack(Gs_all).mean()):.4f}, std={float(torch.stack(Gs_all).std()):.4f}")
            else:
                print(f"Epoch {epoch:03d} | Val Acc={acc:.4f} AUC={auc:.4f} AP={ap:.4f}")
            if ap > best_val_ap:
                best_val_ap = ap
                save_checkpoint(model, ckpt_path)
                print(f"New best Val AP={ap:.4f}. Checkpoint saved to: {ckpt_path}")
        else:
            if steps>0:
                print(f"Epoch {epoch:03d} loss={running/steps:.4f} (pred={running_pred/steps:.4f}, sparse={running_sparse/steps:.4f}, tv={running_tv/steps:.4f})")
            else:
                print(f"Epoch {epoch:03d}")

    # Save artifacts
    # Ensure at least one checkpoint is saved
    if not os.path.exists(ckpt_path):
        save_checkpoint(model, ckpt_path)
        print(f"Checkpoint saved to: {ckpt_path}")

    # Run full forward for explanation & export
    model.eval()
    with torch.no_grad():
        Hs, Gs, _ = model.forward_sequence(graph_sequence, ef_seq if ef_seq else None)
    # simple heatmap of first gate matrix
    try:
        from matplotlib import pyplot as plt  # noqa: F401
        heatmap_png = visualize_gate_heatmap(Gs, timestamps, os.path.join(save_dir, "gate_heatmap_t0.png"))
    except Exception as e:
        heatmap_png = None

    # gradient x gate explanations for positives
    expl_csv = explain_events_gradient_x_gate(model, Hs, Gs, g_df, timestamps, device, save_dir=save_dir)

    # Final test evaluation
    test_ts = timestamps[tr+va:]
    with torch.no_grad():
        Hs_full, Gs_full, _ = model.forward_sequence(graph_sequence, ef_seq if ef_seq else None)
    from sklearn.metrics import roc_auc_score, average_precision_score
    all_p_t, all_l_t = [], []
    for i in range(len(test_ts)-1):
        next_ts = test_ts[i+1]
        nxt = g_df[g_df['ts']==next_ts]
        if len(nxt)==0: continue
        offset = tr + va + i
        H_t = Hs_full[offset]; G_t = Gs_full[offset]
        N=H_t.shape[0]; pairs=[(u,v) for u in range(N) for v in range(u+1,N)]
        pos,neg=[],[]
        for (u,v) in pairs:
            ex=len(nxt[(nxt['u']==u)&(nxt['i']==v)])>0 or len(nxt[(nxt['u']==v)&(nxt['i']==u)])>0
            (pos if ex else neg).append((u,v))
        if not pos or not neg: continue
        m=min(len(pos),len(neg))
        if m==0: continue
        sn=torch.randperm(len(neg))[:m]; neg_s=[neg[j] for j in sn]
        prs=pos+neg_s; labs=[1.0]*len(pos)+[0.0]*len(neg_s)
        prd = model.predict_next_edges(H_t, G_t, torch.tensor(prs, device=device), adj_matrix_t=graph_sequence[offset]).detach().cpu().numpy().tolist()
        all_p_t+=prd; all_l_t+=labs
    if all_p_t:
        preds_np = np.array(all_p_t); labs_np = np.array(all_l_t)
        test_ap = float(average_precision_score(labs_np, preds_np))
        test_auc = float(roc_auc_score(labs_np, preds_np)) if (len(np.unique(labs_np))>1) else 0.5
        test_acc = float(((preds_np>0.5)==labs_np).mean())
    else:
        test_ap = 0.0; test_auc = 0.5; test_acc = 0.0

    print(f"Final Test: Acc={test_acc:.4f} AUC={test_auc:.4f} AP={test_ap:.4f}")
    print(f"Artifacts: ckpt={ckpt_path}, heatmap={heatmap_png}, explanations={expl_csv}")

    return {"best_val_ap": best_val_ap, "test_acc": test_acc, "test_auc": test_auc, "test_ap": test_ap,
            "expl_csv": expl_csv, "heatmap": heatmap_png, "ckpt": ckpt_path}

# ----------------------------- CLI Entrypoint ------------------------------

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description='Train GraphMamba with LAST-LAYER gate head on contagion data, with explanations.')
    p.add_argument('--data', type=str, default='synthetic_icm_ba')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--pos_dim', type=int, default=128)
    p.add_argument('--mamba_state_dim', type=int, default=16)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--lambda_sparse', type=float, default=3e-4)
    p.add_argument('--lambda_tv', type=float, default=1e-3)
    p.add_argument('--gate_temperature', type=float, default=0.9)
    p.add_argument('--save_dir', type=str, default='/mnt/data/experiments/contagion_lastgate')
    p.add_argument('--resume_from', type=str, default=None)
    a=p.parse_args()
    out = train_contagion_lastgate(a.data, a.epochs, a.lr, a.hidden_dim, a.pos_dim, a.mamba_state_dim, a.gpu,
                                   a.lambda_sparse, a.lambda_tv, a.gate_temperature, a.save_dir, a.resume_from)
    print(json.dumps(out, indent=2))