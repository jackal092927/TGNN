
#!/usr/bin/env python3
"""
Event-specific edge influence visualization (Grad × Gate) for your *existing* GraphMamba model.

- No changes to your model or trainer.
- Loads a trained checkpoint and your dataset.
- For a given time index t and target edge (u,v) that appears at t+1,
  computes a per-event influence matrix:
      influence[i,j] = G_t[i,j] * ReLU(d y_hat(u,v) / d G_t[i,j])
  where G_t is the gated adjacency from your *first spatial layer* at time t.

Artifacts saved to --out_dir:
  - event_influence_t{t}_u{u}_v{v}.csv
  - event_heatmap_t{t}_u{u}_v{v}.png
  - event_subgraph_t{t}_u{u}_v{v}.png  (requires networkx; optional)

Usage example (contagion):
  python visualize_event_influence.py \
    --data synthetic_icm_ba \
    --ckpt /mnt/data/experiments/contagion_sexplain/synthetic_icm_ba_best.pt \
    --t_index 12 --u 3 --v 17 \
    --hidden_dim 64 --pos_dim 128 --mamba_state_dim 16 --gnn_layers 2 \
    --out_dir /mnt/data/experiments/contagion_sexplain/visuals

Usage example (triadic):
  python visualize_event_influence.py \
    --data triadic_perfect_long_dense \
    --ckpt /mnt/data/experiments/triadic_sexplain/triadic_perfect_long_dense_best.pt \
    --t_index 25 --u 18 --v 42 \
    --hidden_dim 64 --pos_dim 256 --mamba_state_dim 16 --gnn_layers 2 \
    --out_dir /mnt/data/experiments/triadic_sexplain/visuals
"""
import os, json, math, csv, argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Make sure we can import your existing model from the current folder
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graphmamba_explain import GraphMamba  # <-- your current model

# ---------------------- Data utils (contagion/triadic) ----------------------

def load_edges_csv(data_name: str):
    path=f'./processed/{data_name}/ml_{data_name}.csv'
    if not os.path.exists(path): 
        raise FileNotFoundError(f"Edges CSV not found: {path}")
    return pd.read_csv(path)

def maybe_load_triadic_gt(data_name: str):
    path=f'./processed/{data_name}/ml_{data_name}_gt_fixed.json'
    if os.path.exists(path):
        with open(path,'r') as f:
            gt=json.load(f)
        gt_by_ts={float(ts): set((min(u,v),max(u,v)) for (u,v) in edges) for ts,edges in gt.items()}
        return gt_by_ts
    return None

def build_graph_sequence(g_df: pd.DataFrame, timestamps: List[float]):
    """Adjacency per timestamp t using all edges with ts <= t (undirected)."""
    max_node = int(max(g_df['u'].max(), g_df['i'].max())) + 1
    seq = []
    for ts in timestamps:
        A = torch.zeros(max_node, max_node)
        cur = g_df[g_df['ts'] <= ts]
        for _,r in cur.iterrows():
            u,v = int(r['u']), int(r['i'])
            A[u,v]=1.0; A[v,u]=1.0
        seq.append(A)
    return seq, max_node

# ---------------------------- Explainer helpers -----------------------------

@torch.no_grad()
def _upper_tri_indices(N, device):
    return torch.triu_indices(N, N, offset=1, device=device)

def per_event_influence_grad_x_gate(model: GraphMamba,
                                    graph_sequence: List[torch.Tensor],
                                    t_index: int,
                                    u: int, v: int,
                                    mask_to_edges: bool = True):
    """
    Compute per-event influence at step t for target (u,v) at t+1:
        influence[i,j] = G_t[i,j] * ReLU(d y_hat(u,v) / d G_t[i,j])
    Returns: (influence[N,N], G_t[N,N], yhat_scalar)
    """
    model.eval()  # no dropout etc.

    # Forward *with graph* (no torch.no_grad here) so autograd sees gates -> yhat path
    temporal_embeddings, gates_list = model.forward_sequence(graph_sequence, return_gates=True)
    H_t: torch.Tensor = temporal_embeddings[t_index]  # [N,H]
    G_t: torch.Tensor = gates_list[t_index]           # [N,N] masked to A_t inside the model
    # We want gradients wrt G_t (intermediate). It already has requires_grad=True if upstream params require grad
    # But to be safe for autograd.grad, tell PyTorch to retain grad on this non-leaf tensor.
    G_t.retain_grad()

    pair = torch.tensor([[u, v]], device=H_t.device)
    yhat = model.predict_next_edges(H_t, pair).squeeze()  # scalar

    grad = torch.autograd.grad(yhat, G_t, retain_graph=False, create_graph=False, allow_unused=False)[0]
    influence = torch.relu(grad) * G_t.detach()

    if mask_to_edges:
        A_t = graph_sequence[t_index].to(influence.device)
        influence = influence * A_t

    return influence, G_t.detach(), float(yhat.detach().cpu())

def topk_event_influential_edges(influence: torch.Tensor, k: int = 20):
    N = influence.shape[0]; device = influence.device
    I, J = _upper_tri_indices(N, device)
    vals = influence[I, J]
    if vals.numel() == 0:
        return []
    k = min(k, vals.numel())
    idx = torch.topk(vals, k).indices
    out = [(int(I[i]), int(J[i]), float(vals[i].detach().cpu())) for i in idx]
    return out

def plot_event_influence_heatmap(infl: torch.Tensor, u: int, v: int, t_index: int, out_png: str):
    import matplotlib.pyplot as plt
    A = infl.detach().cpu().numpy()
    plt.figure(figsize=(6,5))
    plt.imshow(A, aspect="auto")
    plt.colorbar()
    plt.title(f"Per-event influence heatmap for ({u},{v}) at t={t_index}")
    plt.tight_layout()
    plt.savefig(out_png)
    return out_png

def plot_event_subgraph(A_t: torch.Tensor,
                        top_edges: List[Tuple[int,int,float]],
                        target: Tuple[int,int],
                        out_png: str):
    """Overlay top-k influential edges on the current graph; dashed target edge."""
    import matplotlib.pyplot as plt
    try:
        import networkx as nx
    except Exception as e:
        # networkx is optional; silently skip
        return None
    G = nx.Graph()
    N = A_t.shape[0]
    G.add_nodes_from(range(N))
    
    # Build the graph and store edge weights
    edge_weights = {}
    for i in range(N):
        for j in range(i+1, N):
            if float(A_t[i,j]) > 0.0:
                G.add_edge(i, j)
                edge_weights[(i,j)] = 0.0  # Default weight for regular edges
    
    # Update edge weights with influence scores
    for (i,j,score) in top_edges:
        if (i,j) in edge_weights:
            edge_weights[(i,j)] = score
        elif (j,i) in edge_weights:
            edge_weights[(j,i)] = score
    
    # Use spring layout with much larger spacing to prevent node overlap
    pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)  # k=3.0 increases repulsion, iterations=100 for better convergence
    plt.figure(figsize=(20,16))  # Much larger figure to accommodate spread-out nodes
    
    # Prepare edge lists and properties
    all_edges = list(G.edges())
    regular_edges = [(u,v) for (u,v) in all_edges if edge_weights.get((u,v), 0.0) == 0.0]
    influential_edges = [(u,v) for (u,v) in all_edges if edge_weights.get((u,v), 0.0) > 0.0]
    
    print(f"[DEBUG] Total edges: {len(all_edges)}")
    print(f"[DEBUG] Regular edges: {len(regular_edges)}")
    print(f"[DEBUG] Influential edges: {len(influential_edges)}")
    
    # Draw ALL edges with constant, visible width first (guaranteed visibility)
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color='black', width=3.0, alpha=0.4)
    print(f"[DEBUG] Drew ALL {len(all_edges)} edges in BLACK with width 3.0, alpha=0.4 (lighter background)")
    
    # Draw influential edges on top with constant RED width (guaranteed visibility)
    if influential_edges:
        nx.draw_networkx_edges(G, pos, edgelist=influential_edges, 
                              edge_color='red', width=6.0, alpha=1.0)
        print(f"[DEBUG] Drew {len(influential_edges)} influential edges in RED with width 6.0")
        
        # Print the influential edges for debugging
        for i, (u,v) in enumerate(influential_edges):
            print(f"[DEBUG] Influential Edge ({u},{v}): score={edge_weights[(u,v)]:.8f}")
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600, alpha=0.8)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw target edge in BLUE dashed line (if it doesn't exist in the graph)
    (u,v) = target
    if G.has_node(u) and G.has_node(v):
        if (u,v) not in G.edges() and (v,u) not in G.edges():
            nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], edge_color='blue', style='dashed', width=6.0, alpha=0.9)
            print(f"[DEBUG] Drew target edge ({u},{v}) in BLUE dashed")
        else:
            print(f"[DEBUG] Target edge ({u},{v}) already exists in graph")
    
    plt.title(f"Event Influence Subgraph: Target ({u},{v})\nGray=Regular, Red=Influential (scaled), Blue=Target", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    return out_png

# ------------------------------- Main routine -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Per-event edge influence visualization (Grad×Gate) for a trained GraphMamba model.")
    ap.add_argument("--data", type=str, required=True, help="Dataset name under ./processed/<data>/ml_<data>.csv")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.pt)")
    # event specification
    ap.add_argument("--t_index", type=int, help="Index into sorted unique timestamps (0..T-1)")
    ap.add_argument("--ts", type=float, help="Timestamp value (we will map it to t_index)")
    ap.add_argument("--u", type=int, required=True, help="Target edge endpoint u (at t+1)")
    ap.add_argument("--v", type=int, required=True, help="Target edge endpoint v (at t+1)")
    ap.add_argument("--topk", type=int, default=20, help="Top-k prior edges to list/visualize")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="/mnt/data/experiments/event_viz")
    # model config (must match training)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--pos_dim", type=int, default=256)
    ap.add_argument("--gnn_layers", type=int, default=2)
    ap.add_argument("--mamba_state_dim", type=int, default=16)
    ap.add_argument("--gate_temperature", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # data
    g_df = load_edges_csv(args.data)
    timestamps = sorted(g_df['ts'].unique().tolist())
    if args.t_index is None:
        if args.ts is None:
            raise ValueError("Please provide either --t_index or --ts (timestamp value).")
        try:
            t_index = timestamps.index(args.ts)
        except ValueError:
            # map to nearest timestamp
            diffs = [(abs(float(t)-float(args.ts)), i) for i,t in enumerate(timestamps)]
            t_index = min(diffs)[1]
            print(f"[info] Provided ts={args.ts} not found; using nearest index t_index={t_index} (ts={timestamps[t_index]})")
    else:
        t_index = int(args.t_index)
    if not (0 <= t_index < len(timestamps)):
        raise ValueError(f"t_index out of range: {t_index} (0..{len(timestamps)-1})")

    graph_sequence, max_nodes = build_graph_sequence(g_df, timestamps)
    graph_sequence = [A.to(device) for A in graph_sequence]

    # model
    model = GraphMamba(max_nodes=max_nodes,
                       pos_dim=args.pos_dim,
                       hidden_dim=args.hidden_dim,
                       gnn_layers=args.gnn_layers,
                       mamba_state_dim=args.mamba_state_dim,
                       dropout=0.1,
                       use_edge_gates=True,
                       gate_temperature=args.gate_temperature).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    # allow both plain state_dict and wrapper dicts
    state_dict = sd.get("model_state", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # compute influence
    os.makedirs(args.out_dir, exist_ok=True)
    infl, G_t, yhat = per_event_influence_grad_x_gate(model, graph_sequence, t_index, args.u, args.v, mask_to_edges=True)
    # save CSV of top-k
    top_edges = topk_event_influential_edges(infl, k=args.topk)
    csv_path = os.path.join(args.out_dir, f"event_influence_t{t_index}_u{args.u}_v{args.v}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_index","u","v","edge_i","edge_j","influence","gate_value","yhat_uv"])
        for (i,j,score) in top_edges:
            w.writerow([t_index, args.u, args.v, i, j, score, float(G_t[i,j].cpu()), yhat])

    # heatmap
    heatmap_png = os.path.join(args.out_dir, f"event_heatmap_t{t_index}_u{args.u}_v{args.v}.png")
    plot_event_influence_heatmap(infl, args.u, args.v, t_index, heatmap_png)

    # subgraph (optional)
    try:
        subgraph_png = os.path.join(args.out_dir, f"event_subgraph_t{t_index}_u{args.u}_v{args.v}.png")
        plot_event_subgraph(graph_sequence[t_index].cpu(), top_edges, (args.u,args.v), subgraph_png)
    except Exception:
        subgraph_png = None

    # also save a tiny JSON index
    with open(os.path.join(args.out_dir, f"event_summary_t{t_index}_u{args.u}_v{args.v}.json"), "w") as f:
        json.dump({
            "t_index": t_index,
            "timestamp": float(timestamps[t_index]),
            "u": args.u, "v": args.v,
            "yhat": yhat,
            "csv": csv_path,
            "heatmap": heatmap_png,
            "subgraph": subgraph_png,
        }, f, indent=2)

    print(json.dumps({
        "csv": csv_path,
        "heatmap": heatmap_png,
        "subgraph": subgraph_png,
        "yhat_uv": yhat,
        "num_top_edges": len(top_edges)
    }, indent=2))

if __name__ == "__main__":
    main()
