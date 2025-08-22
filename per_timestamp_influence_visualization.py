#!/usr/bin/env python3
"""
Per-timestamp visualization of:
- Background graph G_{t-1} (gray edges)
- Random sample (<=10) of new edges at t (blue dashed)
- Top-k influential edges from G_{t-1} (red solid), with k = 4 * (#new edges at t)
Also computes and logs coverage rate of these k edges for new edges at t.
Outputs:
- PNG per timestamp under results_triadic_long_dense/per_timestamp_viz/
- CSV with stats per timestamp
- Markdown report summarizing results and embedding example images
"""

import os
import csv
import json
import random
from typing import List, Tuple, Dict, Set

import numpy as np
import pandas as pd
import torch

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graphmamba_explain import GraphMamba
from visualize_event_influence import per_event_influence_grad_x_gate

random.seed(42)

def load_edges_csv(data_name: str):
    path=f'./processed/{data_name}/ml_{data_name}.csv'
    if not os.path.exists(path): 
        raise FileNotFoundError(f"Edges CSV not found: {path}")
    return pd.read_csv(path)

def build_graph_sequence_on_device(g_df: pd.DataFrame, timestamps: List[float], device: torch.device):
    max_node = int(max(g_df['u'].max(), g_df['i'].max())) + 1
    seq = []
    for ts in timestamps:
        A = torch.zeros(max_node, max_node, device=device)
        cur = g_df[g_df['ts'] <= ts]
        for _,r in cur.iterrows():
            u,v = int(r['u']), int(r['i'])
            A[u,v]=1.0; A[v,u]=1.0
        seq.append(A)
    return seq, max_node

def edges_at_timestamp(g_df: pd.DataFrame, ts_val: float) -> List[Tuple[int,int]]:
    rows = g_df[g_df['ts'] == ts_val]
    out=[]
    for _,r in rows.iterrows():
        u,v = int(r['u']), int(r['i'])
        if u==v: 
            continue
        if u>v:
            u,v=v,u
        out.append((u,v))
    return out

def get_existing_edges(A: torch.Tensor) -> Set[Tuple[int,int]]:
    N=A.shape[0]
    out=set()
    for i in range(N):
        for j in range(i+1,N):
            if float(A[i,j])>0:
                out.add((i,j))
    return out

def rank_edges_by_influence(model: GraphMamba,
                            graph_sequence: List[torch.Tensor],
                            t_index: int,
                            target_u: int,
                            target_v: int) -> List[Tuple[Tuple[int,int], float]]:
    # Compute influence on a representative target (use first new edge if available)
    infl, _, _ = per_event_influence_grad_x_gate(model, graph_sequence, t_index, target_u, target_v, mask_to_edges=True)
    A_prev = graph_sequence[t_index]
    N = infl.shape[0]
    ranked=[]
    for i in range(N):
        for j in range(i+1,N):
            if float(A_prev[i,j])>0:
                ranked.append(((i,j), float(infl[i,j].detach().cpu())))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

def coverage_of_topk(topk_edges: Set[Tuple[int,int]], new_edges: List[Tuple[int,int]]) -> float:
    # Coverage definition: for new edge (u,v), exists w s.t. (u,w) and (v,w) both in topk
    top_adj: Dict[int, Set[int]] = {}
    for (u,v) in topk_edges:
        top_adj.setdefault(u,set()).add(v)
        top_adj.setdefault(v,set()).add(u)
    if not new_edges:
        return 0.0
    covered=0
    for (u,v) in new_edges:
        nu = top_adj.get(u, set())
        nv = top_adj.get(v, set())
        if nu and nv and (len(nu.intersection(nv))>0):
            covered+=1
    return covered/len(new_edges)

def compute_highlight_for_sampled(sampled_new_edges: List[Tuple[int,int]],
                                  topk_edges: List[Tuple[int,int]]):
    """For each sampled new edge (u,v) covered by top-k, pick one common w and
    return influential edges to highlight ((u,w),(v,w)) and the common nodes w.
    """
    topk_set = set(topk_edges)
    top_adj: Dict[int, Set[int]] = {}
    for (u,v) in topk_set:
        top_adj.setdefault(u,set()).add(v)
        top_adj.setdefault(v,set()).add(u)
    highlight_edges: Set[Tuple[int,int]] = set()
    highlight_nodes: Set[int] = set()
    covered_count = 0
    for (u,v) in sampled_new_edges:
        nu = top_adj.get(u, set())
        nv = top_adj.get(v, set())
        common = nu.intersection(nv)
        if common:
            w = next(iter(common))
            e1 = (min(u,w), max(u,w))
            e2 = (min(v,w), max(v,w))
            if e1 in topk_set:
                highlight_edges.add(e1)
            if e2 in topk_set:
                highlight_edges.add(e2)
            highlight_nodes.add(w)
            covered_count += 1
    return list(highlight_edges), list(highlight_nodes), covered_count

def draw_graph(A_prev: torch.Tensor,
               sampled_new_edges: List[Tuple[int,int]],
               topk_edges: List[Tuple[int,int]],
               out_png: str,
               title: str,
               highlight_edges: List[Tuple[int,int]] = None,
               highlight_nodes: List[int] = None):
    import matplotlib.pyplot as plt
    try:
        import networkx as nx
    except Exception:
        return None
    G = nx.Graph()
    N = A_prev.shape[0]
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i+1,N):
            if float(A_prev[i,j])>0:
                G.add_edge(i,j)
    pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)
    plt.figure(figsize=(18,14))
    # Background edges
    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), edge_color='black', width=2.5, alpha=0.35)
    # New edges (blue dashed)
    if sampled_new_edges:
        nx.draw_networkx_edges(G, pos, edgelist=sampled_new_edges, edge_color='blue', style='dashed', width=4.0, alpha=0.9)
    # Top-k influential (red solid)
    if topk_edges:
        nx.draw_networkx_edges(G, pos, edgelist=topk_edges, edge_color='red', width=5.0, alpha=1.0)
    # EXTREME highlight for covered sampled links: their supporting influential edges
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, edge_color='gold', width=8.0, alpha=1.0)
    # Nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.85)
    # Highlight common nodes w
    if highlight_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color='yellow', node_size=800, alpha=0.95, edgecolors='black', linewidths=2.0)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    return out_png

def main():
    data_name = 'triadic_perfect_long_dense'
    out_dir = './results_triadic_long_dense/per_timestamp_viz'
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    g_df = load_edges_csv(data_name)
    timestamps = sorted(g_df['ts'].unique().tolist())

    # Device and model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph_sequence, max_nodes = build_graph_sequence_on_device(g_df, timestamps, device)

    ckpt_path = './results_triadic_long_dense/triadic_perfect_long_dense_best_model.pth'
    model = GraphMamba(max_nodes=max_nodes,
                       pos_dim=128,
                       hidden_dim=128,
                       gnn_layers=2,
                       mamba_state_dim=16,
                       dropout=0.1,
                       use_edge_gates=True,
                       gate_temperature=1.0).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    state_dict = sd.get('model_state', sd) if isinstance(sd, dict) else sd
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    stats_rows = []

    for ti, ts in enumerate(timestamps):
        if ti==0:
            # No G_{t-1}
            stats_rows.append([ti, int(ts), 0, 0, 0, 0.0, 0, 0, 0])
            continue
        A_prev = graph_sequence[ti-1]
        prev_edges = get_existing_edges(A_prev)
        new_edges = edges_at_timestamp(g_df, ts)
        num_new = len(new_edges)
        # Sample up to 10 new edges
        sampled_new = new_edges if num_new<=10 else random.sample(new_edges, 10)

        # Rank by influence using the first new edge as target; if none, skip
        if num_new>0:
            target_u, target_v = new_edges[0]
            ranked = rank_edges_by_influence(model, graph_sequence, ti-1, target_u, target_v)
        else:
            ranked = []

        k = min(len(ranked), 4*max(1,num_new))  # if no new edges, still take some top edges (k=4)
        topk = [edge for (edge,score) in ranked[:k]] if ranked else []
        coverage = coverage_of_topk(set(topk), new_edges) if num_new>0 else 0.0

        # Compute highlights for sampled covered links
        highlight_edges, highlight_nodes, covered_sampled = compute_highlight_for_sampled(sampled_new, topk)

        # Draw figure
        png_path = os.path.join(out_dir, f't{ti:02d}_viz.png')
        title = f't={ti} (ts={ts}) | prev_edges={len(prev_edges)}, new={num_new}, k={k}, cov_all_new={coverage:.3f}, sampled_cov={covered_sampled}/{len(sampled_new)}'
        draw_graph(A_prev, sampled_new, topk, png_path, title, highlight_edges=highlight_edges, highlight_nodes=highlight_nodes)

        stats_rows.append([ti, float(ts), len(prev_edges), num_new, k, float(coverage), len(topk), covered_sampled, len(sampled_new)])

    # Save CSV
    csv_path = os.path.join(out_dir, 'per_timestamp_viz_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_index','timestamp','prev_edges','new_edges','k','coverage','topk_count','sampled_covered','sampled_count'])
        w.writerows(stats_rows)

    # Create markdown summary with few thumbnails
    md_path = os.path.join(out_dir, 'README.md')
    with open(md_path, 'w') as f:
        f.write('# Per-Timestamp Influence Visualization\n\n')
        f.write('- Background: G_{t-1} (gray), Sampled new edges (blue dashed), Top-k influential (red), Highlighted supporting edges (gold) and common nodes w (yellow).\n')
        f.write('- k = 4 × (# new edges at t). Coverage = fraction of new edges triad-covered by top-k.\n\n')
        f.write('## Stats Table\n\n')
        f.write('| t | prev_edges | new_edges | k | coverage | sampled_cov | image |\n')
        f.write('|---|------------|-----------|---|----------|-------------|-------|\n')
        for row in stats_rows:
            ti, ts, prev_e, new_e, k, cov, topk_count, sampled_cov, sampled_cnt = row
            img = f't{ti:02d}_viz.png' if ti>0 else ''
            f.write(f'| {ti} | {prev_e} | {new_e} | {k} | {cov:.3f} | {sampled_cov}/{sampled_cnt} | {img} |\n')

    print(json.dumps({'out_dir': out_dir, 'csv': csv_path, 'md': md_path}, indent=2))

if __name__ == '__main__':
    main()
