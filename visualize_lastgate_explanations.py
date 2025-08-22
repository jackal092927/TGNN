#!/usr/bin/env python3
"""
Visualize per-event (per-link-prediction) subgraphs for the LAST-GATE GraphMamba runner.

Input: gradient_x_gate_explanations.csv with rows:
  time, target_u, target_v, edge_i, edge_j, attr, gate

This script groups by event (time, target_u, target_v), takes top-K attributed edges,
and plots a subgraph with edge widths/colors proportional to Grad×Gate attribution.
Optionally overlays the base graph at that timestamp using processed ml_{data}.csv.
"""

import os
import csv
import argparse
from collections import defaultdict
from typing import Tuple, List

import numpy as np
import pandas as pd


def _nearest_timestamp(ts_values: np.ndarray, target: float) -> float:
    if ts_values.size == 0:
        return target
    idx = int(np.argmin(np.abs(ts_values - target)))
    return float(ts_values[idx])

def load_explanations(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    req = {"time", "target_u", "target_v", "edge_i", "edge_j", "attr", "gate"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {sorted(list(req - set(df.columns)))}")
    return df


def group_events(df: pd.DataFrame):
    grouped = defaultdict(list)
    for _, r in df.iterrows():
        key = (float(r["time"]), int(r["target_u"]), int(r["target_v"]))
        grouped[key].append((int(r["edge_i"]), int(r["edge_j"]), float(r["attr"]), float(r["gate"])) )
    return grouped


def plot_event(all_edges_attr: List[Tuple[int,int,float,float]], u:int, v:int, t:float, out_dir:str,
               overlay_graph: bool=False, data_name: str=None, overlay_mode: str = 'upto', topk: int = 5):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception:
        return None

    os.makedirs(out_dir, exist_ok=True)

    # Load full base graph at time t (or up to t) from original dataset
    base_edges = []
    G_full = None
    if overlay_graph and data_name is not None:
        print(f"DEBUG: Loading base graph for data={data_name}, t={t}")
        data_path = os.path.join("processed", data_name, f"ml_{data_name}.csv")
        if os.path.exists(data_path):
            print(f"DEBUG: Found data file: {data_path}")
            g_df = pd.read_csv(data_path)
            # enforce dtypes
            g_df['u'] = pd.to_numeric(g_df['u'], errors='coerce').astype('Int64')
            g_df['i'] = pd.to_numeric(g_df['i'], errors='coerce').astype('Int64')
            g_df['ts'] = pd.to_numeric(g_df['ts'], errors='coerce')
            print(f"DEBUG: Loaded {len(g_df)} rows, timestamps: {sorted(g_df['ts'].dropna().unique())[:5]}...")
            # resolve nearest timestamp to avoid float mismatch
            ts_unique = np.array(sorted(g_df['ts'].dropna().unique()))
            t_resolved = _nearest_timestamp(ts_unique, t)
            print(f"DEBUG: Target t={t}, resolved to {t_resolved}")
            cur = g_df[g_df["ts"] <= t_resolved] if overlay_mode != 'at' else g_df[g_df['ts'] == t_resolved]
            print(f"DEBUG: Found {len(cur)} edges up to t={t_resolved}")
            base_edges = [(int(r["u"]), int(r["i"])) for _, r in cur.iterrows() if pd.notna(r['u']) and pd.notna(r['i'])]
            print(f"DEBUG: Converted to {len(base_edges)} base edges")
            G_full = nx.Graph()
            G_full.add_edges_from(base_edges)
            G_full.add_node(int(u)); G_full.add_node(int(v))
        else:
            print(f"DEBUG: Data file not found: {data_path}")
    else:
        print(f"DEBUG: Overlay disabled: overlay_graph={overlay_graph}, data_name={data_name}")

    if G_full is None:
        # fallback: build a graph from attributed edges only (not ideal layout)
        G_full = nx.Graph()
        for i,j,attr_val,_g in all_edges_attr:
            G_full.add_edge(int(i), int(j))
        G_full.add_node(int(u)); G_full.add_node(int(v))

    if G_full.number_of_nodes() == 0:
        return None

    # Compute top-k attributed edges; prefer those existing in base graph
    if base_edges:
        base_set = set((min(a,b), max(a,b)) for (a,b) in base_edges)
        existing_attr = [(i,j,attr,g) for (i,j,attr,g) in all_edges_attr if (min(int(i),int(j)), max(int(i),int(j))) in base_set]
        existing_attr_sorted = sorted(existing_attr, key=lambda x: x[2], reverse=True)
        top_edges = existing_attr_sorted[:topk]
        if len(top_edges) == 0:
            print("DEBUG: No attributed edges intersect base; falling back to raw top-k")
            top_edges = sorted(all_edges_attr, key=lambda x: x[2], reverse=True)[:topk]
    else:
        top_edges = sorted(all_edges_attr, key=lambda x: x[2], reverse=True)[:topk]

    # Ensure top edges exist in graph before layout
    for (i,j,_,_) in top_edges:
        if not G_full.has_edge(int(i), int(j)):
            G_full.add_edge(int(i), int(j))

    # Now compute layout on the full graph for stability (after adding top edges)
    pos = nx.spring_layout(G_full, seed=42)
    plt.figure(figsize=(7,6))

    # Draw all base edges in light/dark grey
    if base_edges:
        print(f"DEBUG: Drawing {len(base_edges)} base edges (full graph)")
        nx.draw_networkx_edges(G_full, pos, edgelist=base_edges, edge_color="darkgrey", width=2.5, alpha=0.85)
    else:
        print(f"DEBUG: No base edges to draw")

    # Draw attributed top-k edges on top (fallback if weights are zero)
    edges_attr = [(int(i),int(j)) for (i,j,_,_) in top_edges]
    weights = [max(0.0, float(attr)) for (_,_,attr,_) in top_edges]
    if edges_attr:
        vmax = max(weights) if len(weights) > 0 else 0.0
        if vmax > 0:
            widths = [2.0 + 6.0*(w/(vmax+1e-8)) for w in weights]
            lc = nx.draw_networkx_edges(G_full, pos, edgelist=edges_attr, width=widths,
                                        edge_color=weights, edge_cmap=plt.cm.viridis)
            plt.colorbar(lc, shrink=0.7, label="Grad×Gate (top-k)")
            print(f"DEBUG: Drew {len(edges_attr)} attributed edges (colored)")
        else:
            # fallback: draw with constant color if all weights are zero
            widths = [3.0 for _ in edges_attr]
            nx.draw_networkx_edges(G_full, pos, edgelist=edges_attr, width=widths, edge_color="#ff7f0e", alpha=0.9)
            print(f"DEBUG: Drew {len(edges_attr)} attributed edges (fallback color)")
    else:
        print(f"DEBUG: No attributed edges to draw (empty list)")

    # Draw nodes last (smaller so edges are visible)
    node_colors = []
    for n in G_full.nodes:
        if n == u or n == v:
            node_colors.append("#d62728")
        else:
            node_colors.append("#1f77b4")
    nodelist = list(G_full.nodes)
    nx.draw_networkx_nodes(G_full, pos, nodelist=nodelist, node_color=node_colors, node_size=90, alpha=0.9)
    nx.draw_networkx_labels(G_full, pos, labels={n: str(n) for n in nodelist}, font_size=7)

    # draw the target pair edge explicitly (dashed red) to highlight the event
    if u in G_full.nodes and v in G_full.nodes:
        nx.draw_networkx_edges(G_full, pos, edgelist=[(u, v)], width=2.2, edge_color="#d62728", style="dashed", alpha=0.9)

    plt.title(f"Top-{len(top_edges)} attributed edges for event ({u},{v}) at t={t:.3f} | base={len(base_edges)}")
    plt.axis('off')
    out_path = os.path.join(out_dir, f"event_t{str(t).replace('.', '_')}_u{u}_v{v}.png")
    plt.tight_layout()
    # legend
    legend_handles = []
    legend_handles.append(mpatches.Patch(color="#d62728", label="Target nodes (u,v)"))
    legend_handles.append(mpatches.Patch(color="#1f77b4", label="Other nodes"))
    legend_handles.append(plt.Line2D([0], [0], color="darkgrey", lw=2.5, label=("Existing edges up to t" if overlay_mode=='upto' else "Edges at time t")))
    legend_handles.append(plt.Line2D([0], [0], color="#d62728", lw=2, linestyle="--", label="Target pair (u,v)"))
    legend_handles.append(plt.Line2D([0], [0], color="black", lw=2, label="Top-k attributed edges (colorbar, width∝attr)"))
    plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0, 1), fontsize=8, framealpha=0.85)

    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Visualize per-event subgraphs from Grad×Gate explanations")
    ap.add_argument("--csv", required=True, help="Path to gradient_x_gate_explanations.csv")
    ap.add_argument("--out_dir", required=True, help="Directory to save subgraph plots")
    ap.add_argument("--data", required=False, help="Dataset name under processed/ for optional overlay")
    ap.add_argument("--topk", type=int, default=5, help="Top-k edges to plot per event")
    ap.add_argument("--max_events", type=int, default=200, help="Max number of events to plot")
    ap.add_argument("--overlay", action="store_true", help="Overlay base graph edges present by time t")
    ap.add_argument("--overlay_mode", type=str, default="upto", choices=["upto","at"], help="Edge overlay mode: upto (<= t) or at (== t)")
    args = ap.parse_args()

    df = load_explanations(args.csv)
    grouped = group_events(df)
    os.makedirs(args.out_dir, exist_ok=True)

    count = 0
    for (t,u,v), edges in grouped.items():
        # sort by attribution desc
        edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
        overlay_flag = args.overlay or (args.data is not None)
        plot_event(edges_sorted, int(u), int(v), float(t), args.out_dir, overlay_graph=overlay_flag, data_name=args.data, overlay_mode=args.overlay_mode, topk=args.topk)
        count += 1
        if count >= args.max_events:
            break

    print(f"Saved {count} per-event subgraph plots to {args.out_dir}")


if __name__ == "__main__":
    main()


