# Triadic Closure Visualization Tools

This directory contains comprehensive tools for visualizing and analyzing **Triadic Closure** rule in dynamic graphs. Unlike contagion models where nodes activate other nodes, triadic closure shows how **existing edges enable new edges** by completing triangular structures.

## 🎯 What is Triadic Closure?

**Triadic Closure** is a fundamental principle in network science: if two nodes have a common neighbor, they are more likely to become directly connected. This creates triangular relationships.

### Process Example:
1. **Initial state**: Edges A—B and B—C exist
2. **Triadic opportunity**: A and C share neighbor B but aren't connected
3. **Closure event**: New edge A—C forms, completing the triangle
4. **Result**: Triangle A—B—C—A created

### Ground Truth Format:
```json
{
  "edge_idx": [parent_edge1, parent_edge2]
}
```
- **Key difference from contagion**: Edges cause edges (not nodes causing activations)
- Each triadic closure is enabled by exactly 2 existing edges

## 📁 Files and Tools

### Core Visualization Scripts

#### 1. `visualize_triadic_closure.py`
**Primary tool for creating animated GIFs of triadic closure evolution.**

```bash
# Basic usage - creates animated GIF
python visualize_triadic_closure.py triadic_demo

# Custom animation settings
python visualize_triadic_closure.py triadic_demo --interval 3000 --output my_animation.gif

# Create static summary instead
python visualize_triadic_closure.py triadic_demo --static
```

**Features:**
- **Animated visualization**: Shows triangle formation over time
- **Color coding**: 
  - Gray edges: Existing connections
  - Red edges: New regular edges
  - Dark red edges: Triadic closure edges
  - Yellow triangles: Newly completed triangles
  - Orange nodes: Bridge nodes enabling closures
- **Multi-panel layout**: Graph + Info panel + Growth statistics
- **Real-time stats**: Triadic closure rate, triangle count

#### 2. `compare_triadic_patterns.py`
**Tool for comparing multiple triadic closure patterns side by side.**

```bash
# Generate diverse datasets and compare
python compare_triadic_patterns.py --generate

# Compare specific datasets
python compare_triadic_patterns.py dataset1 dataset2 dataset3

# Custom output file
python compare_triadic_patterns.py --generate --output comparison.png
```

**Features:**
- **Side-by-side comparison**: Multiple datasets in one visualization
- **Pattern analysis**: Different initialization effects
- **Summary statistics**: Triadic rates, triangle counts
- **Growth curves**: Edge vs triadic closure evolution

## 🎬 Generated Visualizations

### Animated GIFs (Example outputs)

1. **`triadic_closure_demo_triadic_closure.gif`** (685K)
   - Basic triadic closure demo
   - 21 timesteps, 20 triadic closures
   - Shows step-by-step triangle formation

2. **`triadic_closure_detailed.gif`** (644K)
   - Slower animation (4 second intervals)
   - Detailed view of closure process
   - Clear visualization of bridge nodes

3. **`triadic_dense_triadic_closure.gif`** (662K)
   - Dense initialization scenario
   - More initial edges → more closure opportunities
   - Demonstrates cascading effects

### Static Summaries

1. **`triadic_closure_demo_triadic_summary.png`** (375K)
   - Before/after comparison
   - Growth statistics
   - Process explanation

2. **`triadic_patterns_comparison_3_datasets.png`** (384K)
   - Sparse vs Medium vs Dense patterns
   - Comparative triadic closure rates
   - Different growth trajectories

## 🔍 Understanding the Visualizations

### Animation Components

#### Main Graph Panel
- **Node colors**:
  - Light blue: Regular nodes
  - Orange: Bridge nodes (enabling new triangles)
- **Edge colors**:
  - Gray: Existing edges
  - Red: New regular edges
  - Dark red: Triadic closure edges
- **Triangle highlighting**: Yellow background for newly formed triangles

#### Information Panel
- Current timestep
- New edges this timestep
- Triangle formation events
- Running total of triadic closures

#### Statistics Panel
- **Blue line**: Total edge count over time
- **Red line**: Cumulative triadic closures
- **Triadic rate**: Percentage of edges formed through closure

### Key Metrics

#### Triadic Closure Rate
```
Triadic Rate = (Triadic Closures / Total Edges) × 100%
```
- **Higher rates**: More structured growth
- **Lower rates**: More random connections
- **Typical range**: 40-70% in good triadic datasets

#### Triangle Count
- Total triangles in final graph
- Not all triangles are from triadic closure (some from initial edges)
- Growth pattern reveals closure cascades

## 📊 Dataset Characteristics

### Generated Test Datasets

| Dataset | Nodes | Init Edges | Timesteps | Triadic Rate | Triangles |
|---------|-------|------------|-----------|--------------|-----------|
| triadic_sparse | 25 | 8 | 15 | 50.0% | 18 |
| triadic_medium | 30 | 15 | 20 | 57.1% | 45 |
| triadic_dense | 35 | 25 | 25 | 51.4% | 31 |

### Observations
- **Medium initialization** achieves highest triadic rates
- **Dense initialization** has fewer closure opportunities (already connected)
- **Sparse initialization** has structural limitations

## 🔧 Customization Options

### Animation Parameters
```bash
--interval MILLISECONDS    # Time between frames (default: 2500ms)
--figsize WIDTH HEIGHT     # Figure dimensions (default: 14 10)
--output FILENAME          # Custom output filename
```

### Dataset Generation
Modify `generate_diverse_triadic_datasets()` in `compare_triadic_patterns.py`:
```python
'params': {
    'num_nodes': 30,          # Network size
    'num_initial_edges': 15,  # Starting connectivity
    'num_timesteps': 20,      # Evolution duration
    'noise_ratio': 0.05       # Random edge probability
}
```

## 🧠 Insights from Visualizations

### Structural Patterns
1. **Bridge nodes** are critical - they enable multiple closures
2. **Cascading effects** - early triangles enable later triangles
3. **Temporal clustering** - closures often happen in bursts

### Network Evolution
1. **Early phase**: Random initial structure
2. **Growth phase**: Triadic opportunities emerge
3. **Saturation**: Fewer closure opportunities remain

### Ground Truth Insights
- **Edge causality**: Each closure has exactly 2 parent edges
- **Temporal dependencies**: Later edges depend on earlier edges
- **Collective effects**: One triangle can enable multiple future closures

## 🎓 Educational Value

### For Understanding Dynamic Networks
- **Structural completion** vs **contagion processes**
- **Edge dependencies** in graph evolution
- **Triangle formation** mechanisms

### For TGNN Research
- **Explanation quality**: How well can models identify parent edges?
- **Temporal reasoning**: Understanding long-range dependencies
- **Structural patterns**: Recognition of triangular motifs

## 🚀 Quick Start Guide

1. **Generate a basic visualization**:
   ```bash
   python visualize_triadic_closure.py my_triadic_test
   ```

2. **Compare different patterns**:
   ```bash
   python compare_triadic_patterns.py --generate
   ```

3. **Create custom animation**:
   ```bash
   python visualize_triadic_closure.py my_dataset --interval 5000 --static
   ```

## 🔍 Troubleshooting

### Common Issues

1. **Empty animations**: Check if dataset has triadic closures
   ```bash
   # Verify ground truth exists
   ls processed/YOUR_DATASET/ml_YOUR_DATASET_gt.json
   ```

2. **Color errors**: Make sure matplotlib is updated
   ```bash
   pip install matplotlib --upgrade
   ```

3. **Font warnings**: Normal - emojis may not render in GIF
   - Affects only emoji symbols (🔺, ➕)
   - Does not affect visualization quality

### Performance Tips

- **Larger datasets**: Use higher `--interval` values
- **File size**: Reduce `figsize` for smaller GIFs
- **Quality**: Increase DPI in static summaries

---

**The triadic closure visualizations reveal the elegant process by which network structure creates its own growth opportunities - existing triangular patterns enabling new triangular completions in a self-reinforcing cycle.** 