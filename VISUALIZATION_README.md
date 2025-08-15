# Contagion Dataset Visualization Tools

This directory contains tools for visualizing the temporal evolution of synthetic contagion datasets as animated GIFs and static summaries.

## Available Tools

### 1. `visualize_contagion.py` - Main Visualization Tool

Creates animated GIFs and static summaries showing how contagion processes unfold over time.

**Usage:**
```bash
# Create animated GIF (default)
python visualize_contagion.py <dataset_name> [options]

# Create static summary
python visualize_contagion.py <dataset_name> --static

# Examples
python visualize_contagion.py synthetic_ltm_ba
python visualize_contagion.py synthetic_icm_ba --static
python visualize_contagion.py synthetic_cc_ws --interval 2000 -o custom_name.gif
```

**Options:**
- `--static`: Create static before/after comparison instead of animation
- `--output`, `-o`: Custom output filename
- `--interval`: Animation frame interval in milliseconds (default: 1500)
- `--figsize`: Figure size as two integers (default: 12 8)

### 2. `compare_contagion_models.py` - Model Comparison

Creates side-by-side comparisons of different contagion models showing their final activation states.

**Usage:**
```bash
# Compare multiple models
python compare_contagion_models.py <dataset1> <dataset2> ... [options]

# Example
python compare_contagion_models.py synthetic_ltm_ba synthetic_icm_ba synthetic_cc_ws synthetic_sd_ba
```

**Options:**
- `--output`, `-o`: Custom output filename

## Output Files

### Generated Visualizations

The tools create several types of visualization files:

1. **Animated GIFs** (`*_contagion.gif`):
   - Show temporal evolution of the contagion process
   - Each frame represents a time step with node activations
   - Color coding: Gray (inactive), Red (active), Orange (newly activated)
   - Red edges highlight activation relationships

2. **Static Summaries** (`*_summary.png`):
   - Before/after comparison showing initial and final states
   - Useful for quick overview of activation patterns

3. **Model Comparisons** (`contagion_models_comparison.png`):
   - Side-by-side comparison of multiple models
   - Shows final activation states and statistics

### Color Scheme

- **Gray nodes**: Inactive/susceptible
- **Red nodes**: Activated/infected
- **Orange nodes**: Newly activated in current frame
- **Red edges**: Edges causing activation (in animations)
- **Gray edges**: Regular network structure

## Interpretation Guide

### Model-Specific Patterns

**Linear Threshold Model (LTM)**:
- Collective activation: nodes activate when enough neighbors are active
- Typically shows cascading activation patterns
- Ground truth explanations include all contributing neighbors

**Independent Cascade Model (ICM)**:
- Probabilistic activation: each active neighbor tries to activate others
- Shows more scattered, probabilistic activation patterns  
- Ground truth explanations include the successful activator

**Complex Contagion (CC)**:
- Requires quorum of neighbors to activate
- Shows cluster-based activation patterns
- Ground truth explanations include all members of the activating quorum

**Structural Diversity (SD)**:
- Based on network structure diversity
- May show limited activation in some network types
- Ground truth explanations include induced subgraph nodes

### Statistics Displayed

For each model, the visualization shows:
- **Activation rate**: Percentage of nodes that became active
- **Number of events**: Total activation events over time  
- **Total edges**: Size of the underlying network
- **Timeline**: Temporal spread of activations

## Example Workflow

```bash
# 1. Generate synthetic datasets (if not already done)
python generate_synthetic_data.py --model ltm --graph ba
python generate_synthetic_data.py --model icm --graph ba
python generate_synthetic_data.py --model cc --graph ws
python generate_synthetic_data.py --model sd --graph ba

# 2. Create individual visualizations
python visualize_contagion.py synthetic_ltm_ba --static
python visualize_contagion.py synthetic_ltm_ba --interval 1000

# 3. Compare models side-by-side
python compare_contagion_models.py synthetic_ltm_ba synthetic_icm_ba synthetic_cc_ws synthetic_sd_ba

# 4. View results
ls *.gif *.png
```

## Technical Details

### Dependencies

Required packages (install with `pip install -r requirements.txt`):
- `matplotlib>=3.7.1`: Plotting and animation
- `pillow>=9.5.0`: GIF creation
- `networkx>=3.0`: Graph operations
- `pandas>=1.5.3`: Data handling
- `numpy>=1.24.3`: Numerical operations

### File Structure

The visualization tools expect the following dataset structure:
```
processed/
├── <dataset_name>/
│   ├── ml_<dataset_name>.csv           # Edge data with columns: u, i, ts, label, idx
│   └── <dataset_name>_explanations.json # Ground truth explanations
```

### Performance Considerations

- **Animation duration**: Scales with number of activation events
- **Large networks**: Consider using `--static` for networks with >100 nodes
- **File sizes**: Animated GIFs can be several MB for long simulations
- **Memory usage**: Large datasets may require reducing figure size

## Troubleshooting

### Common Issues

1. **"No activation events found"**: 
   - Dataset has no causal edges (label=1)
   - Check if dataset generation was successful

2. **"KeyError: 'v'"**:
   - Column name mismatch in dataset
   - Ensure dataset has columns 'u', 'i', 'ts', 'label', 'idx'

3. **"ModuleNotFoundError: matplotlib"**:
   - Install visualization dependencies: `pip install matplotlib pillow`

4. **Large file sizes**:
   - Reduce `--interval` for faster animations
   - Use `--static` for overview visualizations
   - Reduce `--figsize` for smaller files

### Dataset-Specific Notes

- **SD models** may produce no activations in some network types
- **Small datasets** may have all activations in early time periods
- **Dense networks** may show rapid, complete activation
- **Sparse networks** may show limited propagation

## Advanced Usage

### Custom Layouts

To modify network layouts, edit the `visualize_contagion.py` file:
```python
# Change from spring layout to circular layout
pos = nx.circular_layout(G)

# Use hierarchical layout for directed graphs
pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
```

### Color Schemes

Modify the color scheme in `visualize_contagion.py`:
```python
colors = {
    'inactive': 'lightblue',    # Change inactive color
    'active': 'darkred',        # Change active color
    'newly_active': 'yellow',   # Change newly active color
    'activating_edge': 'green'  # Change edge color
}
```

### Batch Processing

For processing multiple datasets:
```bash
# Create all static summaries
for dataset in synthetic_*; do
    python visualize_contagion.py $dataset --static
done

# Create all animations (time-intensive)
for dataset in synthetic_*; do
    python visualize_contagion.py $dataset --interval 1000
done
``` 