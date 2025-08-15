# Temporal Graph Autoregressive Model (TGAM)

## Overview

TGAM (Temporal Graph Autoregressive Model) is a new approach to temporal graph modeling that learns the sequential generation process of temporal graphs. Unlike TGIB, which distinguishes between real and fake edges at specific timesteps, TGAM models the entire temporal graph formation process autoregressively.

## Key Differences from TGIB

| Aspect | TGIB | TGAM |
|--------|------|------|
| **Learning Approach** | Discriminative (real vs fake edges) | Generative (sequential edge prediction) |
| **Training Data** | Positive/negative edge pairs | Temporal sequences of graph states |
| **Architecture** | Temporal convolution + MLP | Graph encoder + Transformer + Edge predictor |
| **Strength** | Works well on real datasets with chaotic patterns | Better suited for synthetic datasets with structured patterns |
| **Use Case** | Link prediction on complex real networks | Graph generation and structured temporal prediction |

## Why TGAM for Synthetic Datasets?

Our analysis revealed that TGIB struggles with synthetic datasets because:

1. **TGIB relies on "messy diversity"** - chaotic temporal neighborhood patterns found in real-world data
2. **Synthetic datasets are "too clean"** - mathematical generation creates predictable, structured patterns
3. **TGIB's discriminative approach** requires extreme variations that synthetic models smooth out

TGAM addresses this by:
- **Learning generative patterns** directly from the data
- **Modeling sequential dependencies** that capture synthetic generation rules
- **Using autoregressive prediction** that works with structured patterns

## Architecture

### Components

1. **GraphStateEncoder**: Encodes current graph state using message passing
2. **TemporalSequenceEncoder**: Uses Transformer to model temporal dependencies
3. **EdgePredictor**: Predicts next edge given current context
4. **PositionalEncoding**: Handles temporal positioning in sequences

### Model Flow

```
Input Sequence → Graph Encoding → Temporal Encoding → Edge Prediction → Output
     ↓               ↓               ↓                    ↓
[Graph States]  [Node Embeddings] [Sequence Context] [Next Edge Probability]
```

## Installation & Setup

### Requirements

```bash
pip install torch pandas numpy scikit-learn tqdm matplotlib seaborn
```

### Quick Test

Run the test script to verify everything works:

```bash
python test_tgam.py
```

## Usage

### 1. Training TGAM on a Dataset

```bash
python train_tgam.py -d <dataset_name> --gpu 0 --n_epoch 50
```

**Arguments:**
- `-d, --data`: Dataset name (e.g., 'wikipedia', 'synthetic_ba')
- `--bs`: Batch size (default: 32)
- `--n_epoch`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--hidden_dim`: Hidden dimension (default: 128)
- `--seq_len`: Sequence length (default: 10)
- `--gpu`: GPU index (default: 0)

### 2. Comparing TGIB vs TGAM

Run comprehensive comparison on all available datasets:

```bash
python compare_models.py --gpu 0 --epochs 20
```

**Options:**
- `--real-only`: Test only real datasets
- `--synthetic-only`: Test only synthetic datasets
- `--datasets dataset1 dataset2`: Test specific datasets
- `--skip-tgib`: Skip TGIB experiments
- `--skip-tgam`: Skip TGAM experiments

### 3. Using TGAM Programmatically

```python
from tgam import TGAM
from train_tgam import prepare_sequences
import torch

# Load your data
# g_df, n_feat, e_feat = load_your_data()

# Prepare sequences
sequences = prepare_sequences(
    src_l, dst_l, ts_l, e_idx_l, 
    node_features, edge_features,
    sequence_length=10, step_size=5
)

# Initialize model
model = TGAM(
    node_feat_dim=n_feat.shape[1],
    edge_feat_dim=e_feat.shape[1],
    hidden_dim=128,
    max_nodes=max_node_id,
    num_graph_layers=2,
    num_temporal_layers=6
)

# Training step
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Forward pass
src_logits, dst_logits, edge_features = model(sequence, target_time)

# Generate new edges
generated_edges, sequence = model.generate_sequence(
    initial_graph, start_time=0, num_steps=10
)
```

## Model Parameters

### GraphStateEncoder
- `node_feat_dim`: Dimension of node features
- `edge_feat_dim`: Dimension of edge features  
- `hidden_dim`: Hidden dimension for embeddings
- `num_layers`: Number of message passing layers (default: 2)

### TemporalSequenceEncoder
- `hidden_dim`: Must match GraphStateEncoder
- `num_heads`: Number of attention heads (default: 8)
- `num_layers`: Number of transformer layers (default: 6)

### EdgePredictor
- `hidden_dim`: Must match other components
- `max_nodes`: Maximum number of nodes in the graph

## Expected Performance

Based on our analysis, TGAM should perform better on:

### Synthetic Datasets
- **Triadic closure models**: Structured triangle formation
- **Preferential attachment**: Scale-free network generation  
- **Small-world models**: Regular rewiring patterns
- **Contagion models**: Propagation-based edge formation

### Characteristics TGAM Handles Well
- **Structured temporal patterns**
- **Mathematical generation rules**
- **Predictable graph evolution**
- **Regular temporal dependencies**

## Output & Results

### Training Output
- Training loss per epoch
- Validation metrics (ACC, AUC, AP, F1) every 5 epochs
- Best model checkpointing
- Early stopping when performance plateaus

### Comparison Results
The comparison script generates:

1. **CSV Results**: `comparison_results/comparison_results_<timestamp>.csv`
2. **Visualization**: `comparison_results/model_comparison_by_type.png`
3. **Summary Report**: `comparison_results/comparison_summary.txt`
4. **Logs**: `comparison_results/comparison_<timestamp>.log`

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--bs` (batch size) or `--hidden_dim`
2. **Slow Training**: Reduce `--seq_len` or `--num_temporal_layers`
3. **Poor Performance**: Increase `--seq_len` or adjust learning rate

### Debug Mode

For debugging, use smaller models:
```bash
python train_tgam.py -d <dataset> --hidden_dim 64 --seq_len 5 --num_temporal_layers 2
```

## Results Interpretation

### When TGAM Wins
- **Synthetic datasets**: Better modeling of generation rules
- **Structured patterns**: Captures mathematical relationships
- **Predictable evolution**: Benefits from autoregressive approach

### When TGIB Wins  
- **Real-world datasets**: Benefits from chaotic neighborhood diversity
- **Complex interactions**: Leverages temporal convolution strengths
- **Irregular patterns**: Thrives on organic complexity

## Future Improvements

1. **Hybrid Models**: Combine TGIB's discrimination with TGAM's generation
2. **Attention Mechanisms**: Better temporal dependency modeling
3. **Multi-scale Modeling**: Handle different temporal resolutions
4. **Graph-specific Architectures**: Specialized encoders for different graph types

## Citation

If you use TGAM in your research, please cite:

```bibtex
@article{tgam2024,
  title={Temporal Graph Autoregressive Model: A Generative Approach for Synthetic Graph Modeling},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 