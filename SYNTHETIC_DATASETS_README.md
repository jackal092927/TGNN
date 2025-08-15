# Synthetic Contagion Datasets for TGIB

This codebase now supports synthetic temporal graph datasets generated from well-established contagion models. These datasets provide **ground truth explanations** for each activation event, enabling rigorous evaluation of temporal graph neural network explainability.

## Overview

The synthetic datasets are generated using four different contagion models, each with distinct causal mechanisms:

### 1. Linear Threshold Model (LTM)
- **Causal Mechanism**: Collective Influence
- **Activation Rule**: Node activates when sum of influence from active neighbors > threshold
- **Ground Truth Explanation**: All active neighbors that contributed to exceeding the threshold
- **Example**: Node A activates because neighbors B, C, D were all active and their combined influence (0.3 + 0.4 + 0.4 = 1.1) exceeded A's threshold (0.8)

### 2. Independent Cascade Model (ICM)  
- **Causal Mechanism**: Individual Stochastic Activation
- **Activation Rule**: Each newly active neighbor gets one chance to activate node with probability p
- **Ground Truth Explanation**: The single neighbor whose activation attempt succeeded
- **Example**: Node A activates because neighbor B's activation attempt succeeded (with probability 0.3)

### 3. Complex Contagion Model (CC)
- **Causal Mechanism**: Collective Quorum
- **Activation Rule**: Node activates when it has ≥ k active neighbors
- **Ground Truth Explanation**: All active neighbors that formed the required quorum
- **Example**: Node A activates because it had 3 active neighbors (B, C, D) and needed k=3

### 4. Structural Diversity Model (SD)
- **Causal Mechanism**: Local Network Structure
- **Activation Rule**: Node activates when structural diversity of active neighbors > threshold
- **Ground Truth Explanation**: The induced subgraph of active neighbors (nodes + edges)
- **Example**: Node A activates because neighbors B, C, D were active with low interconnection (diversity = 0.7 > 0.5)

## Generated Datasets

The following datasets have been created and are ready to use:

```
processed/
├── synthetic_ltm_ba/         # Linear Threshold + Barabási-Albert
├── synthetic_icm_ba/         # Independent Cascade + Barabási-Albert  
├── synthetic_cc_ws/          # Complex Contagion + Watts-Strogatz
└── synthetic_sd_ba/          # Structural Diversity + Barabási-Albert
```

Each dataset directory contains:
- `ml_*.csv` - Processed edge data for training
- `ml_*.npy` - Edge feature vectors
- `ml_*_node.npy` - Node feature vectors
- `*_explanations.json` - Ground truth explanations

## Usage

### Training and Evaluation

```bash
# Train on Linear Threshold Model dataset
python learn_edge.py -d synthetic_ltm_ba --n_epoch 100

# Train on Independent Cascade Model dataset  
python learn_edge.py -d synthetic_icm_ba --n_epoch 100

# Evaluate pre-trained model (use actual model filename)
python learn_edge.py -d synthetic_ltm_ba --eval_only --load_model saved_models/-attn-prod-synthetic_ltm_ba-42.pth
```

### Generating New Datasets

```bash
# Generate Linear Threshold Model with Barabási-Albert graph
python generate_synthetic_data.py --model ltm --graph ba --nodes 1000 --edges 2000

# Generate Independent Cascade with Erdős-Rényi graph
python generate_synthetic_data.py --model icm --graph er --nodes 500 --edges 1000

# Generate Complex Contagion with Watts-Strogatz graph  
python generate_synthetic_data.py --model cc --graph ws --nodes 800 --timesteps 30

# Generate Structural Diversity with more seeds (needed for activations)
python generate_synthetic_data.py --model sd --graph ba --nodes 200 --seeds 0.15
```

### Available Parameters

- `--model`: Contagion model (`ltm`, `icm`, `cc`, `sd`)
- `--graph`: Graph topology (`ba`, `er`, `ws`)
- `--nodes`: Number of nodes (default: 1000)
- `--edges`: Number of edges (default: 2000)
- `--seeds`: Fraction of seed nodes (default: 0.05)
- `--timesteps`: Maximum simulation timesteps (default: 50)
- `--seed`: Random seed for reproducibility (default: 42)

### Model File Naming

Trained models are saved with the pattern: `{prefix}-{agg_method}-{attn_mode}-{dataset}-{seed}.pth`

Examples:
- `saved_models/-attn-prod-synthetic_ltm_ba-42.pth`
- `saved_models/-attn-prod-synthetic_icm_ba-42.pth`
- `saved_models/myexp-attn-prod-synthetic_cc_ws-42.pth` (with `--prefix myexp`)

To find available models: `ls saved_models/*synthetic*`

### Helper Script

Use the provided helper script to list available models and get evaluation commands:

```bash
# List all synthetic dataset models
python list_models.py --synthetic

# List models matching a pattern
python list_models.py --filter ltm

# List all models
python list_models.py
```

## Evaluation Metrics

When training on synthetic datasets, the system automatically evaluates three types of explainability metrics:

### 1. Explanation Faithfulness (Sparsity-Fidelity Curve)
- Measures how well explanations preserve model predictions
- Calculates AUC of sparsity vs. fidelity curve
- Higher AUC = better explanation faithfulness

### 2. Counterfactual Analysis  
- Measures prediction changes when removing important nodes
- Calculates AUC of sparsity vs. flip rate curve
- Higher AUC = more meaningful explanations

### 3. Ground Truth Accuracy (Available only for synthetic datasets)
- Compares model explanations to known ground truth
- Reports Precision@k, Recall@k, and MRR
- Higher scores = better explanation accuracy

## Ground Truth Format

The ground truth explanations are stored as JSON files mapping edge indices to causal explanations:

```json
{
  "0": [66],                    // ICM: single activator
  "1": [4, 14, 31],            // LTM: all contributing neighbors  
  "2": {                       // SD: structural explanation
    "nodes": [1, 4, 7],
    "edges": [[1, 4]],
    "diversity": 0.67
  }
}
```

## Implementation Details

### Temporal Correctness
- All datasets maintain strict temporal ordering
- Activation events only depend on prior states
- Edge timestamps reflect causal relationships

### Data Format Compatibility
- Generated datasets use the same format as existing datasets
- Compatible with all existing TGIB training and evaluation pipelines
- Node IDs are automatically remapped for consistency

### Scalability
- Small datasets (100-200 nodes) for quick testing
- Medium datasets (500-1000 nodes) for full evaluation
- Large datasets can be generated for production use

## Example Results

Training on `synthetic_ltm_ba` with 5 epochs shows explainability metrics:

```
=== Explanation Evaluation ===
Explanation AUC (Sparsity-Fidelity): 0.0
Explanation AUC (Counterfactual): 0.0
Node-level metrics:
  Precision@1: 0.6000
  Precision@2: 0.3000  
  Recall@2:    0.3000
  Recall@5:    0.3000
  MRR:         0.6000
```

This shows that even with minimal training, the model achieves 60% precision at finding the most important explanatory node.

## Troubleshooting

### Model Loading Errors

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory: 'saved_models/model.pth'`

**Solution**: Use the correct model filename pattern. Models are saved as `{prefix}-{agg_method}-{attn_mode}-{dataset}-{seed}.pth`. Use the helper script to find the correct path:

```bash
python list_models.py --synthetic
```

### No Causal Edges in Test Set

**Problem**: "No causal edges found in test set for evaluation"

**Explanation**: This is normal for small synthetic datasets where the temporal split (70% train, 15% val, 15% test) may not include causal edges in the test portion.

**Solutions**:
- Generate larger datasets with more activations
- Use different evaluation split ratios
- Increase seed fraction for more initial activations

### Zero AUC Scores

**Problem**: Explanation AUC and Counterfactual AUC are 0.0

**Explanation**: This often occurs with minimally trained models (few epochs) where:
- The model hasn't learned meaningful patterns yet
- All predictions are similar (no positive predictions)
- Explanations don't significantly affect predictions

**Solutions**:
- Train for more epochs (50-100)
- Use larger datasets with clear patterns
- Verify the model is learning (check accuracy/AUC improves)

## Future Extensions

### Additional Models
- Threshold models with varying weights
- Multi-layer contagion processes
- Adaptive threshold models

### Advanced Features  
- Temporal explanation windows
- Multi-step causal chains
- Heterogeneous node types

### Evaluation Enhancements
- Explanation stability metrics
- Cross-model generalization tests
- Human-interpretable visualizations 