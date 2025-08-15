# GraphMamba Contagion Training and Visualization

This repository contains tools for training self-explaining GraphMamba models on contagion data and visualizing the interpretation results.

## Features

- **Configurable save directory**: Results are saved to a configurable location
- **Detailed predictions storage**: Model predictions, gates, and embeddings are stored for visualization
- **Separate visualization**: Training/evaluation and visualization are completely separated
- **Comprehensive plots**: Multiple visualization types including temporal analysis and gate importance

## Files

- `test_graphmamba_contagion_explain.py`: Main training script with configurable save_dir
- `visualize_contagion.py`: Separate visualization module with comprehensive plotting functions

## Usage

### Training Mode

```bash
# Basic training with default save directory
python test_graphmamba_contagion_explain.py --mode train --data synthetic_icm_ba

# Training with custom save directory
python test_graphmamba_contagion_explain.py --mode train \
    --data synthetic_icm_ba \
    --epochs 200 \
    --save_dir ./experiments/run1

# Full parameter customization
python test_graphmamba_contagion_explain.py --mode train \
    --data synthetic_icm_ba \
    --epochs 200 \
    --lr 0.0005 \
    --hidden_dim 256 \
    --pos_dim 128 \
    --mamba_state_dim 32 \
    --lambda_sparse 1e-5 \
    --lambda_tv 1e-4 \
    --gate_temperature 0.8 \
    --save_dir ./experiments/run2
```

### Visualization Mode

```bash
# Visualize results from training
python test_graphmamba_contagion_explain.py --mode visualize \
    --results_file ./experiments/run1/synthetic_icm_ba_results.json \
    --save_dir ./experiments/run1/plots

# Use separate visualization script
python visualize_contagion.py \
    --results_file ./experiments/run1/synthetic_icm_ba_results.json \
    --save_dir ./experiments/run1/plots

# Only create temporal analysis plots
python visualize_contagion.py \
    --results_file ./experiments/run1/synthetic_icm_ba_results.json \
    --temporal_only
```

## Output Files

### Training Results
- `{data_name}_results.json`: Complete results including metrics, hyperparameters, and detailed predictions
- Contains: accuracy, AUC, AP scores, predictions, gate values, timestamps, and embeddings

### Visualization Outputs
- `contagion_interpretation_plots.png`: 6-panel comprehensive visualization
- `temporal_analysis_plots.png`: 4-panel temporal analysis
- `gate_heatmap.png`: Gate values over time and nodes
- `contagion_interpretation_report.txt`: Detailed analysis report

## Visualization Types

### Main Interpretation Plots
1. **Prediction Distribution**: Histogram of positive vs negative predictions
2. **ROC Curve**: Receiver Operating Characteristic curve
3. **Precision-Recall Curve**: Precision vs recall relationship
4. **Gate Importance Over Time**: How gate values evolve temporally
5. **Gate Sparsity Analysis**: Distribution of gate values
6. **Prediction vs Gate Correlation**: Relationship between predictions and gate values

### Temporal Analysis Plots
1. **Gate Evolution**: Gate values over time
2. **Prediction Accuracy**: Accuracy over time
3. **Gate Sparsity**: Sparsity ratio over time
4. **Prediction Confidence**: Confidence scores over time

### Additional Visualizations
- **Gate Heatmap**: 2D visualization of gate values across nodes and time
- **Detailed Reports**: Text-based analysis with statistics and insights

## Dependencies

### Required
- torch
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

### Optional
- tqdm (for progress bars)

## Example Workflow

1. **Train the model**:
   ```bash
   python test_graphmamba_contagion_explain.py --mode train \
       --data synthetic_icm_ba \
       --epochs 100 \
       --save_dir ./experiments/contagion_run1
   ```

2. **Visualize results**:
   ```bash
   python visualize_contagion.py \
       --results_file ./experiments/contagion_run1/synthetic_icm_ba_results.json \
       --save_dir ./experiments/contagion_run1/plots
   ```

3. **Analyze temporal patterns**:
   ```bash
   python visualize_contagion.py \
       --results_file ./experiments/contagion_run1/synthetic_icm_ba_results.json \
       --temporal_only
   ```

## Configuration Options

### Training Parameters
- `--data`: Dataset name (e.g., synthetic_icm_ba, synthetic_ltm_ba)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--hidden_dim`: Hidden dimension size
- `--pos_dim`: Positional embedding dimension
- `--mamba_state_dim`: Mamba state dimension
- `--lambda_sparse`: Sparsity loss weight
- `--lambda_tv`: Temporal smoothness loss weight
- `--gate_temperature`: Gate temperature for smoothing

### Visualization Parameters
- `--results_file`: Path to results JSON file
- `--save_dir`: Directory to save visualization outputs
- `--no_save`: Don't save plots to files
- `--no_show`: Don't display plots interactively
- `--temporal_only`: Only create temporal analysis plots

## Notes

- The training script automatically creates the save directory if it doesn't exist
- Detailed results are only stored for the best validation model
- Visualization can be run independently after training
- All plots are saved in high resolution (300 DPI) for publication quality
- The system gracefully handles missing visualization dependencies
