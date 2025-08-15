# GraphMamba Contagion Model Evaluation Framework

This framework provides comprehensive evaluation and visualization tools for GraphMamba contagion models, implementing all the TGIB evaluation strategies you mentioned.

## 🎯 Evaluation Strategies Implemented

### 1. **Explanation-Only Performance (TGIB-style)**
- **What it does**: Keeps only top-ρ% gated edges to build R_k, then predicts using only R_k
- **Metrics**: Reports AP/AUC vs sparsity ρ
- **Purpose**: Measures how well explanations preserve model performance
- **Output**: Sparsity vs. AP/AUC curves

### 2. **Fidelity Curves**
- **Deletion AUC**: Remove explanation edges and measure prediction drop
- **Insertion AUC**: Insert explanation edges and measure prediction recovery
- **Purpose**: Quantifies how much the model depends on the selected edges
- **Output**: Sparsity vs. deletion/insertion AUC curves

### 3. **Process-Grounded Metrics**
- **Path-Recall@k**: Fraction of edges on minimal diffusion paths captured by top-k gates
- **Counterfactual Drop**: Re-simulate with edges removed, measure performance drop
- **Temporal Coverage**: How many timesteps do the top edges span
- **Purpose**: Grounds explanations in the actual contagion process
- **Output**: k vs. various process metrics

### 4. **Parsimony & Stability**
- **Sparsity**: |R_k|/|G_k| ratio over time
- **Temporal Variation (TV)**: Mean |gates_t - gates_{t-1}| on common edges
- **Jaccard Stability**: Overlap of selected edges under small perturbations
- **Purpose**: Ensures explanations are sparse, temporally coherent, and robust
- **Output**: Temporal plots and stability metrics

## 🚀 Quick Start

### Option 1: Run Demo (Recommended for first-time users)
```bash
# Install dependencies
pip install -r requirements_evaluation.txt

# Run quick evaluation demo
python evaluate_contagion_model.py --demo
```

This will:
1. Load the `synthetic_icm_ba` dataset
2. Train a GraphMamba model (20 epochs)
3. Run comprehensive evaluation
4. Generate all visualization plots
5. Create summary report

### Option 2: Evaluate Your Own Model
```bash
# Evaluate on specific dataset
python evaluate_contagion_model.py --data synthetic_icm_ba --gpu 0 --save_dir ./my_results

# Evaluate on different dataset
python evaluate_contagion_model.py --data synthetic_ltm_ba --gpu 0 --save_dir ./ltm_results
```

### Option 3: Use the Comprehensive Framework Directly
```python
from evaluate_graphmamba_contagion import ContagionEvaluator, ContagionVisualizer

# Create evaluator
evaluator = ContagionEvaluator(model, graph_sequence, g_df, timestamps, device, logger)

# Run all evaluations
results = evaluator.run_comprehensive_evaluation()

# Visualize results
visualizer = ContagionVisualizer(results, save_dir)
visualizer.plot_explanation_performance()
visualizer.plot_fidelity_curves()
visualizer.plot_process_grounded_metrics()
visualizer.plot_parsimony_stability()

# Get summary report
summary = visualizer.create_summary_report()
print(summary)
```

## 📊 Output Files

The evaluation generates several files in your specified output directory:

### Visualization Plots
- `explanation_performance.png` - AP/AUC vs sparsity curves
- `fidelity_curves.png` - Deletion/insertion AUC curves
- `process_grounded_metrics.png` - Path recall, counterfactual drops, temporal coverage
- `parsimony_stability.png` - Sparsity ratios, temporal variations, Jaccard stability

### Text Reports
- `evaluation_summary.txt` - Comprehensive summary of all metrics

## 🔧 Customization

### Adjusting Evaluation Parameters
```python
# Customize sparsity levels
sparsity_levels = np.arange(0.1, 1.01, 0.1)  # 10% to 100% in 10% steps

# Customize k values for process-grounded metrics
k_values = [5, 10, 15, 20, 25, 30]

# Customize perturbation levels for stability
perturbation_levels = [0.01, 0.02, 0.05, 0.1]
```

### Adding New Metrics
```python
class ContagionEvaluator:
    def evaluate_custom_metric(self):
        """Add your custom evaluation metric here"""
        # Your implementation
        pass
```

## 📈 Interpreting Results

### Good Explanation Performance
- **High AP/AUC at low sparsity** (e.g., AP > 0.8 at 20% sparsity)
- **Steep fidelity curves** (large prediction drops when removing edges)
- **High path recall** (explanations capture actual diffusion paths)
- **Low temporal variation** (explanations are temporally stable)

### Red Flags
- **Flat performance curves** (explanations don't matter)
- **Low fidelity** (model doesn't depend on explanations)
- **High temporal variation** (explanations are unstable)
- **Low path recall** (explanations miss actual causal edges)

## 🎯 Use Cases

### Research & Development
- Compare different model architectures
- Analyze the impact of regularization (sparsity, TV losses)
- Understand temporal dynamics of explanations

### Model Validation
- Ensure explanations are faithful to model behavior
- Verify temporal coherence of edge importance
- Check robustness to perturbations

### Paper Writing
- Generate publication-quality plots
- Quantify explanation quality with multiple metrics
- Compare against baseline methods

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size or use CPU
   python evaluate_contagion_model.py --gpu -1
   ```

2. **Dataset not found**
   ```bash
   # Check dataset path
   ls ./processed/
   # Ensure dataset exists in ./processed/{dataset_name}/
   ```

3. **Model loading errors**
   ```python
   # Check model architecture matches
   # Ensure all required parameters are provided
   ```

### Performance Tips

- **GPU memory**: Use smaller models for large graphs
- **Evaluation speed**: Reduce sparsity levels or k values for faster evaluation
- **Memory usage**: Process timesteps in batches for very long sequences

## 📚 Advanced Usage

### Batch Evaluation
```python
# Evaluate multiple models
datasets = ['synthetic_icm_ba', 'synthetic_ltm_ba', 'synthetic_cc_ws']
for dataset in datasets:
    results = evaluate_trained_model(dataset, gpu_id=0, save_dir=f'./results_{dataset}')
```

### Custom Visualization
```python
# Create custom plots
import matplotlib.pyplot as plt

# Example: Compare multiple models
plt.figure(figsize=(10, 6))
for model_name, results in all_results.items():
    plt.plot(results['explanation_only']['sparsity_levels'], 
             results['explanation_only']['ap_scores'], 
             label=model_name, marker='o')
plt.xlabel('Sparsity Level')
plt.ylabel('Average Precision')
plt.legend()
plt.show()
```

### Integration with Other Frameworks
```python
# Use with Weights & Biases
import wandb

wandb.init(project="graphmamba-evaluation")
for metric_name, values in results.items():
    wandb.log({f"eval/{metric_name}": values})
```

## 🤝 Contributing

To add new evaluation metrics:

1. Add method to `ContagionEvaluator` class
2. Update `run_comprehensive_evaluation()` method
3. Add visualization in `ContagionVisualizer` class
4. Update summary report generation
5. Add tests and documentation

## 📄 Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@misc{graphmamba_evaluation_2024,
  title={Comprehensive Evaluation Framework for GraphMamba Contagion Models},
  author={AI Assistant},
  year={2024},
  note={Implementation of TGIB-style evaluation strategies}
}
```

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Ensure all dependencies are properly installed
4. Verify dataset format matches expected structure

---

**Happy Evaluating! 🎉**

This framework gives you comprehensive insights into how well your GraphMamba model explains contagion dynamics, following the rigorous evaluation standards established by TGIB.
