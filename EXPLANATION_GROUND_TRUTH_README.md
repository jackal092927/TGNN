# Ground Truth Explanations in Synthetic Datasets

This document explains how the ground truth explanations in `synthetic_*_explanations.json` files are constructed and what each entry represents.

## Overview

The ground truth explanations are generated during the synthetic contagion simulation process. Each entry in the JSON file maps an **edge index** to the **nodes that caused that edge to exist** according to the specific contagion model.

## File Structure

```json
{
  "0": [22, 28, 55, 116],
  "1": [22, 28, 55, 116], 
  "2": [22, 28, 55, 116],
  "3": [22, 28, 55, 116],
  "4": [7, 9, 28, 149],
  ...
}
```

**Key**: Edge index (string) from the CSV file's `idx` column
**Value**: List of node IDs that caused this edge to be formed

## Construction Process

### 1. Contagion Simulation

Each synthetic dataset follows this process:

1. **Initialize**: Start with seed nodes (initially activated)
2. **Simulate**: Run contagion model for multiple timesteps
3. **Record**: Track which nodes activate others and why
4. **Convert**: Transform activation events into temporal edges

### 2. Edge Creation Logic

For each activation event during simulation:

```python
# When node X gets activated by neighbors [A, B, C] at time T
activation = (node=X, timestep=T, explanation=[A, B, C])

# This creates edges in the dataset:
# Edge from A to X at time T
# Edge from B to X at time T  
# Edge from C to X at time T
```

### 3. Ground Truth Mapping

The key insight is that **multiple edges can have the same explanation**:

```python
# All edges created from the same activation event share the same explanation
explanations_dict[edge_idx_A_to_X] = [A, B, C]  # Edge index 10
explanations_dict[edge_idx_B_to_X] = [A, B, C]  # Edge index 11  
explanations_dict[edge_idx_C_to_X] = [A, B, C]  # Edge index 12
```

## Model-Specific Explanation Logic

### Linear Threshold Model (LTM)

**Rule**: Node activates when cumulative influence from active neighbors exceeds threshold

**Explanation**: ALL active neighbors that contributed to threshold

```python
# Example: Node 1 has threshold 0.6
# Active neighbors: [22, 28, 55, 116] with weights [0.2, 0.15, 0.25, 0.1]
# Total influence: 0.2 + 0.15 + 0.25 + 0.1 = 0.7 > 0.6 ✓
# Ground truth: [22, 28, 55, 116] (all contributors)
```

**In the dataset**:
- Edge `22 → 1` gets explanation `[22, 28, 55, 116]`
- Edge `28 → 1` gets explanation `[22, 28, 55, 116]`
- Edge `55 → 1` gets explanation `[22, 28, 55, 116]`
- Edge `116 → 1` gets explanation `[22, 28, 55, 116]`

### Independent Cascade Model (ICM)

**Rule**: Each newly active neighbor tries to activate others with probability

**Explanation**: SINGLE successful activator

```python
# Example: Node 5 gets activated by node 7 (successful coin flip)
# Other neighbors [9, 22] either weren't active or failed to activate
# Ground truth: [7] (only the successful activator)
```

### Complex Contagion (CC)

**Rule**: Node needs ≥k active neighbors to activate

**Explanation**: ALL neighbors in the activating quorum

```python
# Example: Node X needs k=2 neighbors, has active neighbors [A, B, C, D]
# Ground truth: [A, B, C, D] (all members of sufficient quorum)
```

### Structural Diversity (SD)

**Rule**: Activation depends on structural diversity of active neighbors

**Explanation**: Nodes in the diverse subgraph structure

## Example Analysis

Let's trace through a specific example from `synthetic_ltm_ba`:

```csv
u,i,ts,label,idx
22,1,1.0,1,1
28,1,1.0,1,2  
55,1,1.0,1,3
116,1,1.0,1,4
```

```json
{
  "1": [22, 28, 55, 116],
  "2": [22, 28, 55, 116],
  "3": [22, 28, 55, 116], 
  "4": [22, 28, 55, 116]
}
```

**Interpretation**:
- At time `1.0`, node `1` was activated by neighbors `[22, 28, 55, 116]`
- This single activation event created 4 edges in the dataset:
  - Edge 1: `22 → 1` (from activating neighbor to activated node)
  - Edge 2: `28 → 1` 
  - Edge 3: `55 → 1`
  - Edge 4: `116 → 1`
- All 4 edges share the same ground truth explanation: `[22, 28, 55, 116]`

## Usage in Evaluation

When evaluating explanations, the system:

1. **Looks up edge index**: For edge with `idx=1`, find explanation `[22, 28, 55, 116]`
2. **Compares with model**: Check if TGNN attention weights highlight nodes `[22, 28, 55, 116]`
3. **Calculates metrics**: Precision@k, Recall@k, MRR based on overlap

## Key Insights

### Why Multiple Edges Share Explanations

This reflects the **collective nature** of many contagion processes:
- In LTM, node activation is a **group effort** by multiple neighbors
- Each individual edge (neighbor → activated node) represents part of this collective influence
- The explanation captures the **complete causal set**, not just the individual edge

### Explanation vs. Edge Semantics

- **Edge semantics**: "Did neighbor A influence node B at time T?"
- **Explanation semantics**: "Which nodes collectively caused B to activate at time T?"

### Ground Truth Granularity

The ground truth captures **activation-level causality**, not **edge-level causality**:
- ✅ "Nodes [A,B,C] collectively caused X to activate"
- ❌ "Edge A→X individually caused X to activate"

## Non-Causal Edges

Approximately 50% of edges in the dataset are non-causal (label=0):

```json
{
  "150": [],  # Empty explanation = non-causal edge
  "151": [],
  ...
}
```

These represent:
- Random edges added as negative examples
- Edges that don't correspond to actual activation events
- Used to test if the model can distinguish causal from non-causal relationships

## Validation

To verify ground truth correctness:

```python
# Check that all nodes in explanation were active before the activation
for edge_idx, explanation in explanations.items():
    edge = dataset[dataset.idx == int(edge_idx)].iloc[0]
    activation_time = edge.ts
    activated_node = edge.i
    
    # All explanation nodes should have been active before activation_time
    for explaining_node in explanation:
        assert was_active_before(explaining_node, activation_time)
```

This ensures temporal consistency and causal validity of the ground truth explanations. 