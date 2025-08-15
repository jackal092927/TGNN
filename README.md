# Self-Explainable Temporal Graph Networks based on Graph Information Bottleneck

The official source code for [KDD24] Self-Explainable Temporal Graph Networks based on Graph Information Bottleneck.
 
Overview of Self-Explainable Temporal Graph Networks based on Graph Information Bottleneck.
![architecture2_page-0001](./architecture.PNG)

Temporal Graph Neural Networks (TGNN) have the ability to capture both the graph topology and dynamic dependencies of interactions within a graph over time. There has been a growing need to explain the predictions of TGNN models due to the difficulty in identifying how past events influence their predictions. Since the explanation model for a static graph cannot be readily applied to temporal graphs due to its inability to capture temporal dependencies, recent studies proposed explanation models for temporal graphs. However, existing explanation models for temporal graphs rely on post-hoc explanations, requiring separate models for prediction and explanation, which is limited in two aspects: efficiency and accuracy of explanation. In this work, we propose a novel built-in explanation framework for temporal graphs, called Self-Explainable
Temporal Graph Networks based on Graph Information Bottleneck (TGIB). TGIB provides explanations for event occurrences by introducing stochasticity in each temporal event based on the Information Bottleneck theory. Experimental results demonstrate the superiority of TGIB in terms of both the link prediction performance and explainability compared to state-of-the-art methods. This is the first work that simultaneously performs prediction and explanation for temporal graphs in an end-to-end manner.

## Requirements
```
pandas==0.24.2
torch==1.1.0
tqdm==4.41.1
numpy==1.16.4
scikit_learn==0.22.1
```
## Dataset
* Download the datasets at this link https://zenodo.org/records/7213796#.Y1cO6y8r30o.
e.g.: 
wget https://zenodo.org/records/7213796/files/wikipedia.zip
wget https://zenodo.org/records/7213796/files/uci.zip

## Run
```
python -u learn_edge.py -d wikipedia --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --n_head 2 --n_epoch 100 --prefix hello_world --gpu 1

//Retuls:
INFO:root:epoch: 6:
INFO:root:Epoch mean loss: 0.48878291973948124
INFO:root:train acc: 0.9006504463313738, val acc: 0.8867744803352948, new node val acc: 0.882607876389674
INFO:root:train auc: 0.9644895878692207, val auc: 0.9531607669527319, new node val auc: 0.9478128516559857
INFO:root:train ap: 0.9639445810851244, val ap: 0.9526156915222724, new node val ap: 0.9480556209311074
INFO:root:start 7 epoch
100%|███████████████████████████████████████████████████████| 552/552 [06:50<00:00,  1.34it/s]
INFO:root:epoch: 7:
INFO:root:Epoch mean loss: 0.46904547060219015
INFO:root:train acc: 0.9050774729661079, val acc: 0.8913043478260869, new node val acc: 0.8882607876389674
INFO:root:train auc: 0.9668899911096596, val auc: 0.9554451043361336, new node val auc: 0.9515643419289284
INFO:root:train ap: 0.9666669310236945, val ap: 0.9560650044683446, new node val ap: 0.951710981020965
INFO:root:start 8 epoch
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 552/552 [06:50<00:00,  1.34it/s]
INFO:root:epoch: 8:
INFO:root:Epoch mean loss: 0.4615722468210842
INFO:root:train acc: 0.9071004789897671, val acc: 0.8936327843867745, new node val acc: 0.8850574712643678
INFO:root:train auc: 0.9682393837542633, val auc: 0.9565579659745868, new node val auc: 0.9477297485775267
INFO:root:train ap: 0.967988221599129, val ap: 0.9570361132805842, new node val ap: 0.9490333461762808
INFO:root:start 9 epoch
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 552/552 [06:51<00:00,  1.34it/s]
INFO:root:epoch: 9:
INFO:root:Epoch mean loss: 0.4458365629745011
INFO:root:train acc: 0.909558930256187, val acc: 0.8951356843486727, new node val acc: 0.8908045977011494
INFO:root:train auc: 0.9703602288809057, val auc: 0.9583199322233443, new node val auc: 0.9555592772022139
INFO:root:train ap: 0.9704891144042713, val ap: 0.9584082181589387, new node val ap: 0.9551119169463808
INFO:root:start 10 epoch
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 552/552 [06:50<00:00,  1.34it/s]
INFO:root:epoch: 10:
INFO:root:Epoch mean loss: 0.43658056778743914
INFO:root:train acc: 0.9127068364903114, val acc: 0.8941619745142034, new node val acc: 0.8836442434520445
INFO:root:train auc: 0.9711504485993177, val auc: 0.9572135871833175, new node val auc: 0.9514134655515817
INFO:root:train ap: 0.9708501543594792, val ap: 0.9578743513138483, new node val ap: 0.9535931665176708
INFO:root:start 11 epoch
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 552/552 [06:50<00:00,  1.34it/s]
INFO:root:epoch: 11:
INFO:root:Epoch mean loss: 0.42512875321315174
INFO:root:train acc: 0.9142263589520284, val acc: 0.8961305617882392, new node val acc: 0.8873186357640852
INFO:root:train auc: 0.9727483625444522, val auc: 0.9608557457952068, new node val auc: 0.9526109405845565
INFO:root:train ap: 0.9725270783049703, val ap: 0.9615459964126714, new node val ap: 0.9526673782387128
INFO:root:start 12 epoch
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 552/552 [06:51<00:00,  1.34it/s]
INFO:root:epoch: 12:
INFO:root:Epoch mean loss: 0.4191080166844924
INFO:root:train acc: 0.9156324842151099, val acc: 0.897294780068583, new node val acc: 0.8898624458262672
INFO:root:train auc: 0.9734891184048188, val auc: 0.9593722897905617, new node val auc: 0.9537066103840987
INFO:root:train ap: 0.9732128721851341, val ap: 0.960159859480736, new node val ap: 0.9563012096870265
INFO:root:start 13 epoch
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 552/552 [06:52<00:00,  1.34it/s]
INFO:root:epoch: 13:
INFO:root:Epoch mean loss: 0.4112673603959116
INFO:root:train acc: 0.9171973655562813, val acc: 0.8980356462469836, new node val acc: 0.8915583192010552
INFO:root:train auc: 0.9741389977501996, val auc: 0.9616071957761556, new node val auc: 0.9579257073623444
INFO:root:train ap: 0.9738046654096006, val ap: 0.9615391295146395, new node val ap: 0.9592146461998059



python -u learn_edge.py -d wikipedia --bs 200 --uniform --lr 0.00001 --n_degree 20 --setting inductive --agg_method attn --attn_mode prod --n_head 2 --n_epoch 10 --prefix hello_world_learn_edge --gpu 1 > hello_world_learn_edge.log 2>&1

python -u learn_edge.py -d uci --bs 32 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 1 --prefix helloworld_test

python -u learn_edge.py -d CanParl --bs 32 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 1 --prefix helloworld_test

python -u -m cProfile -o profile.out learn_edge.py -d CanParl --bs 32 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 1 --prefix helloworld_test
```

<!-- 
pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_sparse-0.6.17+pt113cu117-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_scatter-2.1.0+pt113cu117-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_cluster-1.6.0+pt113cu117-cp310-cp310-linux_x86_64.whl 
-->

```
python -u learn_edge.py -d wikipedia --bs 64 --lr 0.0001 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix test_fix --n_epoch 10 > test_fix_results.log 2>&1

python -u learn_edge.py -d wikipedia --bs 64 --lr 0.001 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix test_fix --n_epoch 10

```

# TGIB (Temporal Graph Information Bottleneck) - Enhanced Version

## Training Modes

This enhanced version provides two training modes to balance safety and performance:

### 1. Sequential Mode (Default - Safer)
Preserves the exact original training logic with information bottleneck loss:
```bash
python -u learn_edge.py -d wikipedia --bs 20 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix safe_run --training_mode sequential
```

**Characteristics:**
- ✅ Exact original logic preservation
- ✅ Includes missing information bottleneck loss
- ✅ Individual edge processing (safer for debugging)
- ⚠️ Slower due to per-edge optimizer steps

### 2. Batch Mode (Optimized - Faster)
Optimized version with vectorized operations:
```bash
python -u learn_edge.py -d wikipedia --bs 20 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix fast_run --training_mode batch
```

**Characteristics:**
- ✅ Major speed improvements (batch processing)
- ✅ Includes missing information bottleneck loss
- ✅ Vectorized computations
- ⚠️ Potential subtle differences from batching

## Debugging Strategy

1. **First run**: Use `--training_mode sequential` to establish baseline with exact original logic
2. **Speed optimization**: Use `--training_mode batch` for faster training
3. **Performance comparison**: Compare results to ensure batch mode maintains accuracy

## Key Fixes Applied

- ✅ **Information Bottleneck Loss**: Added missing critical component from original model
- ✅ **Vectorized Operations**: Batch processing for significant speed improvements  
- ✅ **Gradient Accumulation**: Proper batching of optimizer steps
- ✅ **Flexible Switching**: Command-line option to switch between modes

## Usage Examples

```bash
# Safe mode (matches original exactly)
python -u learn_edge.py -d wikipedia --training_mode sequential --prefix safe

# Fast mode (optimized batching)  
python -u learn_edge.py -d wikipedia --training_mode batch --prefix fast

# Compare results
python -u learn_edge.py -d wikipedia --training_mode sequential --prefix compare_seq --n_epoch 1 --gpu 0
python -u learn_edge.py -d wikipedia --training_mode batch --prefix compare_batch --n_epoch 1

python -u learn_edge.py -d wikipedia --training_mode complex_batch --prefix test_complex_batch --n_epoch 2 --gpu 1 --bs 64

## hybrid mode
python -u learn_edge.py -d wikipedia --training_mode hybrid --prefix test_hybrid --n_epoch 1 --gpu 1 --bs 200
```