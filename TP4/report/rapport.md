# TP4 – Graph Neural Networks

## Dépôt
Lien : [https://github.com/aramz33/CSC8608](https://github.com/aramz33/CSC8608)

## Environnement d'exécution
- Machine : MacBook M3, Apple Silicon
- Device : MPS (Metal Performance Shaders)
- torch : 2.11.0
- torch-geometric : 2.7.0
- Python : 3.14

## Arborescence TP4/
```
TP4/
├── configs/
│   ├── baseline_mlp.yaml
│   ├── gcn.yaml
│   └── sage_sampling.yaml
├── rapport.md
├── requirements.txt
├── runs/               ← gitignored
└── src/
    ├── benchmark.py
    ├── data.py
    ├── models.py
    ├── smoke_test.py
    ├── train.py
    └── utils.py
```

---

## Exercice 1 – Initialisation du TP et smoke test PyG (Cora)

### Q1 – Structure du projet
Voir arborescence ci-dessus.

### Q2–Q3 – Installation des dépendances
```
pip install torch-geometric scipy torch-scatter torch-sparse \
  --find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

### Q4–Q5 – Smoke test

```
torch version  : 2.11.0
device         : mps
gpu name       : Apple MPS (M-series)

--- Cora dataset ---
num_nodes      : 2708
num_edges      : 10556
num_features   : 1433
num_classes    : 7
train nodes    : 140
val nodes      : 500
test nodes     : 1000
edge_index     : torch.Size([2, 10556])
x              : torch.Size([2708, 1433])
y              : torch.Size([2708])

Smoke test passed.
```

Cora est un graphe de citations : 2708 nœuds (articles), 10556 arêtes dirigées, 1433 features binaires (bag-of-words), 7 classes thématiques. Le split standard est très petit (140 nœuds d'entraînement) — typique des benchmarks GNN transductifs.

---

## Exercice 2 – Baseline tabulaire : MLP

### Q1–Q4 – Implémentation

`data.py` expose un `CoraData` dataclass regroupant `x`, `y`, `edge_index`, les masques, et le `pyg_data` complet. `utils.py` fournit `set_seed`, `Timer`, `accuracy`, et `macro_f1` (calcul manuel par classe, sans sklearn). `models.py` implémente `MLP` : deux couches linéaires avec ReLU et Dropout, retournant des logits bruts (CrossEntropyLoss inclut le softmax).

### Q5 – Config MLP
```yaml
seed: 42
device: "cuda"
epochs: 200
lr: 0.01
weight_decay: 0.0005
mlp:
  hidden_dim: 256
  dropout: 0.5
```

### Q6–Q7 – Entraînement

```
model          : mlp
config         : TP4/configs/baseline_mlp.yaml
device         : mps
epochs         : 200
epoch    1 | loss 1.9492 | train_acc 0.9286 | val_acc 0.4280 | epoch_time 1905.5ms
epoch   50 | loss 0.0017 | train_acc 1.0000 | val_acc 0.5660 | epoch_time 6.2ms
epoch  100 | loss 0.0048 | train_acc 1.0000 | val_acc 0.5520 | epoch_time 6.2ms
epoch  150 | loss 0.0047 | train_acc 1.0000 | val_acc 0.5420 | epoch_time 6.2ms
epoch  200 | loss 0.0043 | train_acc 1.0000 | val_acc 0.5380 | epoch_time 6.3ms

--- Final metrics ---
test_acc       : 0.5710
test_macro_f1  : 0.5580
val_acc        : 0.5380
train_loop_time: 3.28s
total_train_time_s: 3.28s
```

Les métriques sont calculées séparément sur chaque masque (train/val/test) pour éviter la fuite d'information : le modèle est évalué uniquement sur les nœuds non vus pendant l'optimisation. Utiliser le masque d'entraînement pour évaluer le test biaiserait les résultats à la hausse.

---

## Exercice 3 – Baseline GNN : GCN (full-batch)

### Q1 – Config GCN
```yaml
seed: 42
device: "cuda"
epochs: 200
lr: 0.01
weight_decay: 0.0005
gcn:
  hidden_dim: 256
  dropout: 0.5
```
Hyperparamètres identiques au MLP pour isoler l'effet du graphe.

### Q2–Q4 – Implémentation
`GCN` : deux `GCNConv` avec ReLU et Dropout. `train.py` dispatche via `--model [mlp|gcn]` : MLP reçoit uniquement `x`, GCN reçoit `(x, edge_index)`.

### Q5 – Entraînement et comparaison

```
model          : gcn
config         : TP4/configs/gcn.yaml
device         : mps
epochs         : 200
epoch    1 | loss 1.9437 | train_acc 0.9500 | val_acc 0.7120 | epoch_time 936.8ms
epoch   50 | loss 0.0136 | train_acc 1.0000 | val_acc 0.7640 | epoch_time 26.8ms
epoch  100 | loss 0.0101 | train_acc 1.0000 | val_acc 0.7620 | epoch_time 22.1ms
epoch  150 | loss 0.0071 | train_acc 1.0000 | val_acc 0.7660 | epoch_time 24.7ms
epoch  200 | loss 0.0069 | train_acc 1.0000 | val_acc 0.7760 | epoch_time 23.4ms

--- Final metrics ---
test_acc       : 0.8050
test_macro_f1  : 0.7999
val_acc        : 0.7760
train_loop_time: 6.73s
total_train_time_s: 6.73s
```

| Modèle | test_acc | test_macro_f1 | total_train_time_s |
|--------|----------|---------------|--------------------|
| MLP    | 0.5710   | 0.5580        | 3.28 s             |
| GCN    | 0.8050   | 0.7999        | 6.73 s             |

### Q6 – Pourquoi GCN surpasse MLP sur Cora

Cora présente une forte homophilie : les articles citent massivement des articles de la même classe. Le GCN exploite ce signal structurel en agrégeant les représentations des voisins (propagation de message), ce qui équivaut à un lissage spectral sur le graphe. Le MLP ignore les arêtes et classe chaque nœud uniquement à partir de ses propres features bag-of-words. Sur Cora, ces features sont partiellement discriminantes mais insuffisantes seules (57 % de test accuracy). L'information de voisinage apporte +23 points d'accuracy sans modifier l'architecture au-delà de l'agrégation.

---

## Exercice 4 – GraphSAGE + neighbor sampling (mini-batch)

### Q1 – Config GraphSAGE
```yaml
seed: 42
device: "cuda"
epochs: 50
lr: 0.005
weight_decay: 0.0005
sage:
  hidden_dim: 256
  dropout: 0.5
sampling:
  batch_size: 64
  num_neighbors_l1: 25
  num_neighbors_l2: 10
```

### Q2–Q4 – Implémentation

`GraphSAGE` : deux `SAGEConv`, même signature `forward(x, edge_index)` que GCN. `NeighborLoader` échantillonne 25 voisins au premier hop et 10 au deuxième pour chaque nœud-graine du mini-batch. L'évaluation reste full-batch (Cora étant petit, aucune contrainte mémoire).

### Q5 – Entraînement

```
model          : sage
config         : TP4/configs/sage_sampling.yaml
device         : mps
epochs         : 50
epoch    1 | loss 1.8419 | train_acc 1.0000 | val_acc 0.7340 | epoch_time 926.6ms
epoch   50 | loss 0.0034 | train_acc 1.0000 | val_acc 0.7700 | epoch_time 92.4ms

--- Final metrics ---
test_acc       : 0.7960
test_macro_f1  : 0.7924
val_acc        : 0.7700
train_loop_time: 6.23s
total_train_time_s: 6.23s
```

| Modèle     | test_acc | test_macro_f1 | total_train_time_s |
|------------|----------|---------------|--------------------|
| MLP        | 0.5710   | 0.5580        | 3.28 s             |
| GCN        | 0.8050   | 0.7999        | 6.73 s             |
| GraphSAGE  | 0.7960   | 0.7924        | 6.23 s             |

### Q6 – Compromis du neighbor sampling

Le neighbor sampling découpe le calcul de propagation en sous-graphes induits par un fanout fixe (ici 25-10). Cela réduit la mémoire de O(N²) à O(batch × fanout^L) et rend GraphSAGE scalable à des graphes de milliards de nœuds. Cependant, le gradient estimé à chaque étape est bruité : l'agrégation n'utilise qu'un sous-ensemble des voisins réels, ce qui introduit une variance dans les mises à jour. Sur les hubs (nœuds avec grand degré), les voisins ignorés peuvent porter une information critique, ce qui dégrade légèrement la qualité par rapport au full-batch GCN. Par ailleurs, le sampling est réalisé sur CPU, ce qui peut créer un goulot d'étranglement sur des graphes denses ou des fanouts élevés.

---

## Exercice 5 – Benchmarks ingénieur

### Q1–Q3 – Checkpoint et benchmark

Les checkpoints sont sauvegardés dans `TP4/runs/` (gitignored). `benchmark.py` charge le modèle, effectue des passes de warmup pour stabiliser le pipeline MPS/CUDA, puis mesure 100 passes en synchronisant le device avant et après chaque mesure.

### Q4 – Résultats benchmark

```
--- MLP ---
model          : mlp | device : mps | num_nodes : 2708
avg_forward_ms : 1.256 ms
ms_per_node_approx: 0.4639 µs/node

--- GCN ---
model          : gcn | device : mps | num_nodes : 2708
avg_forward_ms : 12.988 ms
ms_per_node_approx: 4.7961 µs/node

--- GraphSAGE ---
model          : sage | device : mps | num_nodes : 2708
avg_forward_ms : 12.237 ms
ms_per_node_approx: 4.5187 µs/node
```

### Q5 – Warmup et synchronisation

Le warmup est nécessaire pour initialiser le pipeline GPU : les premiers forward passes déclenchent la compilation JIT des kernels et le remplissage des caches. Sans warmup, les premières mesures sont artificiellement lentes. La synchronisation (`torch.mps.synchronize()`) force l'attente de la fin effective des calculs GPU avant de lire l'horloge CPU, car l'exécution GPU est asynchrone. Sans synchronisation, on mesure uniquement la latence de soumission des opérations dans la queue GPU, ce qui sous-estime le temps réel et produit des mesures instables.

---

## Exercice 6 – Synthèse finale

### Tableau de comparaison

| Modèle     | test_acc | test_macro_f1 | total_train_time_s | avg_forward_ms |
|------------|----------|---------------|--------------------|----------------|
| MLP        | 0.5710   | 0.5580        | 3.28 s             | 1.256 ms       |
| GCN        | 0.8050   | 0.7999        | 6.73 s             | 12.988 ms      |
| GraphSAGE  | 0.7960   | 0.7924        | 6.23 s             | 12.237 ms      |

### Recommandation ingénieur

Le **GCN** est le meilleur choix sur Cora : il atteint la meilleure accuracy (80.5 %) et le meilleur Macro-F1 (0.80) avec un temps d'entraînement compétitif (7.8 s). Il est à privilégier quand le graphe tient en mémoire GPU et que la qualité prime.

Le **MLP** est adapté quand la structure du graphe est absente ou non fiable, ou quand la latence d'inférence est critique (1.3 ms vs 11–15 ms pour les GNN). Son accuracy de 57 % illustre la limite d'une approche tabulaire sur un graphe homophile.

**GraphSAGE** avec neighbor sampling est incontournable à grande échelle (millions de nœuds) où le full-batch est impossible. Sur Cora, il perd légèrement en accuracy (−0.9 points vs GCN) en raison du bruit du sampling, mais son entraînement serait stable et scalable sur des graphes de production.