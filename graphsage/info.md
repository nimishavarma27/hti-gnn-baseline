# Hetero-GraphSAGE vs R-GCN

Separate experiment, the main pipeline in the parent folder is not changed.

`sage_cv.ipynb` reads `../prep/` and `../results/cv_folds.csv` (R-GCN numbers from 05_cv) and writes `results/`.

## What is different from R-GCN

Only the graph layer. Data, folds, splits, negatives, BPR loss, DistMult decoder, early stopping and metrics are the same as 05_cv, and the folds use the same random seeds, so the numbers can be compared directly.

| | R-GCN | Hetero-GraphSAGE |
|---|---|---|
| Own vector | one shared self weight | own weight per relation |
| Neighbours | mean, weight per relation | mean or pool, weight per relation |
| Combining relations | summed | summed over relations ending at the same node type |

Two versions:
- sage-mean: neighbours averaged
- sage-pool: each neighbour through a linear layer + ReLU, then element-wise max (the GraphSAGE pool aggregator)

Popularity baseline (score = number of training compounds hitting the target) is computed on the same folds.

## Cells

| Cell | What it does |
|---|---|
| setup | paths, K = 5, the two aggregators |
| load data | same as 05_cv |
| herb-target setup | herb-target positives, negatives, compounds per herb |
| hetero graphsage layer | SAGELayer and Model |
| helpers | splits, negative sampling, edge lists, metrics |
| train | one model, early stop on val PR-AUC |
| eval | CT filtered ranking, HT test-only pairs, popularity scorer |
| cv loop | 5 folds x 2 splits x (sage-mean, sage-pool, popularity), 20 trained models |
| compare | mean +/- std for r-gcn, sage-mean, sage-pool, popularity |
| plot | bar chart per split |

## Outputs

- `results/sage_cv_folds.csv`: every fold and model
- `results/compare.csv`: summary table
- `results/compare.png`: bar chart

## Results

Not run yet.
