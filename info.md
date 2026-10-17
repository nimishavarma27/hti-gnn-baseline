# GNN Baseline for Herb-Target Interaction (HTI) Prediction

Baseline GNN track. Heterogeneous graph of herbs, compounds and targets, R-GCN encoder with a DistMult decoder, trained on compound-target links and evaluated on both compound-target (CT) and herb-target (HT) prediction.

## Pipeline

```
6 data files (IMPPAT3)
   |
01_prep   -> prep/      index maps, features, two splits, negatives
02_model                model classes + sanity check
03_train  -> ckpt/      one edge hiding model, one compound disjoint model
04_eval   -> results/   first test numbers on the fixed split
05_cv     -> results/   5 fold cross validation, numbers to report
06_check  -> results/   are the predictions sensible, new candidates
07_tune   -> results/   tuning, plant_part relations, evidence weights
```

Run in order: 01, 02, 03, 04, then 05, 06, 07 (these three only need prep/ and ckpt/).

## Data

Files are read from the parent folder (`Dokumenty/`), not from the repo.

| File | Rows | Used for |
|---|---|---|
| compound_features_ecfp4.csv | 2,319 | compound features, 2048 ECFP4 bits |
| target_features_aac.csv | 1,341 | target features, 20 amino acid composition values |
| herb_compound_edges.csv (tab separated) | 8,396 (5,005 unique pairs) | herb-compound edges, plant_part |
| compound_target_labeled.csv | 14,166 | known compound-target positives, evidence_source |
| compound_target_unlabeled.csv | 3,095,613 | every other compound-target pair, negative pool |
| herb_compound_target.csv | 126,830 | herb-target pairs for evaluation (24,835 unique) |

Graph:
- Nodes: 62 herbs, 2,319 compounds, 1,341 targets (3,722 total)
- Node index layout: herbs 0-61, compounds 62-2380, targets 2381-3721
- Relations: herb->compound, compound->herb, compound->target, target->compound
- Herbs have no raw features, they get a learned 128 dim embedding

Only 726 of the 2,319 compounds have any known target. No confirmed negatives exist, so this is positive-unlabeled data.

Columns dropped on purpose: IDs and names (uniprot, ensembl, hgnc, botanical names), label columns (always 1, using them would leak), has_ecfp4 / has_aac / fully_featurized (always True), n_supporting_records (1 for every pair except one). plant_part and evidence_source are tried in 07.

## Model

- Input projection: Linear(2048 -> 128) for compounds, Linear(20 -> 128) for targets, Embedding(62, 128) for herbs
- 2 relational graph conv layers. Each relation has its own weight matrix, messages are averaged over neighbours, plus a self weight
- `h' = W_self h + sum over relations r of mean over neighbours u (W_r h_u) + b`
- ReLU and dropout 0.2 between layers
- Decoder: DistMult, `score(c, t) = sum(z_c * r * z_t)`
- Loss: BPR, `-log sigmoid(score(c, t_pos) - score(c, t_neg))`, one random target as negative per positive
- Adam, lr 1e-3, weight decay 1e-5, batch 1024, max 300 epochs, early stop after 20 checks (every 2 epochs) without val PR-AUC improvement
- Graph stored as edge lists, not a dense adjacency matrix. Same math as a row normalised adjacency per relation
- Only training C-T edges are in the message passing graph

## Splits

Edge hiding (transductive): all nodes stay, C-T positives split 70/15/15.
- train 9,916 / val 2,124 / test 2,126 edges
- val negatives 1:10, test negatives 1:50, drawn from the unlabeled pool

Compound disjoint (inductive): the 726 labeled compounds split 70/15/15.
- 508 / 108 / 110 compounds, 9,750 / 2,172 / 2,244 edges
- val/test compounds have no C-T edges during training, only their herb edges and fingerprints
- negatives drawn only from val/test compounds

## Metrics

No ROC-AUC or accuracy (too optimistic with a huge unlabeled pool).
- PR-AUC on positives vs sampled negatives
- Hits@10, Hits@50: share of held-out true targets ranked in the top 10 / 50 of 1,341
- MRR: mean of 1/rank
- Filtered ranking (05 onwards): when ranking a test pair, the compound's other known targets are taken out of the list
- HT score: `score(h, t) = max over compounds of h of score(c, t)`

---

## 01_prep.ipynb

Reads the 6 data files, writes `prep/`.

| Cell | What it does |
|---|---|
| setup | paths, output folder, seed 42 |
| load csvs | reads the 6 files, keeps only the ID columns needed, drops duplicate herb-compound rows |
| integer index | builds h2i, c2i, t2i maps, prints 62 2319 1341 |
| node features | puts ECFP4 (2319 x 2048) and AAC (1341 x 20) in index order |
| edge arrays | herb-compound edges (2 x 5005), compound-target positives (2 x 14166). Row 0 is source, row 1 is destination |
| split 1: edge hiding | random 70/15/15 split of C-T positives |
| edge hiding negatives | maps the unlabeled pool to indices, samples 21,240 val and 106,300 test negatives |
| herb-target pairs | unique (herb, target) pairs from herb_compound_target.csv, 24,835 |
| split 2: compound disjoint | shuffles the 726 labeled compounds, splits 70/15/15, assigns each C-T edge by its compound |
| compound disjoint negatives | samples negatives only from val/test compounds (21,720 val, 112,200 test) |
| save | writes all arrays as .npy plus meta.json and id_maps.json |

## 02_model.ipynb

Model definition only. The same classes are copied into the later notebooks.

| Cell | What it does |
|---|---|
| imports | torch |
| RelConv | one graph conv layer: per relation weights, mean aggregation with index_add_, self weight, bias |
| Encoder | projects each node type to 128 dims, stacks them into one 3722 x 128 tensor, runs 2 RelConv layers |
| DistMult | learned relation vector, scores a compound-target pair |
| Model | wraps encoder and decoder, `score_ct` looks up compound and target rows with the index offsets |
| sanity check | random features and edges, prints Z shape (3722, 128) |

## 03_train.ipynb

Reads `prep/`, writes `ckpt/best.pt`, `ckpt/best_cd.pt`, `ckpt/train_log.csv`, `ckpt/train_log_cd.csv`.

| Cell | What it does |
|---|---|
| setup | paths, picks cuda if available |
| load prep data | features and edge hiding arrays |
| model classes | same as 02 |
| graph edges | builds the 4 relation edge lists with absolute node indices, C-T from train only |
| model + optimizer | model, Adam, val labels |
| negative sampling + val pr-auc | random target sampler for BPR, val PR-AUC function |
| training loop | BPR loss, val check every 2 epochs, saves best weights, early stop |
| compound disjoint data and graph | loads CD arrays, graph uses only train compounds' C-T edges |
| compound disjoint training loop | same loop for the CD model |

Last run on RTX 4060: edge hiding best val PR-AUC 0.380 (stopped at epoch 106, about 20 s), compound disjoint 0.308 (stopped at 114, about 21 s).

## 04_eval.ipynb

Reads `prep/` and `ckpt/`, writes `results/metrics_ct.csv`, `metrics_ct_cd.csv`, `metrics_ht.csv`, `summary.csv`, `training_curves.png`.

| Cell | What it does |
|---|---|
| setup, load, model classes | as above, plus a function scoring one compound against all targets |
| edge hiding model + embeddings | loads best.pt, computes node embeddings once |
| metric helpers | Hits@K, MRR, PR-AUC |
| edge hiding: CT eval | PR-AUC on test pos vs neg, raw rank of each test positive among all 1,341 targets |
| edge hiding: HT eval | herb score = max over its compounds, ranks all 24,835 herb-target pairs |
| compound disjoint model + embeddings | loads best_cd.pt |
| compound disjoint: CT eval | same CT eval on the CD test set |
| training curves | loss and val PR-AUC for both models |
| summary table | one table with all numbers |

Fixed split result: EH CT PR-AUC 0.133, Hits@10 0.074, Hits@50 0.323, MRR 0.040. CD CT PR-AUC 0.053, Hits@10 0.053, Hits@50 0.209, MRR 0.029.

The HT numbers in 04 should not be reported. About 84% of those herb-target pairs can be reached through training edges via another compound, and the ranking is raw (a herb's other true targets count against it). 05 fixes both.

## 05_cv.ipynb

Reads `prep/`, writes `results/cv_folds.csv`, `cv_summary.csv`, `cv_metrics.png`. These are the numbers to report.

| Cell | What it does |
|---|---|
| setup | paths, K = 5, training settings |
| load data | puts all 14,166 positives back together. Unlabeled pool = every non-positive pair (same 3,095,613 as the csv). Known-positive mask for filtering |
| herb-target setup | herb-target positives (24,835) and negatives (58,307), list of compounds per herb |
| model classes | same model, plus `scores_all` for the full 2319 x 1341 score matrix |
| helpers | edge hiding split, compound disjoint split, negative sampling, edge list builder, metrics |
| train | trains one model with early stopping, returns the best weights and embeddings |
| eval | CT: PR-AUC plus filtered and raw ranking. HT: only pairs reachable through test edges and not through train/val edges (about 1,860 per fold), filtered ranking, PR-AUC against all non-positive herb-target pairs |
| cv loop | 5 folds, each with a new split and new negatives, trains EH and CD models (10 in total) |
| mean and std | summary over folds |
| negative redraw | same model, 10 different 1:50 negative draws: test PR-AUC 0.272 +/- 0.004, so the random draw barely matters |
| plot | bar chart with std |

## 06_check.ipynb

Reads `prep/`, `ckpt/best.pt` and name columns from the csvs, writes `results/check_score_dist.png`, `novel_ct_top200.csv`, `novel_ht_top200.csv`.

| Cell | What it does |
|---|---|
| load data + names | protein names, gene symbols, herb botanical names |
| score every pair | full score matrix, each pair tagged as train / val / test / unlabeled |
| check 1 | score histograms. 40.7% of test positives score above the 95th percentile of negatives (random 5%) |
| check 2 | precision@10 per test compound, train/val targets removed: 0.121 vs 0.004 random. 43.4% of test compounds get at least one hit in their top 10 |
| check 3 | top 10 predicted targets for 3 test compounds with hit labels |
| check 4 | herb precision@10 on test-only pairs: 0.213 vs 0.034 random |
| check 5 | top 10 predicted targets for 3 herbs |
| new predictions | top 200 unlabeled CT and HT pairs, candidates for literature checking |

## 07_tune.ipynb

Reads `prep/`, `herb_compound_edges.csv`, `compound_target_labeled.csv`, writes `results/tune_stage1.csv`, `best_config.json`, `tune_cv_folds.csv`, `tune_compare.csv`.

| Cell | What it does |
|---|---|
| plant_part relations | 13 plant parts grouped into 6 (leaf, underground, flower_fruit_seed, stem_bark, whole, other), each group is its own herb-compound relation, 14 relations in total |
| evidence weights | number of databases reporting a pair (BindingDB, ChEMBL, NPASS): 8,932 pairs from 1, 3,882 from 2, 1,352 from 3. Used as a loss weight |
| model, helpers, train, eval | same as 05 but configurable: lr, hidden dim, dropout, weight decay, negatives per positive, plant_part, evidence weight |
| stage 1 | one change at a time, 3 edge hiding folds, picked on val PR-AUC only |
| stage 2 | combines changes that gained at least 0.01 |
| stage 3 | 5 fold CV with the best config on both splits, compared to 05 |

Stage 1 (mean val PR-AUC over 3 folds):

| Trial | Val PR-AUC |
|---|---|
| baseline | 0.503 |
| plant_part relations | 0.495 |
| evidence weight | 0.488 |
| weight decay 1e-4 | 0.482 |
| dropout 0.4 | 0.471 |
| 10 negatives | 0.468 |
| 5 negatives | 0.452 |
| lr 5e-4 | 0.429 |
| hidden dim 64 | 0.413 |

Nothing beat the baseline, so the final config is the baseline. Stage 3 retrained the same config and moved CT PR-AUC from 0.175 to 0.186, which shows the run to run noise on GPU is about 0.01.

---

## Results (05_cv, 5 folds, mean +/- std)

| Metric | Edge hiding | Compound disjoint |
|---|---|---|
| CT PR-AUC | 0.175 +/- 0.071 | 0.077 +/- 0.016 |
| CT Hits@10 | 0.240 +/- 0.083 | 0.111 +/- 0.018 |
| CT Hits@50 | 0.514 +/- 0.071 | 0.300 +/- 0.024 |
| CT MRR | 0.126 +/- 0.050 | 0.056 +/- 0.007 |
| HT PR-AUC | 0.068 +/- 0.007 | 0.075 +/- 0.027 |
| HT Hits@10 | 0.085 +/- 0.020 | 0.064 +/- 0.011 |
| HT Hits@50 | 0.242 +/- 0.041 | 0.195 +/- 0.034 |
| HT MRR | 0.044 +/- 0.011 | 0.035 +/- 0.009 |

Popularity baseline (score every pair by how many training compounds hit that target, same folds):

| Metric | EH popularity | EH R-GCN | CD popularity | CD R-GCN |
|---|---|---|---|---|
| CT Hits@10 | 0.160 | 0.240 | 0.137 | 0.111 |
| CT Hits@50 | 0.432 | 0.514 | 0.381 | 0.300 |
| CT MRR | 0.070 | 0.126 | 0.059 | 0.056 |
| HT Hits@10 | 0.138 | 0.085 | 0.125 | 0.064 |
| HT Hits@50 | 0.305 | 0.242 | 0.291 | 0.195 |

Summary:
- For compounds the model has seen (edge hiding), R-GCN clearly beats popularity on CT ranking, so it learns compound specific signal
- For unseen compounds and at herb level, it does not beat popularity yet
- Tuning, plant_part and evidence weights did not help

## Known issues

- 04 HT numbers are not valid, use 05
- Edge hiding PR-AUC is inflated by a shortcut: about 71% of EH negatives come from the 1,593 compounds that have no labels at all, and a score that only checks whether a compound has training edges gets PR-AUC 0.303. Hits@K and MRR are not affected (they rank within one compound). Early stopping uses this PR-AUC
- Compound disjoint is slightly optimistic: 5-13% of test compounds have an identical ECFP4 fingerprint to a training compound
- Training edges are also message passing edges, so the model sees the pair it is predicting during training. The model overfits after about 20-40 epochs (train positives score 1.97 on average, test positives 0.79)
- The model leans on popular targets (TYR, NFE2L2, CBX1, POLI appear in many top 10 lists)
- GPU runs are not fully deterministic, differences under about 0.01 are noise

## Next steps

- Draw edge hiding negatives only from labeled compounds, early stop on filtered val MRR
- Degree aware negative sampling so ranking popular targets high stops paying off
- Separate message passing edges from supervision edges during training
- Compare HT aggregation: max, mean, top-3 mean, noisy-OR
- Better features: ESM-2 protein embeddings (public, from UniProt sequences) instead of AAC, herb features from the mean of their compounds' fingerprints
- Scaffold split or dedup of identical fingerprints for compound disjoint
- Other models: HAN, HGT, KG embeddings (TransE, RotatE, ComplEx), MAMGN-HTI

## Other files

- `cdmmm_scraper.ipynb`, `dataset_visualization.ipynb`, `interactive_visualization.ipynb`: earlier dataset work, not part of the GNN pipeline
- `hti_out/figures/`: dataset figures

## Requirements

```
torch (CUDA build), numpy, pandas, scikit-learn, matplotlib, jupyter
```

On Windows, if torch fails with `OMP: Error #15`, set `KMP_DUPLICATE_LIB_OK=TRUE` before starting Jupyter.
