# Herb-Target Interaction Prediction: Baseline Model, Before and After

GNN baseline track. This report compares the first version of the baseline (before) with the corrected and extended version (after). All numbers are means ± standard deviation over the same 5 cross-validation folds, computed directly from the result files of each version.

## 1. Summary

- **Before:** an R-GCN graph neural network that predicts compound-target links, with herb-target links derived from them.
- **A data defect was found:** the fingerprint file had wrong compound IDs, so almost every compound was trained with another compound's fingerprint. This affected every result in the before version.
- **After:** the fingerprints were fixed, the R-GCN's overfitting was addressed, and two models that work on new compounds were added (similarity kNN and a two-tower network), plus their ensemble. The two-tower network uses ESM-2 protein embeddings; the R-GCN keeps the original amino acid composition (AAC) features.
- **Result:** for compound-target prediction, the after models beat the popularity baseline on all three splits, including compounds and targets never seen in training. The R-GCN alone never beats popularity on new compounds.
- **Open problem:** herb-target prediction still does not beat the popularity baseline on Hits@10, Hits@50 or MRR, before or after.

| Split and metric | Popularity | R-GCN before | R-GCN after | Best after model |
|---|---|---|---|---|
| Known compounds, CT Hits@10 | 0.160 | 0.240 | 0.268 | 0.463 (ensemble) |
| Known compounds, CT MRR | 0.070 | 0.126 | 0.129 | 0.278 (ensemble) |
| New compounds, CT Hits@10 | 0.137 | 0.111 | 0.116 | 0.233 (ensemble) |
| New compounds, CT MRR | 0.059 | 0.056 | 0.057 | 0.132 (ensemble) |
| New targets, CT Hits@10 | 0.077 | not tested | not tested | 0.446 (two-tower ESM-2) |
| Known compounds, HT Hits@10 | 0.137 | 0.085 | 0.068 | 0.100 (R-GCN + feature reg.) |
| New compounds, HT Hits@10 | 0.125 | 0.064 | 0.054 | 0.083 (kNN) |

## 2. Data

Source: IMPPAT. Six files, all cross-checked for consistent IDs and duplicates.

| Item | Count |
|---|---|
| Herbs | 62 |
| Compounds | 2,319 (only 726 have a known target) |
| Targets (proteins) | 1,341 |
| Herb-compound links | 5,005 |
| Known compound-target links (positives) | 14,166 |
| Unlabeled compound-target pairs | 3,095,613 (every other pair) |
| Herb-target pairs (derived through compounds) | 24,835 |

There are no confirmed negative pairs. Every pair that is not a known link is unlabeled, so this is positive-unlabeled (PU) data.

## 3. How the models are tested (same before and after)

**Splits** (70% train, 15% validation, 15% test):

| Split | What is held out | Question it answers |
|---|---|---|
| Edge hiding | 15% of compound-target links; every compound stays in the graph | Can the model find missing targets of a known compound? |
| Compound disjoint | 15% of the 726 labeled compounds, with all their links | Can it predict targets of a new compound? |
| Target disjoint (after only) | 15% of the targets, with all their links | Can it predict compounds for a new protein? |

**Negatives:** drawn at random from the unlabeled pairs, 1 positive to 10 negatives for validation and 1 to 50 for test (details in section 3.1).

**Metrics:**
- **PR-AUC:** precision-recall area on test positives against test negatives. Random gives about 0.02 at 1:50.
- **Hits@10 / Hits@50:** each hidden true link is ranked against all 1,341 targets of that compound. Share of true links that land in the top 10 / top 50.
- **MRR:** mean of 1/rank of the true target.
- Ranking is **filtered**: the compound's other known targets are removed from the list before ranking.
- ROC-AUC and accuracy are not used, because a huge unlabeled pool makes them look high even for a weak model.

**Herb-target (HT) evaluation:** score(herb, target) = highest score among the herb's compounds. Only herb-target pairs reachable through test links alone are tested (about 1,860 per fold), so pairs the model could already see through another compound are excluded.

**Cross validation:** 5 folds. Each fold re-splits the data, draws new negatives and trains new models. The before and after versions use the same random seeds, so the folds are identical and the numbers compare directly (details in section 3.1).

**Reference baselines:**
- **Random scores:** chance level.
- **Popularity:** score(compound, target) = number of training compounds that hit that target. It ignores the compound completely, so a model is only useful if it beats this.

### 3.1 Cross validation and negative sampling

**How the cross validation works**

- **5 folds, repeated random splits.** For each fold, the data is split again from scratch at random into 70% train, 15% validation and 15% test, a new set of negatives is drawn, and a new model is trained from zero. The 5 results are reported as mean ± standard deviation.
- This is repeated random splitting (also called Monte Carlo cross validation), not classic k-fold. Classic 5-fold cuts the data into five fixed parts and uses each part as the test set once, which cannot give a 70/15/15 split. With repeated random splits every fold keeps 70/15/15; the trade-off is that a link can be in the test set of more than one fold.
- **What is split in each fold:**
  - Edge hiding: the 14,166 known compound-target links are shuffled and cut 70/15/15 (9,916 / 2,125 / 2,125 links).
  - Compound disjoint: the 726 compounds with known targets are shuffled and cut 70/15/15 (508 / 109 / 109 compounds); every link of a validation or test compound goes with it. Link counts vary a little per fold because compounds have different numbers of targets (for example fold 0: 9,448 / 2,669 / 2,049).
  - Target disjoint (after version only): the targets are shuffled and cut 70/15/15 in the same way.
  - The 5,005 herb-compound links are never split; they stay in the graph in every fold.
- **Seeds:** fold k (k = 0 to 4) uses random seed 100 + k for the edge hiding and compound disjoint splits, and 200 + k for target disjoint. Model weights use seed k. Because every notebook (05, 07, 08, 09) and both versions use the same seeds in the same order, all models see exactly the same folds and the same negatives.

**How the 1:10 and 1:50 negatives are taken**

- **Yes, they are random.** Negatives are drawn uniformly at random from the unlabeled pool, that is, every compound-target pair that is not a known link (3,095,613 pairs). Every unlabeled pair has the same chance of being picked; no hard or popularity-based selection is used for validation or test.
- **Without replacement:** a pair is picked at most once within a set.
- **Size:** validation gets 10 negatives per validation positive, test gets 50 per test positive. In edge hiding that is 21,250 validation and 106,250 test negatives per fold.
- **Known links are never used as negatives.** This was checked in every fold: no negative is a known positive.
- **Compound disjoint:** negatives are drawn only from the unlabeled pairs of that fold's validation or test compounds (about 143,000 to 144,000 pairs available per set), so they are about the same new compounds as the positives. Test needs at most 126,200 per fold, so there are always enough pairs and no pair is repeated.
- **Target disjoint:** negatives are drawn only from pairs involving the held-out targets.
- **Reproducible:** the draws are random but fixed by the seed, so running a notebook again gives exactly the same negatives.
- **Fixed split (notebooks 01, 03, 04):** one split with seed 42. There, validation and test negatives are taken in a single draw, so they never share a pair (21,240 and 106,300 negatives in edge hiding; 21,720 and 112,200 in compound disjoint).
- **Cross validation:** validation and test negatives are drawn separately, so a few pairs end up in both sets: 720 to 780 of the 106,250 test negatives per fold (under 1%). This does not leak anything, because negatives are never used for training; validation negatives only decide when to stop.
- **Effect of the random draw:** the same trained model was scored on 10 different random draws of 1:50 test negatives. Test PR-AUC was 0.272 ± 0.004, so which negatives happen to be drawn changes the result very little.

**Negatives during training (different from the above)**

- The validation and test negatives are fixed sets used only to measure the model. They are never used for training.
- R-GCN training (BPR loss): for every known training link in a batch, one target is drawn uniformly at random from all 1,341 targets as its negative, freshly for every batch. Occasionally this hits a true target; that is accepted in positive-unlabeled learning. The "mixed negatives" fix in notebook 08 draws half of these in proportion to target popularity instead.
- Two-tower training: no sampling. Each training link is scored against all training targets at once (softmax), with the compound's other known targets masked out.

**Herb-target negatives are not sampled.** For herb-target evaluation, every herb-target pair that is not a known (derived) pair is used as a negative: 58,307 pairs.

## 4. Before: the original baseline

### Method
- **Graph:** 62 herbs, 2,319 compounds and 1,341 targets (3,722 nodes), with four relations: herb to compound, compound to herb, compound to target, target to compound. Only training compound-target links are used for message passing.
- **Node features:** compounds use ECFP4 fingerprints (2,048 bits), targets use amino acid composition (AAC, 20 values), herbs get a learned 128-number embedding.
- **Graph storage:** the graph is stored as edge lists, not as a dense adjacency matrix. For each relation there are two arrays of node indices, one for the source node and one for the destination node of every link. A dense adjacency matrix would have 3,722 × 3,722 = 13.8 million cells, of which over 99.8% would be zero (the graph has about 30,000 directed links). Message passing over the edge lists gives exactly the same result as multiplying the row-normalised adjacency matrix of each relation by the node features, without building the matrix.
- **Encoder:** R-GCN with 2 layers. In each layer a node's new vector is W_self × (its own vector) + sum over relations r of W_r × (average of its neighbours' vectors under relation r) + bias. Each relation has its own weight matrix W_r. ReLU and dropout 0.2 between the layers, hidden size 128. Two layers let information travel two steps, which covers the herb to compound to target path.
- **Decoder:** DistMult. The encoder gives every compound and target a 128-number vector (z_c and z_t). DistMult scores a compound-target pair as score(c, t) = sum over the 128 positions of z_c[i] × r[i] × z_t[i], where r is a learned 128-number vector for the relation "compound binds target". A higher score means the link is more likely. DistMult is a standard scoring function from knowledge graph embedding.
- **Loss:** BPR, which pushes a true target above a random target for the same compound. Suited to PU data.
- **Training:** Adam, learning rate 0.001, weight decay 1e-5, batch 1,024, early stopping on validation PR-AUC.

### What was done
1. Data preparation, both splits and negative sampling (notebook 01).
2. Model definition and training (02, 03).
3. First evaluation on one fixed split (04).
4. 5-fold cross validation with filtered ranking and the corrected herb-target test (05).
5. Prediction checks and top new candidates (06).
6. Tuning: learning rate, dropout, hidden size, weight decay, number of negatives, plant-part relations, evidence weighting (07). Nothing beat the baseline settings.

Training on one split: edge hiding loss 0.692 to 0.116, best validation PR-AUC 0.380 at epoch 66, stopped at epoch 106; compound disjoint loss 0.691 to 0.110, best validation PR-AUC 0.308 at epoch 74, stopped at epoch 114.

### Problems found in the before version
1. **Fingerprint file defect (most serious).** In `compound_features_ecfp4.csv`, 2,317 of 2,319 rows carry the wrong compound ID. The fingerprint rows are in sorted ID order, but the ID column is not. Looking fingerprints up by that column gives nearly every compound another compound's fingerprint.
   - Checked with RDKit: fingerprints recomputed from each compound's SMILES match the file by row order exactly (100% identical) and match the ID column almost never (0.1%).
   - Checked independently without RDKit: the number of fingerprint bits should grow with molecule size. With the ID-column lookup used in the before version, the correlation is -0.03 (no relationship). By row order it is 0.82.
2. **Overfitting.** Training links are also message-passing links, so the model memorises them. Validation PR-AUC peaks and then falls while training loss keeps dropping.
3. **Herb-target test in notebook 04 was leaky.** It tested all 24,835 herb-target pairs, and about 84% of them are reachable through training links. Fixed in 05 and used in every result below.
4. **The R-GCN loses to popularity on new compounds** (CT Hits@10 0.111 against 0.137).

## 5. After: what was changed

1. **Fingerprints fixed** (notebook 00_fix_ecfp). The file is repaired by row order and verified against RDKit. All later notebooks use the fixed file.
2. **R-GCN overfitting fixes** (08), added one at a time on the same folds:
   - *Message/supervision split:* each epoch, 30% of training links are removed from the graph and only those are scored.
   - *Hide compounds:* each epoch, 15% of training compounds lose all their links, so training looks like the compound-disjoint test.
   - *Feature regularisation:* dropout 0.3 on the fingerprint input, stronger weight decay on its projection, L2 penalty on embeddings.
   - *Mixed negatives:* half the negatives drawn in proportion to target popularity.
   - The final setting was chosen by compound-disjoint validation PR-AUC only: **hide compounds**.
3. **Better protein features** (00b). Each target is described by an ESM-2 embedding, a pretrained protein language model (1,280 numbers per protein) instead of the 20 AAC values. Used only by the two-tower models and the ensemble; the R-GCN still uses AAC.
4. **New models that work on unseen compounds and targets** (09):
   - *Similarity kNN:* Tanimoto similarity on ECFP4. A compound's score for a target is the summed similarity of its k most similar training compounds that hit that target. k (5, 10, 25 or 50) chosen on validation MRR.
   - *Two-tower network:* a compound tower (ECFP4 counts, FCFP4 and MACCS fingerprints, 4,263 inputs, optionally plus which herbs the compound is found in) and a target tower (ESM-2, or AAC for comparison). Each tower is a 2-layer network to 256 numbers. Score = scaled cosine similarity plus an optional per-target bias. Trained with softmax cross-entropy over all training targets, with the compound's other true targets masked. AdamW, early stopping on validation MRR.
   - *Ensemble:* per-compound ranks of the two-tower (ESM-2 + herbs) and kNN scores averaged. The weight (0.25, 0.5 or 0.75) was chosen on validation MRR and was 0.75 in every fold.
   - *Third split:* target disjoint, to test new proteins.
   - Models in 09 are selected on validation MRR instead of validation PR-AUC.

Training of the R-GCN on one split, after the fingerprint fix: edge hiding loss 0.692 to 0.129, best validation PR-AUC 0.483 at epoch 28, stopped at epoch 68; compound disjoint loss 0.692 to 0.138, best validation PR-AUC 0.250 at epoch 26, stopped at epoch 66.

## 6. Results by split

Rows "R-GCN, before" come from the before version. All other rows come from the after version. Same folds throughout.

"R-GCN, after (fixed fingerprints)" is the same model with the same settings as "R-GCN, before"; the only change is the corrected fingerprints. The four "R-GCN +" rows each add one overfitting fix on top of the previous one. All R-GCN rows use AAC protein features; only the two-tower and ensemble rows use ESM-2 (except "Two-tower, AAC", which uses AAC for comparison).

### 6.1 Edge hiding (known compounds)

**Compound-target**

| Model | CT PR-AUC | CT Hits@10 | CT Hits@50 | CT MRR |
|---|---|---|---|---|
| Random scores | 0.020 ± 0.000 | 0.008 ± 0.001 | 0.037 ± 0.002 | 0.006 ± 0.000 |
| Popularity | 0.109 ± 0.004 | 0.160 ± 0.008 | 0.432 ± 0.009 | 0.070 ± 0.003 |
| R-GCN, before | 0.175 ± 0.071 | 0.240 ± 0.083 | 0.514 ± 0.071 | 0.126 ± 0.050 |
| R-GCN, after (fixed fingerprints) | 0.204 ± 0.059 | 0.268 ± 0.033 | 0.556 ± 0.032 | 0.129 ± 0.020 |
| R-GCN + message/supervision split | 0.187 ± 0.061 | 0.200 ± 0.100 | 0.505 ± 0.095 | 0.099 ± 0.046 |
| R-GCN + hide compounds (selected) | 0.231 ± 0.044 | 0.240 ± 0.024 | 0.532 ± 0.019 | 0.120 ± 0.012 |
| R-GCN + feature regularisation | 0.247 ± 0.032 | 0.281 ± 0.022 | 0.561 ± 0.006 | 0.154 ± 0.023 |
| R-GCN + mixed negatives | 0.264 ± 0.014 | 0.213 ± 0.029 | 0.490 ± 0.034 | 0.094 ± 0.026 |
| Similarity kNN | 0.229 ± 0.007 | 0.215 ± 0.007 | 0.478 ± 0.010 | 0.115 ± 0.008 |
| Two-tower, AAC | 0.149 ± 0.031 | 0.359 ± 0.024 | 0.562 ± 0.022 | 0.221 ± 0.028 |
| Two-tower, ESM-2 | 0.218 ± 0.011 | 0.446 ± 0.011 | 0.693 ± 0.014 | 0.244 ± 0.012 |
| Two-tower, ESM-2 + herbs | 0.224 ± 0.029 | 0.445 ± 0.014 | 0.685 ± 0.017 | 0.240 ± 0.019 |
| Two-tower, ESM-2 + herbs, no bias | 0.216 ± 0.015 | 0.443 ± 0.023 | 0.686 ± 0.021 | 0.241 ± 0.021 |
| Ensemble (two-tower + kNN) | 0.144 ± 0.007 | 0.463 ± 0.010 | 0.707 ± 0.011 | 0.278 ± 0.005 |

**Herb-target**

| Model | HT PR-AUC | HT Hits@10 | HT Hits@50 | HT MRR |
|---|---|---|---|---|
| Random scores | 0.037 ± 0.003 | 0.013 ± 0.004 | 0.065 ± 0.006 | 0.009 ± 0.001 |
| Popularity | 0.071 ± 0.005 | 0.137 ± 0.015 | 0.303 ± 0.025 | 0.068 ± 0.008 |
| R-GCN, before | 0.068 ± 0.007 | 0.085 ± 0.020 | 0.242 ± 0.041 | 0.044 ± 0.011 |
| R-GCN, after (fixed fingerprints) | 0.061 ± 0.006 | 0.068 ± 0.008 | 0.191 ± 0.014 | 0.037 ± 0.007 |
| R-GCN + message/supervision split | 0.089 ± 0.011 | 0.096 ± 0.024 | 0.257 ± 0.028 | 0.044 ± 0.012 |
| R-GCN + hide compounds (selected) | 0.092 ± 0.009 | 0.097 ± 0.006 | 0.256 ± 0.026 | 0.048 ± 0.005 |
| R-GCN + feature regularisation | 0.098 ± 0.008 | 0.100 ± 0.014 | 0.260 ± 0.033 | 0.051 ± 0.007 |
| R-GCN + mixed negatives | 0.075 ± 0.011 | 0.068 ± 0.007 | 0.198 ± 0.041 | 0.031 ± 0.004 |
| Similarity kNN | 0.103 ± 0.013 | 0.099 ± 0.017 | 0.269 ± 0.032 | 0.054 ± 0.009 |
| Two-tower, AAC | 0.067 ± 0.010 | 0.068 ± 0.019 | 0.172 ± 0.038 | 0.037 ± 0.011 |
| Two-tower, ESM-2 | 0.093 ± 0.011 | 0.077 ± 0.008 | 0.239 ± 0.024 | 0.045 ± 0.006 |
| Two-tower, ESM-2 + herbs | 0.098 ± 0.010 | 0.084 ± 0.009 | 0.255 ± 0.021 | 0.048 ± 0.007 |
| Two-tower, ESM-2 + herbs, no bias | 0.095 ± 0.010 | 0.082 ± 0.006 | 0.247 ± 0.026 | 0.046 ± 0.007 |
| Ensemble (two-tower + kNN) | 0.094 ± 0.004 | 0.078 ± 0.004 | 0.280 ± 0.025 | 0.043 ± 0.003 |

### 6.2 Compound disjoint (new compounds)

**Compound-target**

| Model | CT PR-AUC | CT Hits@10 | CT Hits@50 | CT MRR |
|---|---|---|---|---|
| Random scores | 0.020 ± 0.000 | 0.009 ± 0.002 | 0.041 ± 0.002 | 0.006 ± 0.000 |
| Popularity | 0.104 ± 0.010 | 0.137 ± 0.008 | 0.380 ± 0.039 | 0.059 ± 0.002 |
| R-GCN, before | 0.077 ± 0.016 | 0.111 ± 0.018 | 0.300 ± 0.024 | 0.056 ± 0.007 |
| R-GCN, after (fixed fingerprints) | 0.072 ± 0.026 | 0.116 ± 0.017 | 0.341 ± 0.043 | 0.057 ± 0.008 |
| R-GCN + message/supervision split | 0.068 ± 0.011 | 0.090 ± 0.015 | 0.312 ± 0.028 | 0.044 ± 0.002 |
| R-GCN + hide compounds (selected) | 0.063 ± 0.015 | 0.077 ± 0.020 | 0.303 ± 0.034 | 0.040 ± 0.008 |
| R-GCN + feature regularisation | 0.070 ± 0.013 | 0.095 ± 0.031 | 0.308 ± 0.034 | 0.049 ± 0.015 |
| R-GCN + mixed negatives | 0.062 ± 0.015 | 0.076 ± 0.026 | 0.249 ± 0.032 | 0.033 ± 0.011 |
| Similarity kNN | 0.186 ± 0.020 | 0.225 ± 0.050 | 0.487 ± 0.022 | 0.114 ± 0.013 |
| Two-tower, AAC | 0.089 ± 0.018 | 0.174 ± 0.024 | 0.321 ± 0.030 | 0.103 ± 0.014 |
| Two-tower, ESM-2 | 0.118 ± 0.019 | 0.200 ± 0.020 | 0.399 ± 0.027 | 0.120 ± 0.014 |
| Two-tower, ESM-2 + herbs | 0.119 ± 0.023 | 0.199 ± 0.020 | 0.399 ± 0.028 | 0.116 ± 0.017 |
| Two-tower, ESM-2 + herbs, no bias | 0.117 ± 0.021 | 0.197 ± 0.020 | 0.402 ± 0.033 | 0.117 ± 0.017 |
| Ensemble (two-tower + kNN) | 0.144 ± 0.014 | 0.233 ± 0.015 | 0.446 ± 0.022 | 0.132 ± 0.018 |

**Herb-target**

| Model | HT PR-AUC | HT Hits@10 | HT Hits@50 | HT MRR |
|---|---|---|---|---|
| Random scores | 0.043 ± 0.017 | 0.012 ± 0.002 | 0.061 ± 0.005 | 0.009 ± 0.001 |
| Popularity | 0.083 ± 0.032 | 0.125 ± 0.022 | 0.291 ± 0.046 | 0.056 ± 0.006 |
| R-GCN, before | 0.075 ± 0.027 | 0.064 ± 0.011 | 0.195 ± 0.034 | 0.035 ± 0.009 |
| R-GCN, after (fixed fingerprints) | 0.072 ± 0.029 | 0.054 ± 0.021 | 0.185 ± 0.027 | 0.025 ± 0.005 |
| R-GCN + message/supervision split | 0.080 ± 0.015 | 0.076 ± 0.028 | 0.205 ± 0.053 | 0.037 ± 0.012 |
| R-GCN + hide compounds (selected) | 0.081 ± 0.016 | 0.067 ± 0.032 | 0.213 ± 0.075 | 0.036 ± 0.014 |
| R-GCN + feature regularisation | 0.087 ± 0.030 | 0.073 ± 0.024 | 0.226 ± 0.037 | 0.037 ± 0.007 |
| R-GCN + mixed negatives | 0.075 ± 0.020 | 0.067 ± 0.052 | 0.174 ± 0.065 | 0.026 ± 0.012 |
| Similarity kNN | 0.106 ± 0.044 | 0.083 ± 0.019 | 0.238 ± 0.022 | 0.048 ± 0.011 |
| Two-tower, AAC | 0.062 ± 0.018 | 0.045 ± 0.006 | 0.138 ± 0.018 | 0.030 ± 0.006 |
| Two-tower, ESM-2 | 0.078 ± 0.028 | 0.052 ± 0.013 | 0.203 ± 0.018 | 0.031 ± 0.005 |
| Two-tower, ESM-2 + herbs | 0.080 ± 0.029 | 0.051 ± 0.012 | 0.203 ± 0.030 | 0.031 ± 0.004 |
| Two-tower, ESM-2 + herbs, no bias | 0.078 ± 0.029 | 0.053 ± 0.008 | 0.195 ± 0.019 | 0.031 ± 0.003 |
| Ensemble (two-tower + kNN) | 0.108 ± 0.041 | 0.083 ± 0.019 | 0.258 ± 0.032 | 0.039 ± 0.006 |

### 6.3 Target disjoint (new targets, after version only)

| Model | CT PR-AUC | CT Hits@10 | CT Hits@50 | CT MRR |
|---|---|---|---|---|
| Random scores | 0.020 ± 0.000 | 0.056 ± 0.007 | 0.264 ± 0.008 | 0.032 ± 0.003 |
| Popularity | 0.026 ± 0.010 | 0.077 ± 0.053 | 0.310 ± 0.072 | 0.049 ± 0.031 |
| Similarity kNN | 0.020 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.011 ± 0.000 |
| Two-tower, AAC | 0.075 ± 0.016 | 0.274 ± 0.022 | 0.578 ± 0.024 | 0.143 ± 0.011 |
| Two-tower, ESM-2 | 0.197 ± 0.039 | 0.446 ± 0.053 | 0.716 ± 0.018 | 0.244 ± 0.026 |
| Two-tower, ESM-2 + herbs | 0.196 ± 0.034 | 0.443 ± 0.045 | 0.712 ± 0.030 | 0.253 ± 0.027 |
| Two-tower, ESM-2 + herbs, no bias | 0.178 ± 0.039 | 0.435 ± 0.050 | 0.720 ± 0.026 | 0.241 ± 0.023 |
| Ensemble (two-tower + kNN) | 0.095 ± 0.014 | 0.397 ± 0.042 | 0.705 ± 0.031 | 0.164 ± 0.023 |

In this split the true target is ranked only against the held-out targets (about 200), so even random scores reach Hits@10 of 0.056. These numbers should not be compared with the other two splits. Similarity kNN cannot score a protein it has never seen, so it gets 0 here. Herb-target is not evaluated in this split.

## 7. Reading the results

1. **Fixing the fingerprints helped the R-GCN on known compounds** (CT Hits@10 0.240 to 0.268), but not on new compounds (0.111 to 0.116).
2. **The overfitting fixes reduced memorisation** (train-test score gap on known compounds 0.186 to 0.048) and helped herb-target results, but none of them made the R-GCN beat popularity on new compounds.
3. **The R-GCN cannot generalise well to new compounds** because it learns a vector for each compound from its links, and a new compound has no links to learn from.
4. **The new models solve this for compound-target prediction.** On new compounds, kNN reaches CT Hits@10 0.225 and the ensemble 0.233, against 0.137 for popularity. On known compounds the ensemble reaches 0.463 against 0.240 for the original R-GCN.
5. **ESM-2 beats AAC on every split** (two-tower CT Hits@10: known compounds 0.359 to 0.446, new compounds 0.174 to 0.200, new targets 0.274 to 0.446).
6. **Herb-target prediction is still the weak point.** No model beats popularity on HT Hits@10, Hits@50 or MRR in either split or in any fold. Models only beat it on HT PR-AUC. The likely reason is the max over a herb's compounds: a herb can have 100 or more compounds, and one high score is enough to push a target up. Popularity simply ranks the targets that most compounds hit.
7. **PR-AUC and ranking metrics sometimes disagree.** The ensemble has the best Hits and MRR but a lower PR-AUC, because it averages per-compound ranks, which suits ranking within a compound but not comparing scores across compounds.

## 8. Leakage checks

| Check | Result |
|---|---|
| Test links in the message-passing graph | None, only training links are used |
| Test data used for early stopping, tuning, k, or ensemble weight | Never, validation only |
| Label information inside the features (fingerprints, ESM-2, herb membership) | None |
| Other true targets used as negatives | No, they are masked or filtered |
| Herb-target test pairs reachable through training links | Excluded (05, 08, 09); the 04 version was leaky and is not used |
| Compounds with an identical fingerprint across train and test | Mild effect, see below |

**Identical fingerprints in compound disjoint.** 470 compounds share a fingerprint with another compound, mostly stereoisomers. On average 13.6 of the 109 test compounds per fold (12.5%) have an identical twin in training, covering 6.8% of test links. kNN gets Hits@10 of 0.380 on those links and 0.215 on the rest. Without the twins, kNN still beats popularity (0.215 against 0.134), so the conclusion holds, but new-compound kNN results are slightly optimistic (0.225 with twins, 0.215 without).

**Edge-hiding PR-AUC is inflated.** About 71% of edge-hiding negatives come from the 1,593 compounds that have no known target at all, which are easy to rule out. A score that only checks whether a compound has training links gets PR-AUC 0.303. Hits@K and MRR rank targets within one compound and are not affected, so they are the better measures for this split. Compound-disjoint negatives come only from test compounds and do not have this issue.

## 9. Training, validation and test loss

- **Training loss** is logged for the R-GCN (values in sections 4 and 5). It starts at 0.692, which is the value for a model that cannot yet tell true targets from random ones, and falls to about 0.11 to 0.14. Validation PR-AUC peaks and then drops while the loss keeps falling, which is overfitting: at epochs 66 to 74 in the before version and epochs 26 to 28 in the after version. Early stopping keeps the weights from the best validation epoch.
- **Training loss of the two-tower network** (softmax cross-entropy) is not logged; it is stopped on validation MRR.
- **Validation** is measured with PR-AUC (R-GCN) or MRR (two-tower, kNN, ensemble), not with a loss value.
- **Test** is measured only with PR-AUC, Hits@K and MRR. A test loss is not computed, because with positive-unlabeled data a loss on random negatives says little about ranking quality.
- **No overfitting to validation:** in the after version, validation MRR and test MRR are close (two-tower ESM-2 + herbs: known compounds 0.244 and 0.240, new compounds 0.134 and 0.116).
- **Validation PR-AUC is higher than test PR-AUC** (about 0.50 against 0.20 for the R-GCN) because validation uses 1:10 negatives and test 1:50. This is expected and is not overfitting.
- Runs on the GPU are not fully repeatable; differences below about 0.01 are noise.

## 10. Conclusion and next steps

- The before version's results are not valid because of the fingerprint defect and should be replaced by the after results.
- The R-GCN (after fingerprint fix and overfitting fixes) is the GNN baseline. It works for known compounds but not for new ones.
- The two-tower network with ESM-2 and the kNN ensemble are the strongest compound-target models and the first to beat popularity on new compounds and new targets.
- Next steps:
  1. Herb-target prediction: try other ways to combine compound scores (mean, top-k mean, noisy-OR) or train directly on herb-target pairs.
  2. Remove or group identical-fingerprint compounds, or split by chemical scaffold, for a stricter compound-disjoint test.
  3. Draw edge-hiding negatives only from compounds that have known targets, so PR-AUC is meaningful there.
  4. Use these results as the reference for MAMGN-HTI next semester.
