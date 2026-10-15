# Combined test: hidden-type calls plus duplication, scored on new donors

Pre-registered design. It was written and fixed on 2026-10-15, before any full-run result existed. No rule below changed after that point, and every outcome is reported.

Frozen files (md5 of the originals at freeze): design a3f0fa09a25f3e4b774365492a51f2d5, single-pool script 6943da049def76ee229e844ee5489dba, analysis script 570369a3d4f36781a616e8584a00fde0, run script be7742d3d450a38a5bc72645b69f7d19, plan 90cdadaff64c165dd78f73008ec25900. `plan.csv` here is the frozen file byte for byte; `run_one.py`, `analyze.py` and `run_all.sh` differ from the frozen scripts only in paths, imports and comments.

## 1. Question and hypotheses
Question: if a rare-type classifier is trained on donors where a discovery method has called hidden subtypes, does the classifier find the true hidden cells in held-out donors, and does duplication of the called cells help? Scope of the claim: both discovery methods scored the pool with the test donors present but unlabelled (discovery is transductive over the pool and was not rerun on training donors only). The test donors are therefore held out from the classifier, not from discovery. A rerun of discovery on training donors only is declared future work.
- H1 (primary). Our ranking, thresholded at Leiden subclustering's call count, plus duplication ranks the true hidden cells of held-out donors better than Leiden-subclustering calls plus duplication. Metric: average precision (AP), section 6. Readout: logistic regression.
- H2 (secondary). Duplication adds detection beyond a shift of the operating point: ours with versus without duplication at a matched number of calls (recall at the call count of the duplication setting). H2 is tested with logistic regression, where duplication equals sample weighting, and again with random forest and MLP, where duplication differs from weighting. The experiment can support a duplication benefit only through the nonlinear readouts. Under logistic regression it tests the transfer value of the call set.
- H3 (descriptive). Gap to the true-label upper bound: (true labels + duplication) minus (our calls + duplication).
- H4 (damage check, non-inferiority). Pseudo-classes do not lower macro F1 over the broad labels by more than 0.01 against the broad-only classifier. Rule in section 8.

## 2. Pools (fixed)
The 69 pools of the 8-atlas test set. The saved predictions of both methods cover exactly the cell ids of these pools, and reproduce the held-out test (union AP 0.3213 against 0.2204, exact atlas sign-flip p 0.03125). `run_one.py` requires exact cell-id set equality per pool. A pool that fails writes a status row "dropped" and `analyze.py` prints the dropped list.
Pool-seeds that fail the donor rules below are ineligible and written as status rows. Eligibility is computed from metadata only (`plan.csv`): 58 of 69 pools and 158 of 207 pool-seeds are eligible, in all 8 atlases (section 4 explains the 13 pool-seeds lost to the two-donor rule). Pools with an eligible seed per atlas: arthritis 5, cereb 5, hspc 21, htapp 5, hypomap 1, luad 4, ocular 12, psccolon 5.

## 3. Calls and pseudo-labels (no truth used)
Inputs are the saved cell scores of the two methods, 20,000 rows per pool:
- our method: cell_id, score (continuous). No binary call, no group id.
- Leiden subclustering: cell_id, score, call (bool), group (int, -1 = not called). Score is nonzero only for called cells.
Rules, applied to training-donor cells only:
- Leiden calls: a cell with call = True and group >= 0 gets pseudo-class (broad label, group). Setting `leiden_dup`/`leiden_nodup`.
- Leiden collapsed (sensitivity): all called cells of a broad label form one pseudo-class (broad label, *). Setting `leiden_collapsed_dup`. Purpose: same granularity as ours.
- Our calls: our method emits no call set. Rule: K = number of Leiden-called cells among the training cells of the same pool and seed; the K training cells with the highest score are called (stable order for ties); called cells of one broad label form one pseudo-class (broad label, *), as the task rule requires when no group ids exist. K is a method output, not truth. This matches the call count of the two methods (matched operating point rule).
- A pseudo-class with fewer than 5 training cells is dropped (its cells keep the broad label). Uncalled cells keep the broad label. Called cells leave their broad class.
- True hidden labels (upper-bound settings): the target rule of the held-out test (`eval/targets.py`: label L, finest type T != L and T not in {None, nan, Unknown}, n >= 10, n/n(L) < 0.10) on the pool; class (L, T) for training-donor cells, same 5-cell minimum. These labels are used for nothing else in training.

## 4. Split (fixed)
Donor-disjoint within each pool. Donors sorted by sha1(pool|donor). Test block = n_test = max(1, round(D/3)) consecutive donors (cyclic) starting at (seed * n_test) mod D; seeds 0, 1, 2. D = 2 gives 1 train + 1 test donor and only two distinct splits, so seeds 0 and 1 run and seed 2 is ineligible (13 pool-seeds, the 13 two-donor pools; including the only eligible hypomap pool); D = 1 has no donor-disjoint split and is ineligible (30 of 207 pool-seeds, from 10 pools). Pools with D = 4 never test one donor. Both are limitations. Also ineligible: test donors hold < 20 true hidden cells (5 pool-seeds) or training donors hold < 20 true hidden cells in classes of >= 5 cells (1 pool-seed). The eligibility rule uses true hidden counts, identically for every setting, and cannot favour a method. Pool value = mean over eligible seeds, each seed with equal weight (all splits in the mean are distinct).

## 5. Features and classifier (aligned with the Part I code in `part1/`)
- Features, per split: genes with counts in >= 3 training cells; 2,000 highly variable genes (scanpy seurat_v3) chosen on training cells only; log1p of 1e4 x counts / library size over the selected genes (Part I lognorm); no z-scoring or clipping; PCA-50 fitted on at most 30,000 training cells; all cells projected. Deviations from Part I: HVGs and the gene filter use training cells only (the HVG-leakage answer), and the filter is >= 3 training cells instead of nonzero total count.
- Primary readout: LogisticRegression(max_iter=2000, random_state=seed), C = 1, as in Part I. Each fit records n_iter and a flag if it hit the cap; the analysis prints the number of capped fits. A capped fit is reported, not dropped.
- Secondary readouts (separate runs, same code with `--clf`): RandomForest with 500 trees and MLPClassifier(max_iter=500), both as in Part I. kNN is not run.
- Duplication (Part I rule): every pseudo or hidden class is topped up to 256 cells, k = max(0, 256 - n_train(class)) copies drawn with replacement from that class's training cells. Classes that already hold at least 256 training cells get no copies. Broad classes are never duplicated. Column n_added reports the total added cells per setting. Leiden calls form more and smaller pseudo-classes than ours, so Leiden receives more added cells. The collapsed Leiden setting is the control for this dose difference.

## 6. Settings and metrics
Settings (8 rows per pool-seed): `broad` (1); `leiden_dup` (2); `ours_dup` (3); `ours_nodup` (4); `true_dup` (5); `true_nodup` (6); plus two extras: `leiden_nodup` (for matching) and `leiden_collapsed_dup` (granularity sensitivity).
Scores on test donors. U = true hidden cells in test donors (union of the targets). s(c) = summed predicted probability of all pseudo/hidden classes. For `broad`, which has no such class, s = 1 - max probability. This broad score is reported but never compared as a detector.
- ap: AP of s against U. Threshold-free.
- rprec: precision in the top |U| cells by s (ties split at expectation). Both methods are read at the same K = |U|, so rprec is the matched-budget recall for cross-method comparisons.
- recall_argmax, precision_argmax, n_calls: true hidden cells assigned to any pseudo/hidden class at the classifier's own argmax. Operating-point dependent and descriptive only.
- prec_at_kref / rec_at_kref: precision and recall of the s-ranking at K = number of argmax calls of the duplication setting. For a duplication versus no-duplication pair of one method, both settings are read at the same K, so this is a matched operating point. It is never used across methods, because each method's duplication setting has its own call count.
- macro_f1_broad: macro F1 over the original broad labels (predicted pseudo-classes mapped to their parent label).
- Diagnostics (use truth, never used in selection): pseudo_train_purity_diag, n_added, n_iter, hit_iter_cap.

Why the primary metric is AP, and what can show a duplication effect. With logistic regression, duplicating a cell k times equals weighting it k+1 times, and weighting a class mainly moves the decision threshold. Class weighting also changes the coefficients, so the ranking can move, but earlier results show it moves little. We therefore expect (an empirical expectation, not an identity) that AP moves little with duplication while argmax recall rises and precision falls along the curve. A raw argmax comparison of duplication against none is not evidence, and recall, not F1, is the quantity of interest for a rare class. AP is therefore primary for H1: it asks whether the calls carry transferable signal and does not depend on the threshold. Recall at the matched number of calls (rec_at_kref, duplication versus none within one method) is the metric for H2. A duplication effect beyond threshold sliding appears there as a positive matched difference, most plausibly under the nonlinear readouts.

## 7. Tests
`analyze.py` averages seeds within a pool, takes atlas-equal weighted means, a pool-level weighted sign-flip (20,000 draws, seed 0) and the exact sign-flip over atlas mean differences. One call per readout. Output columns: clf, A, B, A_mean, B_mean, diff = A - B, pool_flip_p, atlas_ahead, atlas_exact_p, Holm-adjusted atlas p (secondary family), and a one-sided 95% t lower bound on the atlas-mean differences.
- Primary (P): ours_dup - leiden_dup on ap, logistic regression.
- Secondary family (S, Holm correction over these 4 contrasts, per readout): ours_dup - leiden_dup on rprec; ours_dup - leiden_collapsed_dup on ap; ours_dup - ours_nodup on ap; ours_dup - ours_nodup on rec_at_kref.
- Non-inferiority (N, H4): ours_dup - broad and leiden_dup - broad on macro_f1_broad.
- Descriptive (D, uncorrected, no claim): recall_argmax and rec_at_kref differences for duplication versus none, true_dup - ours_dup (ap, rprec), ours_dup - leiden_dup on macro_f1_broad.
- Not computed: any AP contrast against `broad` (its score is not a detector). The random forest and MLP runs repeat the same families and are labelled exploratory for H1; they carry H2.
Test unit is the atlas. Atlases with no eligible pool are listed. Pools dropped for cell-id mismatch are listed. The analysis stops if any expected eligible pool-seed is missing.

## 8. Decision rule
- H1 SUCCESS iff on ap with logistic regression: diff > 0, atlas exact p < 0.05, and at least n_atl - 1 of the n_atl atlases ahead (n_atl = atlases with an eligible pool; 8 expected). Else if diff > 0 and p >= 0.05: INCONCLUSIVE (say "inconclusive"). Else NO ADVANTAGE. The Leiden-collapsed contrast must point the same direction; if it flips sign, report H1 as granularity dependent.
- H2: a duplication benefit is claimed only if, for random forest or MLP, the matched-recall difference (ours_dup - ours_nodup, rec_at_kref) is > 0 with Holm-adjusted atlas exact p < 0.05. Logistic regression results for H2 are reported without a benefit claim.
- H3: reported as the difference with its interval, no test claim.
- H4 passes iff the one-sided 95% lower bound of (ours_dup - broad) on macro_f1_broad exceeds -0.01. Otherwise report "pseudo-classes may damage the broad labels".
- Secondary and descriptive contrasts beyond the Holm family are exploratory. Whatever the outcome is reported; no re-running with changed rules.

## 9. Decisions fixed before the run, and limitations
1. Our method has no call set. Setting 3 is "our ranking at Leiden's call rate": K is the Leiden call count among the same training cells, which ties our operating point to Leiden's (matched operating point) and makes the number of called cells equal by design. Kept as written in section 3.
2. Granularity and dose confound: Leiden groups give more pseudo-classes per pool than ours, and top-up to 256 gives Leiden more duplicated cells. The collapsed Leiden setting is the control. The primary contrast stays ours_dup - leiden_dup on AP with logistic regression; section 8 requires the collapsed contrast to agree in sign, or H1 is reported as granularity dependent.
3. Duplication with logistic regression is a threshold shift in practice (an empirical expectation, section 6). The nonlinear readouts are pre-declared as secondary runs for H2. All three readouts run: lr first, then mlp and rf.
4. Transductive calls (section 1). A rerun of discovery on training donors only is out of scope.
5. Donor support is thin: 23 of 69 pools have at most 2 donors and 10 have one. With one training donor, generalisation is a single-donor shift. hypomap contributes one pool, so its atlas mean is one noisy number with 1/8 of the weight. The minimum exact p with 8 atlases is 2/256 = 0.0078.
6. Pseudo-class purity can be low. A noisy class may hurt the broad labels. H4 checks it.

## 10. Files and run
`run_one.py` (one pool, one seed, one readout per call), `analyze.py`, `run_all.sh`, `plan.csv` (158 eligible pool-seeds). Python 3.12 with scanpy 1.11.1 and scikit-learn 1.5.2, run with `nice` and 4 threads. All runs are resumable. Order: logistic regression first, then the analysis, then the secondary readouts.
Commands: `combination/run_all.sh lr`, `combination/run_all.sh mlp`, `combination/run_all.sh rf`.
