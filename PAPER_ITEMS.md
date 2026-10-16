# Paper items

The table lists 25 paper items: 3 reproduced, 7 partial, 1 recomputed with another readout, 3 whose scripts are in `supplement/` but that no level runs, and 11 not in this repository.

`reproduce.sh paper` recomputes numbers. "Reproduced" means the notebook recomputes and asserts the values behind the item. "Partial" means it does so for only some of them.

| Paper item | Notebook section | Level | Status |
|---|---|---|---|
| Fig. 1a, augmentation minus duplication over 88 datasets | 04 section 1 | paper | partial: win counts and deep-generator summary, but the per-test counts are not recomputed |
| Fig. 1b, default-threshold recall versus matched recall | 04 section 1 | paper | partial: duplication against no augmentation only |
| Fig. 1c, duplication equals sample weighting | 04 section 2 | paper | reproduced |
| Fig. 1d, union AP of our method, Leiden subclustering, scCAD (per label), classifier uncertainty and the loops | 01 (sections 2, 4, 5) and 02 (section 2) | paper | partial: union AP values except classifier uncertainty, and the random-ranking line is not recomputed |
| Fig. 2a, atlas-equal means of the methods | 01 (section 5) and 02 (sections 1 and 2) | paper | partial: every method except classifier uncertainty, GiniClust3 and FiRE (the figure plots GiniClust3 and FiRE runs on the rebuilt subsamples of Supp. Table S6, while notebook 02 scores earlier runs of both on the original subsamples), but the random-ranking line is not recomputed |
| Fig. 2b, per-atlas differences | 01 (sections 2 and 3) and 02 (section 3) | paper | partial: per-atlas differences against Leiden subclustering are asserted, but only the mean, atlas count and p are asserted against scCAD with labels |
| Fig. 3, separability levels and UMAPs | none | none | not in repo (panel a uses the 10-atlas results. Panels b and c show 8-atlas pools, whose h5ads are not released. No level draws figures) |
| Supp. Fig. S3, hidden-type calls plus duplication | none | none | not in repo (`combination/` holds the code, but the per-pool result rows are not included) |
| Supp. Fig. S1, example pool | none | none | not in repo (needs that pool's h5ad, one of the 69 pool h5ads of the 8-atlas test set listed in [REPRODUCE.md](REPRODUCE.md#not-included-and-known-gaps), and a figure script) |
| Supp. Fig. S2, null twins | 03 (101 pairs, opening cells) and 03 section 2 (69 pairs) | paper | reproduced: rates. The notebook pools the 101 pairs and gets an AUC of 0.61. The paper weights each atlas equally, which gives 0.60 (0.602 for our method, 0.604 for Leiden subclustering) |
| Supp. Table S6, published detectors against our method | 02 sections 2 and 3 | paper | partial: the first block, but the rebuilt-subsample block and the 10-atlas block are not recomputed (their predictions are not released, and the GiniClust3 and FiRE rows of notebook 02 are earlier runs on the original subsamples) |
| Supp., metrics for the called cells | 02 sections 3 and 4 | paper | reproduced |
| 31-type AIDA run (values not printed in the paper) | 04 section 3 | paper | recomputed with another readout, but the paper's 31-type values (matched-operating-point paragraph) are not recomputed |
| Supp., per-subtype results on the 8-atlas pools | none | none | not in repo (separability is computed from the counts of the 69 pools, whose h5ads are not released) |
| Supp. Table S1, datasets | none | none | partial: `ATLAS_MANIFEST.csv`, `data/used_datasets.csv` hold the sources, not the table |
| Supp. Tables S2 to S5 (duplication study conditions, comparator checks, methods compared) and the list of methods missing | none | none | not in repo (S3 and S4 come from the comparator checks, which are not included. S2, S5 and the list are descriptive) |
| Label and donor ablations, strata, CHOIR | none | none | not in repo (the label ablation, strata and CHOIR use the 10-atlas predictions, which we did not release. The donor ablation recomputes features on the 69 pools of the 8-atlas set, whose h5ads are not released) |
| Supp. Fig. S4 and Table S8, labels with hidden subtypes (60 labels) | none | none | not in repo (the UMAPs and separability need the counts of the 69 pools, whose h5ads are not released. No figure scripts are included) |
| Supp., donor correction (Harmony) and the donor-strata table (Table S7) | none | none | not in repo (there is no code here, and each uses both test sets. Harmony rescores from counts, and the 69 pool h5ads of the 8-atlas set are not released. Table S7 reads the saved scores, which we did not release for the 10-atlas set) |
| Supp., biology on a dataset with automated labels | none | none | not in repo (there is no code here, and the blood dataset is row `cede8227` of `ATLAS_MANIFEST.csv`) |
| Supp. Table S9, validation of called groups against annotated types | none | none | not in repo (the gene tests need the counts of the 69 pools, whose h5ads are not released) |
| Supp., further details of the method: time and memory per pool | none | none | not in repo (per-job `/usr/bin/time -l` logs on the run machine) |
| Supp., further analyses of the duplication study (one-versus-rest classifiers, prior-shift control, resampling of evaluation donors, marker recovery, rare types close to another type, scDesign3 against scVI on donor-disjoint splits, where conditional scVI loses recall, a type with no training cells from one ancestry, other latent codes, duplication on foundation-model embeddings, two further interpolation methods (ISMOTE and Just Mix Once), choosing the condition by training-set size, cost to major types, false positive rates) | none | none | scripts in [`supplement/`](supplement/README.md), which no level runs (most need the AIDA, HLCA and lupus inputs, which we did not release) |
| Supp. Table S10 and the distance metrics of the generator papers on AIDA | none | none | scripts in [`supplement/`](supplement/README.md), which no level runs (Table S10 is descriptive. We did not release the AIDA inputs or the generated cells) |
| Supp. Fig. S5 and the further analyses (time, memory and disk, other regressors, other embeddings, unreliable labels, annotation, chromatin accessibility) | none | none | scripts in [`supplement/`](supplement/README.md), which no level runs (most analyses rerun the method on subsamples of the 10-atlas set, which only `reproduce.sh full` rebuilds, and the regressor comparison also needs the training pools and the rebuilt 8-atlas subsamples, which we did not release) |
