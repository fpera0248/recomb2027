# Finding rare cell subtypes hidden inside broad labels

Code for our RECOMB 2027 submission. Given a pool of cells with a coarse visible label (say "T cell"), the method ranks cells by how likely each belongs to a finer, unlabelled subtype. It never sees the finer annotation.

## Quick check

```
pip install -r requirements.txt    # Python 3.12
./reproduce.sh paper
```
- Download: 1.2 GB of release assets, about 2.5 GB on disk.
- Time: about 4 minutes plus the download (measured on a laptop, 4 threads).
- Success: the run prints `paper: all notebooks ran and every assertion held`.
- Checks: the `paper` tier recomputes numbers from the saved predictions and result tables and asserts them (the bootstrap interval to 0.005). It does not refit the model. Paper items lists what it covers.
- `./reproduce.sh rerun` applies a refit scorer as a consistency check, so its numbers differ a little (atlas-equal union AP 0.3195 against 0.3213).
- Container: the `docker run` line under Install (image `v0.1.0`; the `paper` and `rerun` tiers pass in it).
- Reproduction problems: https://github.com/fpera0248/recomb2027/issues

## Paper items

The table lists 22 paper items: 4 reproduced, 6 partial, 1 recomputed with another readout, and 11 not in this repository.

`reproduce.sh paper` recomputes numbers; no tier draws a figure. "Reproduced" means the notebook recomputes and asserts the values behind the item. "Partial" means it does so for only some of them.

| Paper item | Notebook section | Tier | Status |
|---|---|---|---|
| Fig. 1a, augmentation minus duplication over 88 datasets | 04 section 1 | paper | partial: win counts and deep-generator summary; the per-test counts are not recomputed |
| Fig. 1b, default threshold against matched point | 04 section 1 | paper | partial: duplication against no augmentation only |
| Fig. 1c, duplication equals sample weighting | 04 section 2 | paper | reproduced |
| Fig. 1d, union AP of our method, Leiden subclustering, scCAD and the loops | 01 sections 2, 4, 5; 02 section 2 | paper | partial: union AP values; the random-ranking line is not recomputed |
| Fig. 2a, atlas-equal means of the methods | 01 section 5; 02 sections 1 and 2 | paper | partial: every method except classifier uncertainty |
| Fig. 2b, per-atlas differences | 01 sections 2 and 3; 02 section 3 | paper | partial: per-atlas differences against Leiden subclustering are asserted; against scCAD with labels only the mean, atlas count and p |
| Fig. 3, separability tiers and UMAPs | none | none | not in repo (panel a uses the 10-atlas results; no tier draws figures) |
| Supp. Fig. S3, hidden-type calls plus duplication | none | none | not in repo (`combination/` holds the code; the per-pool result rows are not included) |
| Supp. Fig. S1, example pool | none | none | not in repo (needs that pool's h5ad, one of the 69 below, and a figure script) |
| Supp. Fig. S2, null twins | 03 (101 pairs, opening cells) and 03 section 2 (69 pairs) | paper | reproduced: rates; the AUC recomputes to 0.61 against the 0.60 printed |
| Supp. Table S6, published detectors against our method | 02 sections 2 and 3 | paper | reproduced |
| Supp., metrics for the called cells | 02 sections 3 and 4 | paper | reproduced |
| 31-type AIDA run (values not printed in the paper) | 04 section 3 | paper | recomputed with another readout; the paper's 31-type values (matched-operating-point paragraph) are not recomputed |
| Supp., per-subtype results on the 8-atlas pools | none | none | not in repo |
| Supp. Table S1, datasets | none | none | partial: `ATLAS_MANIFEST.csv`, `data/used_datasets.csv` hold the sources, not the table |
| Supp. Tables S2 to S5 (duplication study conditions, comparator checks, methods compared) and the list of methods missing | none | none | not in repo |
| Label and donor ablations, strata, CHOIR | none | none | not in repo |
| Supp. Fig. S4 and Table S8, labels with hidden subtypes (60 labels) | none | none | not in repo |
| Supp., donor correction (Harmony) and the donor-strata table (Table S7) | none | none | not in repo |
| Supp., biology on a dataset with automated labels | none | none | not in repo |
| Supp. Table S9, validation of called groups against annotated types | none | none | not in repo |
| Supp., further details of the method: time and memory per pool | none | none | not in repo (per-job `/usr/bin/time -l` logs on the run machine) |

## Layout

- `hiddenrare/`: our method (candidate groups, marker and halo features, scorer, blend).
- `models/`: the refit scorer `refit_dev_nat.pkl`.
- `eval/`: the evaluation and the checks of the `paper`, `rerun` and `full` tiers.
- `notebooks/`: the four notebooks the `paper` tier runs.
- `data/`: builders of the test and twin pools, reference pool lists and targets ([data/README.md](data/README.md) rebuilds the test sets).
- `scripts/`: `run_pool.sh` (our method on one pool) and `run_tool.sh` (one published tool).
- `tools/`: wrappers of the published detectors, their pinned repositories and commits (`sources.txt`) and `fetch_sources.sh`, which clones them ([tools/README.md](tools/README.md)).
- `part1/`: the study of known rare types, duplication against generators ([part1/README.md](part1/README.md)).
- `combination/`: the hidden-type calls plus duplication experiment ([combination/README.md](combination/README.md)).
- `docker/`: the two image definitions (see `CONTAINER.md`).
- `.github/workflows/`: CI (self-tests on every push; it runs no tier), the image builds and the container check (`paper` and `rerun` tiers in the published image, manual).

## Method

1. `hiddenrare/candidates.py`: Leiden clustering of each label at 8 resolutions in two spaces (pool PCA-50, and per-label HVG + PCA-50). Clusters with at least 10 cells and under 10% of the label are candidate groups.
2. `hiddenrare/markers.py`, `halo.py`: marker and neighbourhood ("halo") statistics per group. With geometry features, these make 41 features per group.
3. `hiddenrare/scorer.py`: a gradient-boosted regressor (`models/refit_dev_nat.pkl`) predicts how pure each group is. A cell's score is the best group it belongs to, plus a small isolation tie-break.
4. `hiddenrare/baselines.py`, `blend.py`: the final score is 0.75 * rank(scorer) + 0.25 * rank(Leiden subclustering score).

`models/refit_dev_nat.pkl` is a refit of the same recipe; we did not keep the fit behind the saved scores of the 8-atlas test set. The `paper` tier reads the saved scores and is exact; the `rerun` tier applies the refit and is a consistency check (`eval/rerun.py`).

Feature-table column names (`mk_*`, `hl_*`) are kept from the training tables because `hiddenrare/scorer.py` reads them by name in the order used to fit the model.

## Install

Pick one.

**Python environment.** Python 3.12, `pip install -r requirements.txt` or `conda env create -f environment_hiddenrare.yml`. The versions are those of the interpreter that computed the 10-atlas results (macOS arm64); `piplist_hiddenrare.txt` is the dependency closure of `requirements.txt` in that interpreter. scikit-learn must be 1.5.2 (the model pickle was written with it). Leiden partitions depend on the numerical stack, so per-cell scores differ on another machine. Rerunning reproduces the pipeline and the evaluation, not the paper's exact per-cell scores. The published tools have their own environment (`environment_tools.yml`, `Rpackages_tools.csv`, `piplist_tools.txt`); see `ENVIRONMENTS.md`.

**Container.** Two images, defined in `docker/`. `.github/workflows/build-images.yml` builds them and pushes them to GHCR on every release or manual dispatch; both are published with the release tag `v0.1.0` (use this one; the `v0.1.1` images exist but no check has run them; `latest` and `main` move with later builds) and pull without a login. The `recomb2027-hiddenrare` image passes the `paper` and `rerun` tiers on a GitHub Actions runner ([container check](https://github.com/fpera0248/recomb2027/actions/runs/38827295374)); the `recomb2027-tools` image has not been run end to end. They are amd64 only (emulated on Apple silicon, for example `docker run --platform linux/amd64`):

| Image | Contents |
|---|---|
| `ghcr.io/fpera0248/recomb2027-hiddenrare` | our method, the evaluation, the notebooks (Python 3.12) |
| `ghcr.io/fpera0248/recomb2027-tools` | the published detectors: Python tools, CellSIUS, GapClust, SCISSORS (R 4.6.1, Seurat 4.4.0) |

```
docker pull ghcr.io/fpera0248/recomb2027-hiddenrare:v0.1.0
docker run -v "$PWD/out":/out ghcr.io/fpera0248/recomb2027-hiddenrare:v0.1.0 \
    env ASSETS=/out/assets OUT=/out/reproduce_out reproduce paper
```
For Apptainer and more details, see `CONTAINER.md`.

## Reproduce the numbers

```
./reproduce.sh paper     # download the release assets, recompute the Part II numbers of the 8-atlas test set plus the Part I panel and extra checks, assert the match
./reproduce.sh rerun     # rescore the 69 held-out pools with this repository's scorer
./reproduce.sh full      # rebuild the 10-atlas test pools from CELLxGENE and rerun our method, Leiden subclustering and (TOOLS=1) the published tools on them (untested)
```

| Tier | Input | Exactness | Time and disk |
|---|---|---|---|
| paper | saved predictions of every method run on the 69 pools of the 8-atlas test set, and the Part I result tables | exact: values, tests, null-test rates and the Part I numbers of notebook 04 are asserted; the bootstrap interval is checked to 0.005 | about 4 minutes plus the download, 1.2 GB downloaded and about 2.5 GB on disk (measured on a laptop, 4 threads) |
| rerun | frozen candidate tables and Leiden scores | not exact (refit scorer, see Method): mean Spearman of per-cell scores 0.878 against the saved ones, atlas-equal union AP 0.3195 against 0.3213, single pools can move by up to 0.2; asserted within 0.01 and Spearman above 0.8 (`eval/rerun.py`) | about 2 minutes, about 1.05 GB downloaded (pools, predictions, rerun inputs) (measured) |
| full | CELLxGENE source files | approximate: atlas-equal means and differences within 0.03 of the paper, same sign, atlas-level exact p below 0.05 (`eval/full_check.py`) | 29.3 GB of downloads (sum of the manifest sizes); time and disk for the pools and the run are not measured; untested end to end |

`full` does not rebuild the twin pools or run the leak check or the Part I panel. It also skips the strata, separability tiers, ablations, CHOIR and figures. `TOOLS=1` also runs the published tools. They need their own Python environment (scikit-learn 1.9.1, which conflicts with `requirements.txt`; point `TOOLS_PYTHON` at it, see `ENVIRONMENTS.md`) and `SCGPT_CKPT`. A tool that already has an output is skipped; failures go to `$OUT/tool_failures.txt`. Without `TOOLS=1` only the Leiden comparison is checked and T2 (against scCAD) is not evaluated. `eval/full_check.py` asserts 57 pools in 10 atlases.

At the `paper` tier every value the notebooks print is recomputed from the saved predictions and asserted (see Paper items for what is partial), so a mismatch fails the run. The `rerun` and `full` tiers rerun code, so they can differ from the paper (see table). In the `recomb2027-hiddenrare` container on a GitHub Actions runner, `paper` took 5 min 18 s including the download and `rerun` took 3 min 0 s.

The code release is v0.1.1. The release assets and the checked container images stay on v0.1.0 (v0.1.1 has no assets), and the v0.1.0 image holds the v0.1.0 code, which differs from v0.1.1 only in documentation, the image tag in `Apptainer.def`, the manual container-check workflow, the seeds of `part1/run.sh` (no tier runs it) and a print format in notebook 03. `fetch_data.sh` downloads the assets of a release (`TAG`, default v0.1.0) and checks them against `ASSETS.md5`; for `full` it downloads the source atlases listed in `ATLAS_MANIFEST.csv` and checks their sizes (and md5 where the manifest has one). `python data/build_manifest.py --check` re-verifies the URLs; `--refresh` rewrites the manifest from the CELLxGENE curation API (it then points to the latest dataset versions).

A missing release or asset gives 404 (check `TAG`; the assets are on release v0.1.0, and release v0.1.1, the version in `CITATION.cff`, has none). `fetch_data.sh` then falls back to `gh release download` (install the GitHub CLI and log in). Set `ASSET_SRC=DIR` to take the assets from a local directory, for offline use.

## Atlases

The source atlases are public CELLxGENE files. `ATLAS_MANIFEST.csv` lists every one with its collection where recorded, its `dataset_version_id` download URL, size, md5 where known and the set it belongs to (`key` for the 10-atlas set, `name` for the 8-atlas set, is the short id used by the builders; 10 rows have no URL, and 5 of them have a note in `notes`; the other 5 are `panel` rows with no recorded version, so download them by `dataset_id` from the page in `collection_url`, which serves the current version). `data/atlases.csv` is the list of the 10-atlas test set, `data/used_datasets.csv` the datasets used before the test. Download them with `./fetch_data.sh full DIR` (`reproduce.sh full` calls it; `data/download_atlases.sh` checks neither size nor md5). `full` downloads the 10 rows of set `test10` (29.3 GB) and checks each against its size, and against its md5 for the 3 rows that have one. The 8-atlas set (`test8`, 69 pools), which the `paper` and `rerun` tiers check, and the 10-atlas set (`test10`, 57 pools), which `full` and `data/build_pools.py` build, share no dataset or collection.

## Notebooks

`reproduce.sh paper` executes `notebooks/01_heldout_test.ipynb` (our method against Leiden subclustering and the in-house baselines), `02_published_detectors.ipynb` (published detectors, metrics used in the MARS and scCAD papers), `03_null_test.ipynb` (twin pools with the hidden cells removed: 101 real and twin pairs from 12 atlases, then the 69 pairs of the 8-atlas test set) and `04_part1_summary.ipynb` (the 88-dataset panel, an extra 31-type check, duplication as sample weighting). The notebooks only call `eval/paper.py`, `eval/nulltest.py` and `part1/`; they assert numbers and draw no figures.

The `paper` tier does not recompute:

- the 10-atlas results (the 10-atlas test, strata, separability tiers, label ablation, donor ablation, CHOIR and the tools that ran only there);
- scGAD on 32 pools;
- the random-ranking line of Fig. 1d;
- the Part I runs outside notebook 04 (generator training, the full data panel, the AIDA, HLCA and lupus runs). Notebook 04 lists what it covers.

The `rerun` tier rescores frozen candidate tables with this repository's scorer; it does not rebuild embeddings or Leiden partitions. Both tiers use the 69 pools of the 8-atlas test set (notebook 03 also reads the twins of 12 atlases). The 10-atlas test set (57 pools) is a different set; `full` (untested) produces its predictions.

## One pool

```
scripts/run_pool.sh POOL.h5ad PILOTS.parquet OUT_DIR
```
`POOL.h5ad` holds raw counts. `PILOTS.parquet` has columns `obs_name`, `label` and `donor`; the pool id is the parquet file name. Cell ids are opaque hashes of the barcode. Output: `OUT_DIR/preds/hiddenrare/<pool>.parquet` (`cell_id`, `score`), plus the Leiden and isolation-forest baselines and the intermediate tables in `OUT_DIR/cands`.

## Datasets used before the test

Datasets used before the test are listed in `data/used_datasets.csv`. A Human Breast Cell Atlas file from collection 48259aa8 was also used during development (row `hbca` of `ATLAS_MANIFEST.csv`); its dataset id was not recorded, so the cell-level leak check may not cover it. The collection itself is in the list (dataset 55003f67), so the collection check of `data/build_pools.py` excludes every dataset of that collection.

## Not included and known gaps

- CELLxGENE data (downloaded by the builders), the training pools and the tool checkouts. `hiddenrare/scorer.py fit` is the training script; it needs the training pools. Run it from the repository root (it imports eval/targets.py).
- The 10-atlas predictions: the 10-atlas results (strata, separability tiers, ablations, CHOIR) cannot be rerun from the release.
- Figure scripts; the notebooks assert numbers and draw nothing.
- The 69 pool h5ads of the 8-atlas test set, needed by `combination/`, Supp. Fig. S1 and any rescoring of those pools from counts. No builder here writes them (`data/build_pools.py` builds the 10-atlas pools), so there is no route to them yet.
- The inputs of the Part I runs and the training of the generators: the AIDA, HLCA and lupus h5ad files that `part1/` reads (`ATLAS_MANIFEST.csv` has no download URL for its `aida` and `hlca` rows) and the CFGen and scDiffusion checkouts (no commit recorded).
- Reproductions of the published tools; only their wrappers and pinned environments are included.
- RaceID and CHOIR, which ran from the authors' released code on the 10-atlas pools only: no wrapper, commit in tools/sources.txt or image entry is included for them.
- The `full` tier is untested, and the `recomb2027-tools` image has not been run end to end.
- The time and memory per pool (Supp., further details of the method; per-job `/usr/bin/time -l` logs on the run machine).
- The 101 twin pools of the null test (12 atlases): they ship only as predictions in `predictions.tgz`, and no tier rebuilds them.

## Citation

Please cite the paper. `CITATION.cff` holds the metadata; GitHub's "Cite this repository" button gives BibTeX. Until the paper is published:

```bibtex
@unpublished{peraltacastro2026hidden,
  author = {Peralta Castro, Fernando and Zhang, Jiaqi and Singh, Ritambhara},
  title  = {Finding Rare Cell Subtypes Hidden Inside Broad Labels},
  note   = {RECOMB 2027 submission. Code: https://github.com/fpera0248/recomb2027, release v0.1.1},
  year   = {2026}
}
```

## License

The code in this repository is under the MIT license (`LICENSE`). Third-party code keeps its own license:

- The published tools, cloned by `tools/fetch_sources.sh` at the commits in `tools/sources.txt`: scCAD and FiRE (GPL-3), GapClust (GPL-3, per its DESCRIPTION file), rflann (GPL), GiniClust3, SCISSORS, mars and scGPT (MIT), CellSIUS (Apache-2.0).
- The Python and R packages at the versions in `piplist_hiddenrare.txt`, `piplist_tools.txt` and `Rpackages_tools.csv`, among them leidenalg (GPL-3.0-or-later) and igraph (GPL-2.0-or-later).

The container images contain these programs under their own licenses. The `recomb2027-tools` image keeps the full source of each published tool in `/opt/tools_src`; the source of every Python and R package is available from PyPI, CRAN and Bioconductor at the version installed in the image, and the source of the operating-system packages from the Debian (`recomb2027-hiddenrare`, based on `python:3.12-slim-bookworm`) and Ubuntu (`recomb2027-tools`, based on `rocker/r-ver:4.6.1`) archives. On request, the authors (open an issue at https://github.com/fpera0248/recomb2027/issues) will provide the complete corresponding source of any GPL-licensed component of the images for three years after each image release.

The release assets are derived from the public CELLxGENE datasets listed in `data/used_datasets.csv` and the CELLxGENE rows of `ATLAS_MANIFEST.csv`, which CELLxGENE distributes under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The assets are under the same license; please cite the original datasets.
