# Finding rare cell subtypes hidden inside broad labels

Code for our RECOMB 2027 submission, "Duplication, Generators and Hidden Rare Cell Subtypes". Until the paper appears, [CITATION.cff](CITATION.cff) holds its metadata. Given a pool of cells with a coarse visible label (say "T cell"), our method ranks cells by how likely each belongs to a finer, unlabelled subtype. It never sees the finer annotation. Use release v0.1.2.

## Quick check

```
pip install -r requirements.txt    # Python 3.12
./reproduce.sh paper
```

- Download: 1.2 GB of release assets, about 2.5 GB on disk.
- Time: about 4 minutes plus the download (measured on a laptop, 4 threads).
- Success: the run prints `paper: all notebooks ran and every assertion held`.
- The `paper` level recomputes numbers from the saved predictions and result tables. It does not refit the model. `./reproduce.sh rerun` applies a refit scorer as a consistency check, so its numbers differ a little. The container image is `v0.1.2`, and the `paper` and `rerun` levels pass in it. Reproduction problems: https://github.com/fpera0248/recomb2027/issues. More checks are in [REPRODUCE.md](REPRODUCE.md#quick-check-details).

## Paper items and code

| Paper item | Code | Status |
|---|---|---|
| Fig. 1, duplication study and union AP of the methods | notebooks 01, 02 and 04 | partial |
| Fig. 2, atlas-equal means and per-atlas differences | notebooks 01 and 02 | partial |
| Fig. 3, separability levels and UMAPs | none | not in repo |
| Supplement (figures, tables and analyses) | see [PAPER_ITEMS.md](PAPER_ITEMS.md) | reproduced, partial, recomputed with another readout or not in repo |

The method lives in `hiddenrare/`. [REPRODUCE.md](REPRODUCE.md#layout) maps the other folders.

## Install

Python 3.12:

```
pip install -r requirements.txt
```

Or the container:

```
docker pull ghcr.io/fpera0248/recomb2027-hiddenrare:v0.1.2
```

See [ENVIRONMENTS.md](ENVIRONMENTS.md) for the Python pins and [CONTAINER.md](CONTAINER.md) for the images and Apptainer.

## Reproduce

```
./reproduce.sh paper
```

This downloads the release assets, recomputes the Part II numbers of the 8-atlas test set plus the Part I panel and extra checks, and asserts the match. A mismatch with any asserted value fails the run. The `rerun` and `full` levels, the data, releases, notebooks and known gaps are in [REPRODUCE.md](REPRODUCE.md).

In the container:

```
docker run -v "$PWD/out":/out ghcr.io/fpera0248/recomb2027-hiddenrare:v0.1.2 \
    env ASSETS=/out/assets OUT=/out/reproduce_out reproduce paper
```

## Data

[`ATLAS_MANIFEST.csv`](ATLAS_MANIFEST.csv) lists every CELLxGENE dataset we used. It has 123 rows for 122 datasets, because AIDA appears in two sets. Each row gives the download URL, size and md5 where we recorded them. The `set` column maps to the paper as follows.

| `set` | Rows | Size | In the paper |
|---|---|---|---|
| `test8` | 8 | 22.3 GB | Hidden-subtype study: the 8-atlas test set (69 subsamples) |
| `test10` | 10 | 29.3 GB | Hidden-subtype study: the 10-atlas test set (57 subsamples) |
| `panel` | 90 | 443.7 GB | Duplication study: the dataset panel, including AIDA and lupus. Four rows are also the extra atlases of the 12-atlas null test |
| `part1` | 2 | not recorded | Duplication study: the AIDA and HLCA files, with no download URL |
| `development` | 10 | 90.3 GB | Training and tuning before the test. Three rows are also in the duplication-study panel |
| `analysis` | 3 | 1.0 GB | Other datasets: the chromatin accessibility pilot and two biology checks |

The duplication-study panel queued 93 datasets (90 `panel` rows and 3 `development` rows). The 88 that pass the preflight check form the panel.

What each level downloads:

| Level | Downloads | Cells |
|---|---|---|
| `paper` | 1.2 GB of release assets: the 69 subsample definitions of the 8-atlas test set, the saved scores of every method and the duplication-study result tables | The definitions list 1,380,000 cells by id, label and donor. They hold no expression counts |
| `rerun` | 1.05 GB of release assets: the same definitions, the saved scores of our method and the baselines, and the frozen candidate tables | The same 1,380,000 cells, without counts |
| `full` | The 10 `test10` atlases from CELLxGENE (29.3 GB) | 2.31 million cells. `full` builds 57 subsamples of 20,000 cells (1,140,000 cells) and reruns our method, Leiden subclustering and, with `TOOLS=1`, the published tools. We have not run it end to end |

No level downloads the other sets. The 8-atlas results are recomputed from saved scores, not rebuilt from counts, because no script builds its 69 subsamples from the atlases. To use any other row, download it from its URL (see [REPRODUCE.md](REPRODUCE.md#atlases)).

## Citation

Please cite the paper. `CITATION.cff` holds the metadata, and GitHub's "Cite this repository" button gives BibTeX. Until the paper is published:

```bibtex
@unpublished{peraltacastro2026hidden,
  author = {Peralta Castro, Fernando and Zhang, Jiaqi and Singh, Ritambhara},
  title  = {Duplication, Generators and Hidden Rare Cell Subtypes},
  note   = {RECOMB 2027 submission. Code: https://github.com/fpera0248/recomb2027, release v0.1.2},
  year   = {2026}
}
```

## License

The code is under the MIT license ([LICENSE](LICENSE)). Third-party code and the data keep their own licenses, listed in [REPRODUCE.md](REPRODUCE.md#license-details).
