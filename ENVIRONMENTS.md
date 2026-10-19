# Environments

The pins of the `hiddenrare` environment (`requirements.txt`, `environment_hiddenrare.yml`) are the package versions of the Python 3.12 interpreter (macOS arm64) that reproduces the paper's 10-atlas predictions exactly. Run on the rebuilt pools, on the machine that computed the results, it matched the saved predictions on all 57 pools. nbconvert and nbformat only run the notebooks and were not installed in it. With the pins of v0.1.2 (numba 0.61.0, pynndescent 0.5.13, h5py 3.12.1, scikit-misc 0.5.1) the predictions differed. That run used scipy 1.18.1, because the pinned scipy 1.15.1 does not import on that machine. scikit-learn must be 1.5.2, because the model pickle was written with it. Leiden partitions depend on the numerical stack, so per-cell scores differ on another machine. This repository does not measure by how much. On another machine, rerunning reproduces the pipeline and the evaluation, not the paper's exact per-cell scores.

The code uses two environments, kept apart because their dependencies differ. Our method needs Python 3.12 and the numerical packages pinned in `requirements.txt`. The published tools need a pinned R 4 / Seurat 4.4.0 stack plus PyTorch.

| Environment | Used for | Runtime |
|---|---|---|
| `hiddenrare` | our method, evaluation, notebooks, builders, `reproduce.sh` | Python 3.12 |
| `tools` | published detectors run by `scripts/run_tool.sh` | Python 3.12 + R 4.6.1, Seurat 4.4.0 |

## Quick start

```
conda env create -f environment_hiddenrare.yml
conda activate hiddenrare
./reproduce.sh paper
```
Without conda, run `pip install -r requirements.txt` in a Python 3.12 environment in place of the first two lines.
For the tools, use the container (`CONTAINER.md`) or follow `docker/Dockerfile.tools` step by step.

## Spec files

- `environment_hiddenrare.yml`, `requirements.txt`: the packages the code imports, pinned to the versions of that interpreter (macOS arm64), except the notebook packages nbconvert and nbformat. `piplist_hiddenrare.txt` is the dependency closure of `requirements.txt` in a fresh install with these pins. Its 84 packages that the interpreter also has carry the interpreter's versions. The other 17 are nbconvert, nbformat and their dependencies.
- `environment_tools.yml`, `tools/requirements.txt`: the Python side of the tools, pinned to the virtual environment they ran in. `piplist_tools.txt` is its complete `pip list` (macOS arm64, Python 3.12.9). GiniClust3 and scGPT were installed from source with `--no-deps` and are not listed. PyTorch comes from the CPU wheel index in the container.
- `Rpackages_tools.csv`: the full set of R package versions (dependency closure of CellSIUS, GapClust, SCISSORS and Seurat) in the R 4.6.1 installation the tools ran in. GapClust and SCISSORS do not support Seurat 5, so Seurat 4.4.0 and SeuratObject 4.1.4 are required. The image pins these two and the tool packages. The other R packages come from CRAN and Bioconductor at build time. The `rocker/r-ver:4.6.1` base image sets CRAN to `https://p3m.dev/cran/__linux__/noble/latest`, so their versions are not pinned and can differ from `Rpackages_tools.csv`.

Part I training code (CFGen, scDiffusion, scVI, scDesign3) is not covered by either environment. Neither environment was created with conda, so there is no conda export with build strings. The tools and their commits are in `tools/sources.txt`.

## Manual build of the tools environment

```
python3 -m venv venv && venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tools/requirements.txt
TOOLS_SRC=tools_src tools/fetch_sources.sh
venv/bin/pip install --no-deps tools_src/GiniClust3 tools_src/scGPT
export FIRE_BUILD=$PWD/fire_build; mkdir -p $FIRE_BUILD; cp tools_src/FiRE/python/FiRE/* $FIRE_BUILD/; cp tools/fire_setup.py $FIRE_BUILD/setup.py
(cd $FIRE_BUILD && BOOST_INC=/usr/include ../venv/bin/python setup.py build_ext --inplace)      # Boost headers needed
# R packages: follow the RUN lines of docker/Dockerfile.tools (Seurat 4.4.0, SeuratObject 4.1.4, rflann with tools/Makevars.rflann, then CellSIUS, GapClust, SCISSORS)
# MCL (CellSIUS): install the mcl binary (Debian and Ubuntu package mcl, as in docker/Dockerfile.tools) and set MCL to its path
```
Set `TOOLS_PYTHON` (`PYTHON` stays the hiddenrare environment), `RETICULATE_PYTHON`, `TOOLS_SRC`, `FIRE_BUILD`, `SCGPT_CKPT` and `MCL` as described in `scripts/run_tool.sh` and `tools/io.R`.
