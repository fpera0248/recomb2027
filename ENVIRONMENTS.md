# Environments

The code uses two environments, kept apart because their dependencies differ (Python 3.12 numerics for our method, a pinned R 4 / Seurat 4.4.0 stack plus PyTorch for the published tools).

| Environment | Used for | Runtime |
|---|---|---|
| `hiddenrare` | our method, evaluation, notebooks, builders, `reproduce.sh` | Python 3.12 |
| `tools` | published detectors run by `scripts/run_tool.sh` | Python 3.12 + R 4.6.1, Seurat 4.4.0 |

## Quick start

```
conda env create -f environment_hiddenrare.yml     # or: pip install -r requirements.txt
conda activate hiddenrare
./reproduce.sh paper
```
For the tools, use the container (`CONTAINER.md`) or follow `docker/Dockerfile.tools` step by step.

## Spec files

- `environment_hiddenrare.yml`, `requirements.txt`: the packages the code imports, pinned to the versions of the interpreter the results were computed with (macOS arm64). `piplist_hiddenrare.txt` is the dependency closure of `requirements.txt` in that interpreter.
- `environment_tools.yml`, `tools/requirements.txt`: the Python side of the tools, pinned to the virtual environment they ran in. `piplist_tools.txt` is its complete `pip list` (GiniClust3 and scGPT were installed from source with `--no-deps` and are not listed). PyTorch comes from the CPU wheel index in the container.
- `Rpackages_tools.csv`: the full set of R package versions (dependency closure of CellSIUS, GapClust, SCISSORS and Seurat) in the R 4.6.1 installation the tools ran in. GapClust and SCISSORS do not support Seurat 5, so Seurat 4.4.0 and SeuratObject 4.1.4 are required; the image pins these two and the tool packages. The other R packages come from CRAN and Bioconductor at build time. The `rocker/r-ver:4.6.1` base image sets CRAN to `https://p3m.dev/cran/__linux__/noble/latest`, so their versions are not pinned and can differ from `Rpackages_tools.csv`. Part I training code (CFGen, scDiffusion, scVI, scDesign3) is not covered by either environment.

Neither environment was created with conda, so there is no conda export with build strings. The tools and their commits are in `tools/sources.txt`.

## Manual build of the tools environment

```
python3 -m venv venv && venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tools/requirements.txt
TOOLS_SRC=tools_src tools/fetch_sources.sh
venv/bin/pip install --no-deps tools_src/GiniClust3 tools_src/scGPT
export FIRE_BUILD=$PWD/fire_build; mkdir -p $FIRE_BUILD; cp tools_src/FiRE/python/FiRE/* $FIRE_BUILD/; cp tools/fire_setup.py $FIRE_BUILD/setup.py
(cd $FIRE_BUILD && BOOST_INC=/usr/include ../venv/bin/python setup.py build_ext --inplace)      # Boost headers needed
# R packages: follow the RUN lines of docker/Dockerfile.tools (Seurat 4.4.0, SeuratObject 4.1.4, rflann with tools/Makevars.rflann, then CellSIUS, GapClust, SCISSORS)
```
Set `TOOLS_PYTHON` (`PYTHON` stays the hiddenrare environment), `RETICULATE_PYTHON`, `TOOLS_SRC`, `FIRE_BUILD`, `SCGPT_CKPT` and `MCL` as described in `scripts/run_tool.sh` and `tools/io.R`.
