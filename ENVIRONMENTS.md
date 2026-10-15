# Environments

Two environments. Their dependencies differ (Python 3.12 numerics for our method, a pinned R 4 / Seurat 4.4.0 stack plus PyTorch for the published tools), so they are kept apart.

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
- `Rpackages_tools.csv`: R package versions (dependency closure of CellSIUS, GapClust, SCISSORS and Seurat) in the R 4.6.1 installation the tools ran in. Seurat 4.4.0 and SeuratObject 4.1.4 are required because GapClust and SCISSORS do not support Seurat 5. The image pins only these two and the tool packages (it uses the R 4.6.1 base image's dated CRAN snapshot for the rest); the CSV is the record of the full set. Part I training code (CFGen, scDiffusion, scVI, scDesign3) is not covered by either environment.

There is no fully pinned conda export: neither environment was created with conda, so no build strings exist to record. The tools and their commits are in `tools/sources.txt`.

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
