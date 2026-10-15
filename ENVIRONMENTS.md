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

- `environment_hiddenrare.yml`, `requirements.txt`: the packages the code imports, pinned to the versions of the interpreter the results were computed with (macOS arm64). `piplist_hiddenrare.txt` is that interpreter's complete `pip list`.
- `environment_tools.yml`, `tools/requirements.txt`: the Python side of the tools, pinned to the virtual environment they ran in. `piplist_tools.txt` is its complete `pip list` (GiniClust3 was installed from source with `--no-deps` and is not listed). PyTorch comes from the CPU wheel index in the container.
- `Rpackages_tools.csv`: R package versions (dependency closure of CellSIUS, GapClust, SCISSORS and Seurat) in the R 4.6.1 installation the tools ran in. Seurat 4.4.0 and SeuratObject 4.1.4 sit in a separate library placed first on `.libPaths()`; the default library holds Seurat 5, which GapClust and SCISSORS do not support.

There is no fully pinned conda export: neither environment was created with conda, so no build strings exist to record. The tools and their commits are in `tools/sources.txt`.

## Manual build of the tools environment

```
python3 -m venv venv && venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tools/requirements.txt
TOOLS_SRC=tools_src tools/fetch_sources.sh
venv/bin/pip install --no-deps tools_src/GiniClust3 tools_src/scGPT
(cd tools_src/FiRE && ../../venv/bin/python setup.py build_ext --inplace)      # Boost headers needed
Rscript -e 'remotes::install_version("SeuratObject", "4.1.4"); remotes::install_version("Seurat", "4.4.0")'
for p in CellSIUS GapClust SCISSORS; do R CMD INSTALL tools_src/$p; done
```
Set `PYTHON`, `RETICULATE_PYTHON`, `TOOLS_SRC`, `FIRE_BUILD`, `SCGPT_CKPT` and `MCL` as described in `scripts/run_tool.sh` and `tools/io.R`.
