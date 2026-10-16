## Published tools

Run the commands from the repository root.

`tools/` holds the wrappers that run the published tools on a pool.

`tools/sources.txt` lists the official repositories and commits; `tools/fetch_sources.sh` clones them into `$TOOLS_SRC`. The wrappers only adapt input and output, except that `run_cellsius.R` patches one type test of the CellSIUS function in memory so it runs on R>=4.2 (`class(mat.norm) != "matrix"`).

`run_cellsius.R` on the 2,000 variable genes of `tools/adapt_pool.py` gave the CellSIUS (per label) scores of Fig. 3a. The later all-gene CellSIUS run of the supplement reached the 32 GB memory limit and has no result.

R tools need Seurat 4.4.0 / SeuratObject 4.1.4. The `recomb2027-tools` image (`docker/Dockerfile.tools`) installs all of this. `Rpackages_tools.csv` and `piplist_tools.txt` are the exact package sets of the environment the tools ran in.

Python tools run in their own Python 3.12 environment; the pins come from a macOS arm64 environment. Set `TOOLS_PYTHON` for `scripts/run_tool.sh` and `reproduce.sh full`. Install the Python packages with:

```
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tools/requirements.txt
pip install --no-deps $TOOLS_SRC/GiniClust3 $TOOLS_SRC/scGPT
```

Build FiRE by copying `python/FiRE` of its checkout and `tools/fire_setup.py` (as `setup.py`) into `FIRE_BUILD`, then run `python setup.py build_ext --inplace` there (Boost headers needed).

Set `SCGPT_CKPT` to the scGPT whole-human checkpoint dir and `MCL` to the mcl binary (CellSIUS; Debian and Ubuntu package `mcl`). `$TOOLS_SRC/scCAD/scCAD.py` must have md5 a0865c37513aaee5eae13551e48da6f2 (`tools/fetch_sources.sh` checks this). The reported scGPT run used `DEVICE=mps`. For the R steps see `ENVIRONMENTS.md`, Manual build.

`run_giniclust3.py` applies no gene or cell filter and uses 5 neighbors. This is the paper's GiniClust3 (no filters, 5 neighbors). It does not regenerate the GiniClust3 (swapped filters) rows, whose runner is not included.
