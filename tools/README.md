## Published tools

Run the commands from the repository root.

`tools/sources.txt` lists the official repositories and commits; `tools/fetch_sources.sh` clones them into `$TOOLS_SRC`. The wrappers in `tools/` adapt input and output and turn each tool's output into a cell score (rule in each wrapper's header). `run_cellsius.R` also patches one test of the published CellSIUS function at run time (`class(mat.norm) != "matrix"` fails on R >= 4.2). CellSIUS has no result in the paper: it reached the 32 GB memory limit of the run machine. R tools need Seurat 4.4.0 / SeuratObject 4.1.4. The `recomb2027-tools` image (`docker/Dockerfile.tools`) installs all of this; `Rpackages_tools.csv` and `piplist_tools.txt` are the exact package sets of the environment the tools ran in.

Python tools run in their own Python 3.12 environment (set `TOOLS_PYTHON` for `scripts/run_tool.sh` and `reproduce.sh full`; R steps in `ENVIRONMENTS.md`, Manual build): `pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tools/requirements.txt; pip install --no-deps $TOOLS_SRC/GiniClust3 $TOOLS_SRC/scGPT`; build FiRE by copying `python/FiRE` of its checkout and `tools/fire_setup.py` (as `setup.py`) into `FIRE_BUILD` and running `python setup.py build_ext --inplace` there (Boost headers needed); set `SCGPT_CKPT` to the scGPT whole-human checkpoint dir and `MCL` to the mcl binary (CellSIUS; Debian and Ubuntu package `mcl`). `$TOOLS_SRC/scCAD/scCAD.py` must have md5 a0865c37513aaee5eae13551e48da6f2 (`tools/fetch_sources.sh` checks this). The reported scGPT run used `DEVICE=mps`.

`run_mars.py` gave the row the paper names MARS (our reimplementation): the MARS model code at the pinned commit with our preprocessing, distance score and 95th-percentile call.

`run_giniclust3.py` applies no gene or cell filter and uses 5 neighbors (the paper's GiniClust3 (no filters, 5 neighbors)); it does not regenerate the GiniClust3 (swapped filters) rows, whose runner is not included.
