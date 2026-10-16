## Published tools

Run the commands from the repository root.

`tools/` holds the wrappers that run the published tools on a pool.

`tools/sources.txt` lists the official repositories and commits; `tools/fetch_sources.sh` clones them into `$TOOLS_SRC`. The wrappers only adapt input and output, except that `run_cellsius.R` patches one type test of the CellSIUS function in memory so it runs on R>=4.2 (`class(mat.norm) != "matrix"`).

`run_cellsius.R` on the 2,000 variable genes of `tools/adapt_pool.py` gave the CellSIUS (per label) scores of Fig. 3a. The later all-gene CellSIUS run of the supplement reached the 32 GB memory limit and has no result.

R tools need Seurat 4.4.0 / SeuratObject 4.1.4. The `recomb2027-tools` image (`docker/Dockerfile.tools`) installs the tool sources, these R packages, the Python packages, FiRE and mcl; it does not include the scGPT checkpoint. `Rpackages_tools.csv` and `piplist_tools.txt` are the exact package sets of the environment the tools ran in.

Python tools run in their own Python 3.12 environment; the pins come from a macOS arm64 environment. Set `TOOLS_PYTHON` for `scripts/run_tool.sh` and `reproduce.sh full`. The pip, FiRE and R install steps are under Manual build in `ENVIRONMENTS.md`.

Set `SCGPT_CKPT` to the scGPT whole-human checkpoint dir and `MCL` to the mcl binary (CellSIUS; Debian and Ubuntu package `mcl`). `$TOOLS_SRC/scCAD/scCAD.py` must have md5 a0865c37513aaee5eae13551e48da6f2 (`tools/fetch_sources.sh` checks this). The reported scGPT run used `DEVICE=mps`.

`run_giniclust3.py` applies no gene or cell filter and uses 5 neighbors, giving the paper's GiniClust3 (no filters, 5 neighbors). It does not regenerate the GiniClust3 (swapped filters) rows, whose runner is not included.

`run_fire.py` (via `scripts/run_tool.sh fire`) writes the within-label percentile of the FiRE score, the paper's FiRE (per label) row, to `tool_out/fire_lab`. In `tool_predictions.tgz` that score is `fire_lab_pct`, and `fire_lab` holds the raw within-label score, so a new run's `fire_lab` corresponds to the release's `fire_lab_pct`.
