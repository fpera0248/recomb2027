## Published tools

Run the commands from the repository root.

`tools/` holds the wrappers that run the published tools on a pool.

`tools/sources.txt` lists the official repositories and commits. `tools/fetch_sources.sh` clones them into `$TOOLS_SRC`. The wrappers only adapt input and output, except that `run_cellsius.R` patches one type test of the CellSIUS function in memory so it runs on R>=4.2 (`class(mat.norm) != "matrix"`).

`run_cellsius.R` on the 2,000 variable genes of `tools/adapt_pool.py` gave the CellSIUS (per label) scores of Fig. 3a. The later all-gene CellSIUS run of the supplement reached the 32 GB memory limit and has no result.

R tools need Seurat 4.4.0 / SeuratObject 4.1.4. The `recomb2027-tools` image (`docker/Dockerfile.tools`) installs the tool sources, these R packages, the Python packages, FiRE and mcl, but it does not include the scGPT checkpoint. `Rpackages_tools.csv` and `piplist_tools.txt` are the exact package sets of the environment the tools ran in.

Python tools run in their own Python 3.12 environment. The pins come from a macOS arm64 environment. Set `TOOLS_PYTHON` for `scripts/run_tool.sh` and `reproduce.sh full`. The pip, FiRE and R install steps are under Manual build in `ENVIRONMENTS.md`.

Set `SCGPT_CKPT` to the scGPT whole-human checkpoint dir and `MCL` to the mcl binary (Debian and Ubuntu package `mcl`), which CellSIUS uses. `$TOOLS_SRC/scCAD/scCAD.py` must have md5 a0865c37513aaee5eae13551e48da6f2 (`tools/fetch_sources.sh` checks this). The reported scGPT run used `DEVICE=mps`.

`run_giniclust3.py` applies no gene or cell filter and uses 5 neighbors, giving the paper's GiniClust3 (no filters, 5 neighbors). It does not regenerate the GiniClust3 (swapped filters) rows, whose runner is not included.

`run_giniclust3_readme.py` (tool `giniclust3_readme`) follows the README call sequence with its filters: genes seen in at least 3 cells, then cells with at least 200 genes, and 3 neighbors for the Gini clustering. It writes `giniclust3_readme` and `giniclust3_readme_lab`; cells the 200-gene filter removes score below every kept cell and are not called.

`run_raceid.R` and `run_choir.R` (tools `raceid` and `choir`) run RaceID (vignette settings for large data) and CHOIR (README quick start) on `py/` (genes detected in at least 3 cells; CHOIR's `min.cells = 5` then gives the gene set of the paper run); `run_lab.py` derives their per-label arms (`raceid_lab`, `choir_lab`) from the clusters. `install_raceid_choir.R raceid|choir` installs them from the `tools/fetch_sources.sh` checkouts. CHOIR needs Seurat >= 5 (the paper's CHOIR ran on Seurat 5.5.1 and 5.6.0), so it goes into its own library and `scripts/run_tool.sh` runs it with `CHOIR_RSCRIPT`; it needs mrtree with `tools/mrtree.patch` applied, which drops two imports that only plotting and simulation use. CHOIR took minutes to hours per subsample (no time cap in the paper run).

`run_gapclust_official.R` and `run_scissors_official.R` (tools `gapclust_official`, `scissors_official`) are the runs behind the paper's GapClust (default and ranked by rare scores) and SCISSORS rows: GapClust on CP10K counts without the log, SCISSORS with `ReclusterCells(auto = TRUE)` and `IntegrateSubclusters`, both on `py/`, with the call rules in the file headers. They write `gapclust_official`, `gapclust_official_adapter`, `scissors_official` and `scissors_official_lab`. `run_gapclust.R` and `run_scissors.R` (2,000 HVG input) are the earlier runs.

`run_fire.py` (via `scripts/run_tool.sh fire`) writes the within-label percentile of the FiRE score, the paper's FiRE (per label) row, to `tool_out/fire_lab`. In `tool_predictions.tgz` that score is `fire_lab_pct`, and `fire_lab` holds the raw within-label score, so a new run's `fire_lab` corresponds to the release's `fire_lab_pct`.
