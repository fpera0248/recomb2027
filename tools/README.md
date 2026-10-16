## Published tools

Run the commands from the repository root.

`tools/sources.txt` lists the official repositories and commits; `tools/fetch_sources.sh` clones them into `$TOOLS_SRC`. The wrappers in `tools/` only adapt input and output. R tools need Seurat 4.4.0 / SeuratObject 4.1.4. The `recomb2027-tools` image (`docker/Dockerfile.tools`) installs all of this; `Rpackages_tools.csv` and `piplist_tools.txt` are the exact package sets of the environment the tools ran in.

Python tools run in their own environment (set `TOOLS_PYTHON` for `scripts/run_tool.sh` and `reproduce.sh full`): `pip install -r tools/requirements.txt; pip install --no-deps $TOOLS_SRC/GiniClust3 $TOOLS_SRC/scGPT`; build FiRE by copying `python/FiRE` of its checkout and `tools/fire_setup.py` (as `setup.py`) into `FIRE_BUILD` and running `python setup.py build_ext --inplace` there (Boost headers needed); set `SCGPT_CKPT` to the scGPT whole-human checkpoint dir and `MCL` to the mcl binary. `$TOOLS_SRC/scCAD/scCAD.py` must have md5 a0865c37513aaee5eae13551e48da6f2 (`tools/fetch_sources.sh` checks this). The reported scGPT run used `DEVICE=mps`.
