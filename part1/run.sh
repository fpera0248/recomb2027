#!/bin/bash
# Part I commands, run from this directory. Not executed by reproduce.sh: they need the AIDA, HLCA and lupus
# h5ad files, a GPU, CFGen/ and the scDiffusion checkout, and take days. Smoke runs: add --smoke to the generator scripts.
export AIDA_DIR=${AIDA_DIR:?dir with AIDA_AbundantPool_120k.h5ad and AIDA_RarePool.h5ad}
export SCDIFFUSION_DIR=${SCDIFFUSION_DIR:?scDiffusion checkout}
export RSCRIPT=${RSCRIPT:-Rscript}          # R env with scDesign3; MEM_PATCH=<mem_patch.R> must point at the memory patch

# AIDA head-to-head of all generators and duplication
python3 aida.py

# statistical augmentation conditions on the same donor-disjoint split
python3 split_cache.py 0 20
python3 statistical.py run 0 20
python3 statistical.py report
python3 dup_weight.py

# second and third datasets
H5AD=$HLCA_H5AD python3 hlca.py
H5AD=$LUPUS_H5AD python3 lupus.py

# CELLxGENE panel: CL_OBO=<cl-basic.obo> python3 panel_census.py pull|report builds panel_eligible.csv (shipped here)
export PANEL_DATA=${PANEL_DATA:-panel_data}
for _ in $(cat panel_queue_shuffled.txt); do ./panel_run.sh || true; done   # each call takes the next unclaimed dataset
PANEL_DIR=panel_runs PANEL_OUT=panel python3 panel_analysis.py
