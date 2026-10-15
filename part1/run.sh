#!/bin/bash
# Part I commands, run from this directory. Not executed by reproduce.sh: they need the AIDA, HLCA and lupus
# h5ad files, CFGen/ and the scDiffusion checkout, and take days on CPU. Smoke runs: add --smoke to the generator scripts.
export AIDA_DIR=${AIDA_DIR:?dir with AIDA_AbundantPool_120k.h5ad and AIDA_RarePool.h5ad}
export SCDIFFUSION_DIR=${SCDIFFUSION_DIR:?scDiffusion checkout}
export RSCRIPT=${RSCRIPT:-Rscript}          # R env with scDesign3; MEM_PATCH is the memory patch (default mem_patch.R, i.e. part1/mem_patch.R); scdesign3.R sets SD3_MEM=1, which turns it on

# AIDA head-to-head of all generators and duplication
python3 aida.py

# statistical augmentation conditions on the same donor-disjoint split
python3 split_cache.py 0 20
python3 statistical.py run 0 20
python3 statistical.py report
python3 dup_weight.py
HVG_TRAIN=1 python3 split_cache.py 0 20                        # training-only gene selection
REP=splittrain CLF=lr,knn,rf,mlp TAG=_trhvg python3 statistical.py run 0 20
REP=splittrain CLF=lr,knn,rf,mlp TAG=_trhvg python3 statistical.py report
P31=1 HVG_TRAIN=1 python3 split_cache.py 0 20                  # 31-type run
REP=split31train CLF=lr,knn,rf,mlp TAG=_p31 python3 statistical.py run 0 20
CLF=lr,knn,rf,mlp TAG=_clf python3 statistical.py run 0 20     # four classifiers

# second and third datasets
SAVE_GENE=hlca_genes H5AD=${HLCA_H5AD:?} python3 hlca.py
GENE_DIR=hlca_genes READOUT_CSV=hlca_clf.csv python3 panel_readout.py run
GENE_DIR=hlca_genes READOUT_CSV=hlca_clf.csv python3 panel_readout.py report
H5AD=${LUPUS_H5AD:?} python3 lupus.py

# CELLxGENE panel: CL_OBO=<cl-basic.obo> python3 panel_census.py pull|report builds panel_eligible.csv (shipped here)
export PANEL_DATA=${PANEL_DATA:-panel_data}
for ds in $(cat panel_queue.txt); do ./panel_run.sh $ds || true; done
PANEL_DIR=panel_runs PANEL_OUT=panel python3 panel_analysis.py
