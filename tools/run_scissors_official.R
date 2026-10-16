# SCISSORS (Leary), https://github.com/jr-leary7/SCISSORS commit 79001b24001bba4fb9f45a8795707805468ebb5a: the run behind the paper's SCISSORS rows.
# usage: Rscript run_scissors_official.R pool.h5ad out.parquet
# README workflow: PrepareData -> ReclusterCells(auto = TRUE) -> IntegrateSubclusters. Needs Seurat 4.4.0. merge.clusters stays at the package default (FALSE).
# IntegrateSubclusters (as published) re-folds the largest subcluster of every earlier reclustering on each pass; not patched.
# SCISSORS has no rare-cell output. Adapter (ours): final cluster = seurat_clusters after IntegrateSubclusters, score = 1 - share, call = share <= 10%. group = final cluster (feeds the per-label arm, tools/run_lab.py).
a <- commandArgs(TRUE)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "io.R"))
suppressMessages({
  library(Seurat)
  library(SCISSORS)
})
p <- read_pool(a[1])
s <- CreateSeuratObject(p$counts)
s <- PrepareData(s, n.HVG = 4000, n.PC = 15, which.dim.reduc = "umap", initial.resolution = 0.4, use.parallel = FALSE, random.seed = 629)
res <- ReclusterCells(s, auto = TRUE, n.HVG = 4000, n.PC = 15, k.vals = c(30, 40, 50), resolution.vals = c(.2, .3, .4), use.parallel = FALSE, random.seed = 312)
if (!is.list(res)) res <- list(res)   # no candidate cluster -> empty list (ReclusterCells.R)
s <- IntegrateSubclusters(s, reclust.results = res)
final <- setNames(as.character(s$seurat_clusters), colnames(s))
share <- as.numeric(table(final)[final]) / length(final)
write_out(names(final), 1 - share, share <= 0.10, a[2], final)
cat("n_called", sum(share <= 0.10), "\n")
