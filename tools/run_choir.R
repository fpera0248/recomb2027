# CHOIR (Sant 2025), https://github.com/corceslab/CHOIR commit e9ebfbc9089beeaf4ca088c7b81b18f39758b0bc (needs mrtree 720f469, tools/install_raceid_choir.R).
# usage: Rscript run_choir.R pool.h5ad out.parquet [cores]       (cores default 4)
# README quick start: CreateSeuratObject(counts, min.cells = 5) -> NormalizeData() -> CHOIR(object, n_cores); final clusters = CHOIR_clusters_0.05 (buildTree + pruneTree, alpha 0.05, Bonferroni).
# min.features is not applied (the vignette's 100 would drop cells we must score). Unsupervised: no labels are read.
# CHOIR returns clusters only, so score = 1 - cluster size / pool size and call = score >= 0.9 (cluster of at most 10% of the pool), our size rule. group = CHOIR cluster (feeds tools/run_lab.py).
a <- commandArgs(TRUE)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "io.R"))
suppressMessages({
  library(Seurat)
  library(CHOIR)
})
p <- read_pool(a[1])
nc <- if (length(a) > 2) as.integer(a[3]) else 4L
object <- CreateSeuratObject(p$counts, min.cells = 5)
object <- NormalizeData(object, verbose = FALSE)
stopifnot(ncol(object) == ncol(p$counts))
object <- CHOIR(object, n_cores = nc)
cl <- object@meta.data[["CHOIR_clusters_0.05"]]
stopifnot(length(cl) == ncol(p$counts), !anyNA(cl))
cl <- setNames(as.character(cl), colnames(object))[colnames(p$counts)]
score <- 1 - as.numeric(table(cl)[cl]) / length(cl)
write_out(colnames(p$counts), score, score >= 0.9, a[2], cl)
cat("n_clusters", length(unique(cl)), "n_called", sum(score >= 0.9), "\n")
