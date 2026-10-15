# SCISSORS (Leary), https://github.com/jr-leary7/SCISSORS commit 79001b24001bba4fb9f45a8795707805468ebb5a.
# usage: Rscript run_scissors.R pool.h5ad out.parquet [cluster|label]
# cluster (default, published workflow): PrepareData, broad Louvain clusters, then ReclusterCells on the clusters auto = TRUE would pick (mean silhouette < .25).
# label: obs['label'] replaces the broad clusters and every label is reclustered (label-aware option; SCISSORS takes a clustering as input).
# Settings = the package README example (n.HVG 4000, n.PC 15, k 30/40/50, res .2/.3/.4, cutoff.score .25 default); use.parallel = FALSE. Needs Seurat 4.4.0 (pre-5 slots).
# score = -log(fraction of pool in the cell's final (sub)cluster) + 0.01 * (1 - cosine silhouette in PCA); call = final (sub)cluster < 5% of pool.
a <- commandArgs(TRUE)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "io.R"))
suppressMessages({
  library(Seurat)
  library(SCISSORS)
  library(dplyr)
})
p <- read_pool(a[1])
mode <- if (length(a) > 2) a[3] else "cluster"
s <- CreateSeuratObject(p$counts)
s <- PrepareData(s, n.HVG = 4000, n.PC = 15, which.dim.reduc = "umap", initial.resolution = 0.4, use.parallel = FALSE, random.seed = 629)   # README example values (n.PC = "auto" picks 1 PC on small pools)
if (mode == "label") {
  s$seurat_clusters <- factor(as.integer(factor(p$label[colnames(s)])) - 1L)
  Idents(s) <- s$seurat_clusters
}
clusters <- if (mode == "label") as.integer(levels(s$seurat_clusters)) else as.integer(names(which(ComputeSilhouetteScores(s) < 0.25)))   # = auto = TRUE, cutoff.score .25
# one ReclusterCells call per cluster (same result as a multi-cluster call, which returns the same list of objects)
# README ReclusterCells values; called without do.call (do.call inlines the Seurat object into the call and deparsing that call hangs)
res <- list()
for (w in clusters) {
  res[[length(res) + 1]] <- ReclusterCells(s, which.clust = w, n.HVG = 4000, n.PC = 15, k.vals = c(30, 40, 50),
                                           resolution.vals = c(.2, .3, .4), use.parallel = FALSE, random.seed = 312)
}
final <- setNames(as.character(s$seurat_clusters), colnames(s))
for (i in seq_along(res)) {                                    # relabel cells of reclustered parents as parent.sub
  o <- res[[i]]
  cl <- colnames(o)
  final[cl] <- paste0(final[cl], ".", as.character(o$seurat_clusters))
}
frac <- as.numeric(table(final)[final]) / length(final)
sil <- cluster::silhouette(dist = CosineDist(Embeddings(s, "pca")), x = as.integer(factor(final)))
score <- -log(frac) + 0.01 * (1 - sil[, 3])
call <- frac < 0.05
write_out(names(final), score, call, a[2])
cat("n_final_clusters", length(unique(final)), "n_called", sum(call), "\n")
