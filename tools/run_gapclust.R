# GapClust (Fa 2021), https://github.com/fabotao/GapClust commit bdec2ad21a8791b85315588d94e2e5537def146f.
# usage: Rscript run_gapclust.R pool.h5ad out.parquet
# Unsupervised (the tool takes no labels), so no label-aware option. Defaults: k=200. Input = CP10K-normalized counts (genes x cells);
# the package applies log2(x+1) itself.
a <- commandArgs(TRUE)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "io.R"))
suppressMessages({
  library(Seurat)
  library(GapClust)
})
p <- read_pool(a[1])
N <- ncol(p$counts)
X <- cp10k_log(p$counts)
X@x <- expm1(X@x)                 # CP10K without the log (undo the log1p of the helper)
set.seed(0)
res <- GapClust(X, k = 200)
call <- rep(FALSE, N)
score <- rep(0, N)
if (is.list(res)) {
  S <- res$rare_score
  score <- apply(S, 1, max, na.rm = TRUE)             # strongest gap statistic over all minor-cluster sizes
  idx <- unlist(res$rare_cell_indices)
  call[idx] <- TRUE
  score[call] <- score[call] + max(score) + 1                               # cells inside detected rare groups rank first
}
write_out(colnames(p$counts), score, call, a[2])
cat("n_called", sum(call), "\n")
