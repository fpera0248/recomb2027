# GapClust (Fa 2021), https://github.com/fabotao/GapClust commit bdec2ad21a8791b85315588d94e2e5537def146f: the run behind the paper's GapClust rows.
# usage: Rscript run_gapclust_official.R pool.h5ad out_call.parquet out_adapter.parquet
# GapClust(X, k = 200) on the full CP10K-normalised genes x cells matrix without the log (the package applies log2(x+1) and its own vst gene selection). Needs Seurat 4.4.0 and rflann built with tools/Makevars.rflann.
# out_call: score = the tool's own call (rare_cell_indices) as 0/1, call = the same. out_adapter: score = per-cell max of the tool's rare_score columns (our ranking adapter), call as above.
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
adapter <- rep(0, N)
if (is.list(res)) {
  adapter <- apply(res$rare_score, 1, max, na.rm = TRUE)
  adapter[!is.finite(adapter)] <- 0
  call[unlist(res$rare_cell_indices)] <- TRUE
}
write_out(colnames(p$counts), as.numeric(call), call, a[2])
write_out(colnames(p$counts), adapter, call, a[3])
cat("n_called", sum(call), "\n")
