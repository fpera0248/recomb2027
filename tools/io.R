# shared IO for the R runners: h5ad via reticulate + anndata, parquet via pandas
# Env: RETICULATE_PYTHON = python with anndata, scipy and pandas; R_EXTRA_LIBS = colon-separated library paths searched first
# (GapClust and SCISSORS read Seurat v3/v4 slots, so put Seurat 4.4.0 / SeuratObject 4.1.4 there).
if (nzchar(Sys.getenv("R_EXTRA_LIBS"))) .libPaths(c(strsplit(Sys.getenv("R_EXTRA_LIBS"), ":")[[1]], .libPaths()))
suppressMessages(library(Matrix))
reticulate::use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
read_pool <- function(path) {
  ad <- reticulate::import("anndata")
  sp <- reticulate::import("scipy.sparse")
  a <- ad$read_h5ad(path)
  X <- as(sp$csc_matrix(a$X), "CsparseMatrix")            # cells x genes, raw counts
  cells <- as.character(a$obs_names$to_list())
  genes <- as.character(a$var_names$to_list())
  counts <- t(X)
  dimnames(counts) <- list(genes, cells)    # genes x cells
  list(counts = counts, label = setNames(as.character(a$obs[["label"]]), cells))
}
write_out <- function(cell_id, score, call, path, group = NULL) {   # group (optional): cluster id per cell, kept for the per-label arm
  pd <- reticulate::import("pandas", convert = FALSE)
  l <- list(cell_id = as.list(cell_id), score = as.list(as.numeric(score)), call = as.list(as.logical(call)))
  if (!is.null(group)) l$group <- as.list(as.character(group))
  df <- pd$DataFrame(reticulate::r_to_py(l))
  df$to_parquet(path, index = FALSE)
}
cp10k_log <- function(counts, base = exp(1)) {              # CP10K then log(x+1) (base selectable)
  s <- Matrix::colSums(counts)
  s[s == 0] <- 1
  X <- counts %*% Matrix::Diagonal(x = 1e4 / s)
  X@x <- log1p(X@x) / log(base)
  dimnames(X) <- dimnames(counts)
  X
}
