# CellSIUS (Wegmann 2019), https://github.com/Novartis/CellSIUS commit 3c88967128dbc427015e909652aa44c64924ea5a.
# usage: Rscript run_cellsius.R pool.h5ad out.parquet [label|cluster]
# groups=label (default): coarse obs['label'] is the CellSIUS group_id input (label-aware; the tool itself takes a clustering).
# groups=cluster: Seurat default Louvain clusters (unsupervised input). CellSIUS defaults otherwise.
a <- commandArgs(TRUE); source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "io.R"))
suppressMessages({library(CellSIUS); library(data.table)})
a[1] <- normalizePath(a[1]); a[2] <- file.path(normalizePath(dirname(a[2])), basename(a[2])); setwd(tempdir())   # CellSIUS writes tmp.mcl.* in cwd
p <- read_pool(a[1]); mode <- if (length(a) > 2) a[3] else "label"
norm <- as.matrix(cp10k_log(p$counts, base = 2))                 # log2(CP10K+1), dense "matrix" as the tool requires
if (mode == "label") grp <- p$label else {
  suppressMessages({library(Seurat)}); s <- CreateSeuratObject(p$counts)
  s <- NormalizeData(s, verbose = FALSE); s <- FindVariableFeatures(s, verbose = FALSE); s <- ScaleData(s, verbose = FALSE)
  s <- RunPCA(s, verbose = FALSE); s <- FindNeighbors(s, dims = 1:10, verbose = FALSE); s <- FindClusters(s, verbose = FALSE)
  grp <- setNames(as.character(Idents(s)), colnames(p$counts))
}
# R>=4.2 errors on `if (class(mat)!="matrix")` (length-2 class) in the published code; patch only that test, in memory.
CellSIUS_fn <- CellSIUS::CellSIUS
body(CellSIUS_fn) <- parse(text = sub('class(mat.norm) != "matrix"', '!is.matrix(mat.norm)', paste(deparse(body(CellSIUS_fn)), collapse = "\n"), fixed = TRUE))[[1]]
environment(CellSIUS_fn) <- asNamespace("CellSIUS")
out <- CellSIUS_fn(mat.norm = norm, group_id = unname(grp[colnames(norm)]), min_n_cells = 10, min_fc = 2, corr_cutoff = NULL,
                iter = 0, max_perc_cells = 50, fc_between_cutoff = 1,
                mcl_path = Sys.getenv("MCL", "mcl"))
cells <- colnames(norm); score <- setNames(numeric(length(cells)), cells); call <- setNames(rep(FALSE, length(cells)), cells)
if (is.data.frame(out) && nrow(out) > 0) {
  m <- out[grepl("_1$", sub_cluster)]                              # member rows (suffix _1 = cell is in the subcluster)
  if (nrow(m) > 0) {
    m[, n_genes := length(unique(gene_id)), by = sub_cluster]
    m[, lfc := mean(log2FC), by = sub_cluster]
    best <- m[, .(s = max(n_genes + lfc / 100), ng = max(n_genes)), by = cell_idx]   # more signature genes = stronger; mean log2FC breaks ties
    score[best$cell_idx] <- best$s; call[best$cell_idx] <- best$ng >= 3   # min_n_genes=3 as in CellSIUS_final_cluster_assignment
  }
}
write_out(cells, score, call, a[2]); cat("n_called", sum(call), "\n")
