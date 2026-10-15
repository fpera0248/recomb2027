#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Matrix); library(SingleCellExperiment); library(scDesign3)
})
a <- commandArgs(TRUE)
DIR <- a[1]; NC <- as.integer(a[2]); SEED <- as.integer(a[3])

# strips each fit's captured environment; without it 8 forks need ~125 GB at 2000 genes x 12k cells
MEMP <- Sys.getenv('MEM_PATCH', 'mem_patch.R')
stopifnot(file.exists(MEMP)); Sys.setenv(SD3_MEM='1'); source(MEMP)
set.seed(SEED)

cnt <- readMM(file.path(DIR,'train_counts.mtx'))          # genes x cells
ct  <- read.csv(file.path(DIR,'train_meta.csv'), stringsAsFactors=FALSE)$cell_type
want<- read.csv(file.path(DIR,'want.csv'), stringsAsFactors=FALSE)   # cell_type, n
stopifnot(ncol(cnt)==length(ct))
cnt <- as(cnt,'dgCMatrix'); rownames(cnt) <- sprintf('g%04d',seq_len(nrow(cnt)))
colnames(cnt) <- sprintf('c%06d',seq_len(ncol(cnt)))

sce <- SingleCellExperiment(list(counts=cnt),
        colData=DataFrame(cell_type=droplevels(factor(ct)), library=Matrix::colSums(cnt)))
tab <- table(colData(sce)$cell_type)
cat(sprintf('  %d cells x %d genes, %d types, sizes %d..%d\n',
            ncol(sce),nrow(sce),length(tab),min(tab),max(tab)))
# a 1-cell group aborts fit_copula only after fit_marginal has run for hours
stopifnot(min(tab) >= 2)

t0 <- Sys.time(); el <- function() as.numeric(Sys.time()-t0, units='secs')

tb <- NULL
trap <- function(e) {
  cs <- sys.calls()
  tb <<- c(paste('MESSAGE:', conditionMessage(e)),
           paste('CALL   :', paste(deparse(conditionCall(e)), collapse=' ')),
           'STACK (innermost first):',
           rev(vapply(cs, function(z) substr(paste(deparse(z), collapse=' '), 1, 220), '')))
}
# no fallback to real cells: a failed fit writes FAILED.txt and exits non-zero
ok <- tryCatch(withCallingHandlers({

  jd <- construct_data(sce=sce, assay_use='counts', celltype='cell_type',
        pseudotime=NULL, spatial=NULL, other_covariates=c('library'),
        ncell=ncol(sce), corr_by='cell_type', parallelization='mcmapply', BPPARAM=NULL)
  stopifnot(length(unique(jd$dat$corr_group)) == length(tab))
  cat(sprintf('  construct_data %.0fs, %d corr groups, %d genes filtered\n',
              el(), length(unique(jd$dat$corr_group)), length(jd$filtered_gene)))

  key <- paste(substr(tools::md5sum(file.path(DIR, c('train_counts.mtx', 'train_meta.csv'))), 1, 10), collapse='')
  mrf <- file.path(dirname(DIR), sprintf('mr_s%d_%dx%d_%s.rds', SEED, nrow(sce), ncol(sce), key))
  good <- function(m) sum(vapply(m, function(x) is.list(x) && !is.null(x$fit), TRUE))
  if (file.exists(mrf)) {
    mr <- readRDS(mrf)
    if (good(mr) != length(mr)) { cat('  cache is poisoned, refitting\n'); file.remove(mrf); mr <- NULL }
    else cat(sprintf('  fit_marginal CACHED (%s)\n', basename(mrf)))
  } else mr <- NULL
  if (is.null(mr)) {
    mr <- fit_marginal(data=jd, predictor='gene',
          mu_formula='cell_type + offset(log(library))', sigma_formula='cell_type',
          family_use='nb', n_cores=NC, usebam=FALSE, edf_flexible=TRUE,
          parallelization='mcmapply', BPPARAM=NULL)
    g <- good(mr)
    if (g != length(mr)) stop(sprintf('fit_marginal returned %d/%d usable fits (n_cores=%d). Forked workers died; step0b runs this at n_cores=1.', g, length(mr), NC))
    saveRDS(mr, mrf); cat(sprintf('  fit_marginal %.0fs, %d/%d fits ok (cached)\n', el(), g, length(mr)))
  }

  zf  <- Matrix::rowMeans(counts(sce) == 0)
  imp <- zf <= 0.80
  if (sum(imp) < 2L) imp <- rep(TRUE, length(imp))
  names(imp) <- rownames(sce)
  cat(sprintf('  important_feature %d/%d kept\n', sum(imp), length(imp)))

  cp <- fit_copula(sce=sce, assay_use='counts', marginal_list=mr, family_use='nb',
        copula='gaussian', DT=TRUE, pseudo_obs=FALSE, n_cores=NC, input_data=jd$dat,
        important_feature=imp, if_sparse=FALSE, parallelization='mcmapply', BPPARAM=NULL)
  cat(sprintf('  fit_copula %.0fs\n', el()))

  rows_by_ct <- split(seq_len(nrow(jd$dat)), as.character(jd$dat$cell_type))
  nc <- do.call(rbind, lapply(seq_len(nrow(want)), function(i) {
          cls <- as.character(want$cell_type[i]); k <- as.integer(want$n[i])
          pool <- rows_by_ct[[cls]]
          stopifnot(!is.null(pool), length(pool) > 0)
          jd$dat[sample(pool, k, replace=TRUE), , drop=FALSE]
        }))
  tc <- table(as.character(nc$cell_type)); thin <- names(tc)[tc < 2L]
  if (length(thin)) {
    ex <- do.call(rbind, lapply(thin, function(cl)
            jd$dat[sample(rows_by_ct[[cl]], 2L - tc[[cl]], replace=TRUE), , drop=FALSE]))
    nc <- rbind(nc, ex)
  }
  nc$corr_group <- as.character(nc$cell_type)
  rownames(nc) <- paste0('Cell', seq_len(nrow(nc)))
  cat(sprintf('  simulating %d cells over %d classes\n', nrow(nc), nrow(want)))

  pa <- extract_para(sce=sce, assay_use='counts', marginal_list=mr, n_cores=NC,
        family_use='nb', new_covariate=nc, parallelization='mcmapply', BPPARAM=NULL,
        data=jd$dat)

  sim <- simu_new(sce=sce, assay_use='counts', mean_mat=pa$mean_mat, sigma_mat=pa$sigma_mat,
        zero_mat=pa$zero_mat, quantile_mat=NULL, copula_list=cp$copula_list, n_cores=NC,
        family_use='nb', nonnegative=TRUE, nonzerovar=TRUE, input_data=jd$dat,
        new_covariate=nc, important_feature=imp, parallelization='mcmapply',
        BPPARAM=NULL, filtered_gene=jd$filtered_gene)
  cat(sprintf('  simu_new %.0fs  -> %d x %d\n', el(), nrow(sim), ncol(sim)))
  stopifnot(nrow(sim) == nrow(sce), ncol(sim) == nrow(nc))   # genes x cells, padded back to full G

  writeMM(as(as.matrix(sim),'dgCMatrix'), file.path(DIR,'sim_counts.mtx'))
  write.csv(data.frame(cell_type=as.character(nc$cell_type)),
            file.path(DIR,'sim_meta.csv'), row.names=FALSE)
  TRUE
}, error=trap), error=function(e){
     out <- if (is.null(tb)) conditionMessage(e) else tb
     cat('  FAILED:', conditionMessage(e), '\n')
     cat(paste0('    ', head(out, 12)), sep='\n')
     writeLines(out, file.path(DIR,'FAILED.txt')); FALSE })
cat(sprintf('  total %.0fs  ok=%s\n', el(), ok))
quit(status=if (ok) 0 else 1)
