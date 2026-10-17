#!/usr/bin/env Rscript
# scDesign3 arm for the rare-class augmentation experiment.
#
# Follows the call structure of the working stage-2 augmenter of our earlier scDesign3 augmentation
# pipeline (a separate project)
# rather than a hand-rolled split of the pipeline. Four things differ from a naive port and all four
# were failure causes here:
#   1. filtered_gene comes from construct_data (joint_data$filtered_gene). fit_copula does NOT
#      return it, and simu_new dies with a dimnames error if it is NULL.
#   2. important_feature is an explicit named logical vector passed to BOTH fit_copula and simu_new.
#   3. fit_copula needs DT=TRUE, pseudo_obs=FALSE, if_sparse=FALSE.
#   4. new_covariate rows are SAMPLED FROM joint_data$dat, never hand-built, so every column,
#      factor level and offset matches what the marginals were fitted against.
#
# NO FALLBACK. If a fit fails this writes a failure marker and exits non-zero. The earlier pipeline's wrapper
# silently resampled REAL cells and labelled them synthetic when scDesign3 failed, which would turn
# this arm into the upsampling arm and hide the very effect we are measuring.
suppressPackageStartupMessages({
  library(Matrix); library(SingleCellExperiment); library(scDesign3)
})
a <- commandArgs(TRUE)
DIR <- a[1]; NC <- as.integer(a[2]); SEED <- as.integer(a[3])

# The earlier pipeline's env-strip patch: fitted objects keep formula/terms whose
# .Environment holds that gene's training rows (5202 -> 19 bytes/cell/gene).
# Unpatched, 2000 genes x 12010 cells = 125 GB across 8 forks, so every
# worker dies and fit_marginal returns try-errors. Patched it is 456 MB.
MEMP <- file.path(Sys.getenv("HR_WORK", path.expand("~/work")), "step0b_ht/mem_patch.R")
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
# A 1-cell group aborts fit_copula, and scDesign3 does not find out until AFTER fit_marginal.
# Measured with this guard removed (sd3_arm_nofloor.R) on the gen9 donor-disjoint matrix:
#   2000 genes: fit_marginal 7361 s ok, then fit_copula -> "Argument 'dim.' must be an integer
#               vector of length two", matrixStats::colMeans2 via scDesign3:::cal_cor. Total 7673 s.
#    200 genes: identical error in 1229 s, so it is the CELL axis, not the gene axis.
# Failing here costs 4 s instead of 2 h. 2 is the tested floor.
stopifnot(min(tab) >= 2)

t0 <- Sys.time(); el <- function() as.numeric(Sys.time()-t0, units='secs')

# withCallingHandlers fires BEFORE the stack unwinds, so sys.calls() still holds the real frames.
# A bare conditionMessage ("$ operator is invalid for atomic vectors") names no call and cost an
# entire 23-minute fit to learn nothing from.
tb <- NULL
trap <- function(e) {
  cs <- sys.calls()
  tb <<- c(paste('MESSAGE:', conditionMessage(e)),
           paste('CALL   :', paste(deparse(conditionCall(e)), collapse=' ')),
           'STACK (innermost first):',
           rev(vapply(cs, function(z) substr(paste(deparse(z), collapse=' '), 1, 220), '')))
}
ok <- tryCatch(withCallingHandlers({

  jd <- construct_data(sce=sce, assay_use='counts', celltype='cell_type',
        pseudotime=NULL, spatial=NULL, other_covariates=c('library'),
        ncell=ncol(sce), corr_by='cell_type', parallelization='mcmapply', BPPARAM=NULL)
  # corr_by must be ONE column: a multi-column corr_by is passed through mclust and silently
  # collapses the group universe to a handful of clusters of factor codes.
  stopifnot(length(unique(jd$dat$corr_group)) == length(tab))
  cat(sprintf('  construct_data %.0fs, %d corr groups, %d genes filtered\n',
              el(), length(unique(jd$dat$corr_group)), length(jd$filtered_gene)))

  # fit_marginal is ~40 min at 2000 genes x 12k cells and is deterministic given (data, formulas),
  # so cache it per temp dir. Keyed on the matrix shape to make a stale cache impossible to reuse.
  # cache in the PARENT of the per-run temp dir, keyed by seed: the driver makes a fresh mkdtemp
  # every run, so a cache inside DIR can never be reused across runs.
  # shape alone collided across panel datasets with equal train counts (another dataset's fit was
  # loaded), so the key also carries the md5 of this run's counts + labels.
  key <- paste(substr(tools::md5sum(file.path(DIR, c('train_counts.mtx', 'train_meta.csv'))), 1, 10), collapse='')
  mrf <- file.path(dirname(DIR), sprintf('mr_s%d_%dx%d_%s.rds', SEED, nrow(sce), ncol(sce), key))
  # A forked fit_marginal can return a try-error for EVERY gene and still exit 0; the failure only
  # surfaces later as "$ operator is invalid for atomic vectors" inside fit_copula. Validate here,
  # and never write a poisoned cache.
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
    if (g != length(mr)) stop(sprintf('fit_marginal returned %d/%d usable fits (n_cores=%d). Forked workers died; the earlier pipeline runs this at n_cores=1.', g, length(mr), NC))
    saveRDS(mr, mrf); cat(sprintf('  fit_marginal %.0fs, %d/%d fits ok (cached)\n', el(), g, length(mr)))
  }

  # important_feature as an explicit named logical vector, exactly as the earlier pipeline builds it.
  zf  <- Matrix::rowMeans(counts(sce) == 0)
  imp <- zf <= 0.80
  if (sum(imp) < 2L) imp <- rep(TRUE, length(imp))
  names(imp) <- rownames(sce)
  cat(sprintf('  important_feature %d/%d kept\n', sum(imp), length(imp)))

  cp <- fit_copula(sce=sce, assay_use='counts', marginal_list=mr, family_use='nb',
        copula='gaussian', DT=TRUE, pseudo_obs=FALSE, n_cores=NC, input_data=jd$dat,
        important_feature=imp, if_sparse=FALSE, parallelization='mcmapply', BPPARAM=NULL)
  cat(sprintf('  fit_copula %.0fs\n', el()))

  # new_covariate rows are SAMPLED FROM jd$dat so factor levels and the library offset are
  # guaranteed consistent with the fitted marginals. Hand-building this frame is what broke.
  rows_by_ct <- split(seq_len(nrow(jd$dat)), as.character(jd$dat$cell_type))
  nc <- do.call(rbind, lapply(seq_len(nrow(want)), function(i) {
          cls <- as.character(want$cell_type[i]); k <- as.integer(want$n[i])
          pool <- rows_by_ct[[cls]]
          stopifnot(!is.null(pool), length(pool) > 0)
          jd$dat[sample(pool, k, replace=TRUE), , drop=FALSE]
        }))
  # simu_new needs >= 2 rows for any corr group it sees; a 1-row group drops a dimension.
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
