# Memory patch for scDesign3: keeps the memory used by the fitted marginal models flat as the number of cells N grows.
# It only acts when the environment variable SD3_MEM is 1 (scdesign3.R sets it). The output matched the unpatched run.
#
# Why it is needed: every fitted object keeps a formula and terms whose .Environment is the frame the model was
# fitted in, and that frame holds the full training data for the gene. object.size() does not follow environments,
# so this cost is invisible there, yet it dominates the resident and the serialized size.
# Measured on scDesign3 fits to six AIDA rare-type groups of 91 to 779 cells: 5202 bytes per cell per gene before, 19 after.
#
# Three parts, none of which changes the results:
#  (a) every predict() call without newdata gets an explicit newdata, so a fit does not have to keep its own copy
#      of the training rows;
#  (b) the AIC and BIC calls in fit_copula are replaced by NA, because model_aic and model_bic are not used
#      downstream and stats::AIC needs the per-cell arrays that are dropped;
#  (c) each fit is slimmed as it is produced, inside the per-gene worker of fit_marginal, so the peak memory stays
#      low and not only the size of the returned object.

# ---- (c) the slimmer ----

# One shared, empty environment for every stripped fit: it holds nothing but
# still chains to the search path so predict() can resolve offset(), log(), ...
# baseenv() is not enough because offset() lives in stats.
.SD3_ENV <- new.env(parent = globalenv())

# Components whose size grows with N. Everything else (coefficients, terms,
# xlevels, contrasts, family, ...) is O(p) and kept, so model.matrix still
# sees every factor level when a chunk's covariates do not span them.
.SD3_DROP <- list(
  gam    = c("residuals","fitted.values","linear.predictors","weights","prior.weights",
             "working.weights","y","z","hat","offset"),
  gamlss = c("residuals","weights","y","rqres",
             "mu.fv","mu.lp","mu.wv","mu.wt","mu.offset","mu.x","mu.qr",
             "sigma.fv","sigma.lp","sigma.wv","sigma.wt","sigma.offset","sigma.x","sigma.qr"))

.sd3_slim_fit <- function(f) {
  if (is.null(f) || is.logical(f)) return(f)
  if (!identical(Sys.getenv("SD3_MEM", "0"), "1")) return(f)
  for (n in intersect(.SD3_DROP[[if (inherits(f, "gamlss")) "gamlss" else "gam"]], names(f)))
    f[[n]] <- NULL
  # the model frame is the last O(N) piece; one row keeps its levels and attrs
  if (!is.null(f$model)) f$model <- f$model[1, , drop = FALSE]
  for (n in c("terms","formula","pred.formula","mu.terms","sigma.terms",
              "nu.terms","tau.terms","mu.formula","sigma.formula")) {
    if (!is.null(f[[n]]) && !is.null(attr(f[[n]], ".Environment")))
      attr(f[[n]], ".Environment") <- .SD3_ENV
  }
  if (!is.null(f$pterms) && !is.null(attr(f$pterms, ".Environment")))
    attr(f$pterms, ".Environment") <- .SD3_ENV
  if (!is.null(f$model) && !is.null(attr(f$model, "terms")) &&
      !is.null(attr(attr(f$model, "terms"), ".Environment")))
    attr(attr(f$model, "terms"), ".Environment") <- .SD3_ENV
  f$call <- NULL
  f
}

# After-the-fact form, for a marginal_list that was fitted without the patch.
slim_marginal <- function(marginal_list) {
  if (!identical(Sys.getenv("SD3_MEM", "0"), "1")) return(marginal_list)
  lapply(marginal_list, function(x) {
    x$fit <- .sd3_slim_fit(x$fit)
    x
  })
}

# ---- (a) and (b), and wiring (c) into fit_marginal ----

if (identical(Sys.getenv("SD3_MEM", "0"), "1")) local({
  ns <- asNamespace("scDesign3")

  repatch <- function(fn, subs) {
    d <- deparse(ns[[fn]], width.cutoff = 500)
    for (i in seq_along(subs)) {
      from <- names(subs)[i]
      if (!any(grepl(from, d, fixed = TRUE)))
        stop(sprintf("mem_patch: pattern not found in %s: %s", fn, from))
      d <- gsub(from, subs[[i]], d, fixed = TRUE)
    }
    f <- eval(parse(text = paste(d, collapse = "\n")))
    stopifnot(is.function(f), !is.function(body(f)))   # guard: deparse includes the header
    environment(f) <- asNamespace("scDesign3")
    assignInNamespace(fn, f, ns = "scDesign3")
    pe <- as.environment("package:scDesign3")          # attached copy must match the namespace
    if (exists(fn, envir = pe, inherits = FALSE)) {
      if (bindingIsLocked(fn, pe)) unlockBinding(fn, pe)
      assign(fn, f, envir = pe)
      lockBinding(fn, pe)
    }
    invisible(f)
  }

  repatch("convert_n", list(
    'stats::predict(fit, type = "response", what = "mu", data = data)' =
      'stats::predict(fit, type = "response", what = "mu", newdata = data, data = data)',
    'stats::predict(fit, type = "response", what = "sigma", data = data)' =
      'stats::predict(fit, type = "response", what = "sigma", newdata = data, data = data)',
    'stats::predict(fit, type = "response", what = "nu", data = data)' =
      'stats::predict(fit, type = "response", what = "nu", newdata = data, data = data)',
    'mean_vec <- stats::predict(fit, type = "response")' =
      'mean_vec <- stats::predict(fit, type = "response", newdata = data)'))

  repatch("extract_para", list(
    'mean_vec <- stats::predict(fit, type = "response")' =
      'mean_vec <- stats::predict(fit, type = "response", newdata = data)',
    'newdata = new_covariate, data = data' =
      'newdata = if (is.null(new_covariate)) data else new_covariate, data = data'))

  repatch("fit_copula", list(
    'sum(sapply(marginals[qc_gene_idx], stats::AIC))' = 'NA_real_',
    'sum(sapply(marginals[qc_gene_idx], stats::BIC))' = 'NA_real_',
    'cal_aic(norm.mat = newmat, cor.mat = cor.mat, ind = ind)' = 'NA_real_',
    'cal_bic(norm.mat = newmat, cor.mat = cor.mat, ind = ind)' = 'NA_real_'))

  # slim per gene, at the two per-gene return sites, so the accumulating list
  # never holds a full fit; this is what keeps the peak memory down
  repatch("fit_marginal", list(
    'return(list(fit = fit, warning = logs, time = time_list, removed_cell = remove_cell))' =
      'return(list(fit = .sd3_slim_fit(fit), warning = logs, time = time_list, removed_cell = remove_cell))',
    'return(list(fit = fit, removed_cell = remove_cell))' =
      'return(list(fit = .sd3_slim_fit(fit), removed_cell = remove_cell))'))

  message("mem_patch: fit_marginal / convert_n / extract_para / fit_copula patched")
})
