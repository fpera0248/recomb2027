# mem_patch.R -- P5: hold scDesign3's marginal_list footprint flat in N.
# Gated on SD3_MEM=1.  Verified output-identical (see p5 validation).
#
# Why this is needed: every fitted object keeps formula/terms whose
# .Environment is the frame it was fitted in, which holds that gene's full
# training data.  object.size() does not follow environments, so the cost is
# invisible there while dominating the resident and serialized footprint.
# Measured on step0b bins: 5202 -> 19 bytes per cell per gene.
#
# Three parts, all behaviour-preserving:
#  (a) give every no-newdata predict() an explicit newdata, so no fit has to
#      retain its own copy of the training rows;
#  (b) neutralise the AIC/BIC sites in fit_copula -- model_aic/model_bic are
#      consumed by nothing downstream, but stats::AIC needs the per-cell
#      arrays we drop;
#  (c) slim each fit as it is produced, inside fit_marginal's per-gene worker,
#      so peak memory is held down and not merely the returned object.

# ---- (c) the slimmer -------------------------------------------------------

# One shared, empty environment for every stripped fit: it holds nothing but
# still chains to the search path so predict() can resolve offset(), log(), ...
# baseenv() is NOT enough -- offset() lives in stats.
.SD3_ENV <- new.env(parent = globalenv())

# Components whose size is O(N).  Everything else (coefficients, terms,
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

# Post-hoc form, for a marginal_list that was fitted without the patch.
slim_marginal <- function(marginal_list) {
  if (!identical(Sys.getenv("SD3_MEM", "0"), "1")) return(marginal_list)
  lapply(marginal_list, function(x) { x$fit <- .sd3_slim_fit(x$fit); x })
}

# ---- (a) + (b) + wiring (c) into fit_marginal ------------------------------

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
      assign(fn, f, envir = pe); lockBinding(fn, pe)
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
  # never holds a full fit -- this is what keeps PEAK memory down
  repatch("fit_marginal", list(
    'return(list(fit = fit, warning = logs, time = time_list, removed_cell = remove_cell))' =
      'return(list(fit = .sd3_slim_fit(fit), warning = logs, time = time_list, removed_cell = remove_cell))',
    'return(list(fit = fit, removed_cell = remove_cell))' =
      'return(list(fit = .sd3_slim_fit(fit), removed_cell = remove_cell))'))

  message("mem_patch: fit_marginal / convert_n / extract_para / fit_copula patched")
})
