# Installs RaceID, or mrtree (CHOIR dependency, patched) and CHOIR, from the checkouts of tools/fetch_sources.sh.
# usage: TOOLS_SRC=tools_src Rscript tools/install_raceid_choir.R raceid|choir        (run from the repository root)
# raceid: into the tools library (RaceID does not use Seurat). choir: into a separate library with Seurat >= 5 (the paper's CHOIR ran on Seurat 5.5.1 / SeuratObject 5.4.0,
#   mini, and 5.6.0 / 5.4.0, laptop), since the tools library holds Seurat 4.4.0 for GapClust and SCISSORS; run it with that library's Rscript (CHOIR_RSCRIPT in scripts/run_tool.sh).
# mrtree 720f469 imports ggimage (plotting only) and SymSim (simulation only), which need ImageMagick and phytools; tools/mrtree.patch drops both imports from a copy of the checkout.
what <- commandArgs(TRUE)[1]
stopifnot(what %in% c("raceid", "choir"))
src <- normalizePath(Sys.getenv("TOOLS_SRC", "tools_src"))
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4)
if (what == "raceid") {
  remotes::install_local(file.path(src, "RaceID"), dependencies = NA, upgrade = "never")
  stopifnot(packageVersion("RaceID") == "0.4.2")
} else {
  stopifnot(packageVersion("Seurat") >= "5.0.0")
  patch <- normalizePath("tools/mrtree.patch")
  BiocManager::install(c("ggtree", "bluster", "BiocGenerics"), update = FALSE, ask = FALSE)
  install.packages(c("harmony", "ranger", "spatstat.univar", "ape", "checkmate", "clustree", "tidytree", "plyr", "progress",
                     "dendextend", "data.tree", "reshape2", "RSpectra", "uwot", "yaml", "tibble"))
  mr <- file.path(tempdir(), "mrtree")
  stopifnot(file.copy(file.path(src, "mrtree"), tempdir(), recursive = TRUE))
  stopifnot(system2("git", c("-C", mr, "apply", patch)) == 0)
  install.packages(mr, repos = NULL, type = "source")
  install.packages(file.path(src, "CHOIR"), repos = NULL, type = "source")   # as the original install; install_local would follow CHOIR's Remotes: field to the unpatched mrtree
  stopifnot(all(c("CHOIR", "mrtree") %in% rownames(installed.packages())))
}
