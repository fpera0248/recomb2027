# RaceID (Grun lab), CRAN 0.4.2 = https://github.com/dgrun/RaceID3_StemID2_package commit 403c1f26a7dde0b8edfb74690317c6ae6b4b1b7a.
# usage: Rscript run_raceid.R pool.h5ad out.parquet [cores]       (cores default 4)
# Vignette "Parameters: Large Datasets": SCseq -> filterdata(mintotal = 2000) -> compdist(pearson) -> clustexp(samp = 1000, FUNcluster = "hclust") -> findoutliers.
# samp = 1000 is the vignette's example value (its text suggests 10-25% of the cells). The sparse count matrix goes straight into SCseq. Unsupervised: no labels are read.
# call = the cell is in sc@out$out (the outliers findoutliers flags), the tool's own output. score = call + (1 - share)/2, share = size of the cell's final cluster / kept cells
# (the size term is our adapter). Cells the tool's filter drops get score 0, no call and group "dropped". group = final cluster (feeds the per-label arm, tools/run_lab.py).
a <- commandArgs(TRUE)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "io.R"))
suppressMessages(library(RaceID))
p <- read_pool(a[1])
nc <- if (length(a) > 2) as.integer(a[3]) else 4L
sc <- SCseq(p$counts)
sc <- filterdata(sc, mintotal = 2000, verbose = FALSE)
sc <- compdist(sc, metric = "pearson", no_cores = nc)
sc <- clustexp(sc, samp = 1000, FUNcluster = "hclust", verbose = FALSE)
sc <- findoutliers(sc, verbose = FALSE)
cp <- sc@cpart
cells <- colnames(p$counts)
stopifnot(all(names(cp) %in% cells))
i <- match(names(cp), cells)
call <- rep(FALSE, length(cells)); score <- numeric(length(cells)); group <- rep("dropped", length(cells))
share <- as.numeric(table(cp)[as.character(cp)]) / length(cp)
is_out <- names(cp) %in% sc@out$out
call[i] <- is_out
score[i] <- is_out + (1 - share) / 2
group[i] <- as.character(cp)
write_out(cells, score, call, a[2], group)
cat("n_dropped", length(cells) - length(cp), "n_called", sum(call), "\n")
