# export Ensembl 110 gene coordinates (Bioconductor AnnotationHub EnsDb sqlite, read with RSQLite) as a GTF (gene + transcript lines) for snapATAC2
suppressMessages({library(AnnotationHub); library(RSQLite)})
ah <- AnnotationHub(); q <- query(ah, c("EnsDb", "Homo sapiens", "110")); print(mcols(q)[, c("title", "genome")])
f <- cache(q[length(q)]); print(f)
con <- dbConnect(SQLite(), f); g <- dbGetQuery(con, "select * from gene"); print(head(g, 2))
g <- g[g$seq_name %in% c(1:22, "X", "Y", "MT"), ]
gtf <- data.frame(seqname = g$seq_name, source = "EnsDb110", feature = "gene", start = g$gene_seq_start, end = g$gene_seq_end, score = ".",
                  strand = ifelse(g$seq_strand == 1, "+", "-"), frame = ".",
                  attr = sprintf('gene_id "%s"; gene_name "%s"; gene_type "%s"; transcript_id "%s";', g$gene_id, g$gene_name, g$gene_biotype, g$gene_id))
tr <- gtf; tr$feature <- "transcript"
write.table(rbind(gtf, tr), path.expand("~/work/jiaqi1011/data/w6_atac/ensdb110_genes.gtf"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
cat("genes", nrow(g), "\n")
