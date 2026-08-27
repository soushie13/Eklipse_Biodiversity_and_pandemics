# =============================================
# Knowledge gap Text Analysis: Bag-of-Words + Clustering
#  (fixed for reproducibility)
# =============================================

library(tidyverse)
library(tidytext)
library(tm)          # still useful for some preprocessing
library(factoextra)  # for visualization of clusters

set.seed(42)  # ensure clustering step is reproducible

# -------------------------------------------------
# 1. Read data
# -------------------------------------------------
# Assumes a single free-text column (one knowledge gap per line), like
# eklipse_rec_tm.csv. See note above if that assumption is wrong.
raw_lines <- read_lines("eklipse_gap_tm.csv")  # read_lines auto-detects the UTF-8 BOM

data <- tibble(
  text = raw_lines[-1]                            # drop header row
) %>%
  mutate(
    text = str_trim(text),
    text = str_remove(text, '^"'),
    text = str_remove(text, '"$')
  ) %>%
  filter(text != "") %>%
  mutate(doc_id = row_number())

cat("Rows read:", nrow(data), "\n")

# -------------------------------------------------
# 2. Tidy preprocessing with tidytext
# -------------------------------------------------
custom_stopwords <- c("future", "should", "could", "efforts", "must",
                       "also", "however", "therefore", "thus",
                       "recommend", "recommendation", "study", "research",
                       "need", "needed", "important", "key", "urgent")

tidy_gaps <- data %>%
  select(doc_id, text) %>%
  unnest_tokens(output = "word",
                input = text,
                token = "ngrams",
                n = 5, n_min = 1) %>%
  filter(!str_detect(word, "^[0-9]+$")) %>%                # remove pure numbers
  filter(str_length(word) >= 2) %>%                        # remove very short tokens
  anti_join(stop_words, by = "word") %>%                   # standard English stopwords
  anti_join(tibble(word = custom_stopwords), by = "word")  # custom stopwords

# -------------------------------------------------
# 3. Bag-of-Words: Term Frequency & Document Frequency
# -------------------------------------------------
term_stats <- tidy_gaps %>%
  count(word, sort = TRUE) %>%
  rename(Frequency = n) %>%
  mutate(Document_Frequency = map_int(word, ~ sum(tidy_gaps$word == .x))) %>%
  mutate(Doc_Freq_Percent = round(Document_Frequency / length(unique(tidy_gaps$doc_id)) * 100, 2))

write_csv(term_stats, "knowledge_gap_bagofwords.csv")
cat("Top 20 expressions:\n")
print(head(term_stats, 20))

# -------------------------------------------------
# 4. Document-Term Matrix for Clustering
# -------------------------------------------------
dtm <- tidy_gaps %>%
  count(doc_id, word) %>%
  cast_dtm(document = doc_id, term = word, value = n)

dtm_matrix <- as.matrix(dtm)

# Remove very rare terms to reduce noise
dtm_matrix <- dtm_matrix[, colSums(dtm_matrix) >= 3]

# -------------------------------------------------
# 5. Cluster Analysis
# -------------------------------------------------
# --- A. Hierarchical Clustering on Terms ---
# Transpose so terms are rows
term_dtm <- t(dtm_matrix)

# Cosine similarity BETWEEN TERMS (term_dtm has terms as rows).
# tcrossprod(x) = x %*% t(x) gives the terms x terms matrix of dot
# products between term row-vectors, which is what's needed here.
cosine_sim_terms <- function(x) {
  x <- as.matrix(x)
  norm2 <- rowSums(x^2)
  num <- tcrossprod(x)
  denom <- sqrt(outer(norm2, norm2))
  denom[denom == 0] <- NA  # guard against all-zero rows
  num / denom
}

sim_matrix <- cosine_sim_terms(term_dtm)

# Convert similarity to distance
dist_matrix <- as.dist(1 - sim_matrix)

# Hierarchical clustering
hc <- hclust(dist_matrix, method = "ward.D2")

# Cut into 32 clusters (based on hc branches)
clusters <- cutree(hc, k = 32)

# Create cluster results
term_clusters <- tibble(
  Expression = rownames(term_dtm),
  Cluster = clusters
) %>%
  left_join(term_stats, by = c("Expression" = "word")) %>%
  arrange(Cluster, desc(Frequency))

write_csv(term_clusters, "knowledge_gap_term_clusters.csv")

# --- B. Visualize clusters ---
dend_plot <- fviz_dend(hc, k = 32, cex = 0.6,
                        main = "Hierarchical Clustering of Knowledge Gaps")
ggsave("knowledge_gap_dendrogram.png", dend_plot, width = 14, height = 8, dpi = 300)

# Cluster size summary
cluster_summary <- term_clusters %>%
  count(Cluster, name = "Num_Terms") %>%
  arrange(desc(Num_Terms))
print(cluster_summary)

# -------------------------------------------------
# 6. Export representative terms per cluster
# -------------------------------------------------
top_per_cluster <- term_clusters %>%
  group_by(Cluster) %>%
  slice_max(Frequency, n = 8) %>%
  arrange(Cluster)

write_csv(top_per_cluster, "knowledge_gap_clusters_top_terms.csv")

# -------------------------------------------------
# Reproducibility record
# -------------------------------------------------
cat("\n---- Session info (keep with your publication materials) ----\n")
print(sessionInfo())# Use cosine similarity 
cosine_sim <- function(x) {
  x <- as.matrix(x)
  crossprod(x) / sqrt(tcrossprod(colSums(x^2)))
}

sim_matrix <- cosine_sim(term_dtm)

# Convert similarity to distance
dist_matrix <- as.dist(1 - sim_matrix)

# Hierarchical clustering
hc <- hclust(dist_matrix, method = "ward.D2")

# Cut into 35 clusters (based on hc branches)
clusters <- cutree(hc, k = 35)

# Hierarchical clustering
hc <- hclust(dist_matrix, method = "ward.D2")

# Cut into 32 clusters (based on hc branches)
clusters <- cutree(hc, k = 32)

# Create cluster results
term_clusters <- tibble(
  Expression = rownames(term_dtm),
  Cluster = clusters
) %>%
  left_join(term_stats, by = c("Expression" = "word")) %>%
  arrange(Cluster, desc(Frequency))

# Save cluster results
write_csv(term_clusters, "knowledge_gap_term_clusters.csv")

# --- B. Visualize clusters ---
# Dendrogram (first 100 terms for readability)
fviz_dend(hc, k = 32, cex = 0.6, main = "Hierarchical Clustering of Knowledge gaps")

# Cluster size summary
cluster_summary <- term_clusters %>%
  count(Cluster, name = "Num_Terms") %>%
  arrange(desc(Num_Terms))

print(cluster_summary)

# 6. Export representative terms per cluster -------------------
top_per_cluster <- term_clusters %>%
  group_by(Cluster) %>%
  slice_max(Frequency, n = 8) %>%
  arrange(Cluster)

write_csv(top_per_cluster, "knowledge_gap_clusters_top_terms.csv")
