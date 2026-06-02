# =============================================
# Knowledge gap Text Analysis: Bag-of-Words + Clustering
# =============================================

library(tidyverse)
library(tidytext)
library(tm)        # still useful for some preprocessing
library(RWeka)     # for n-gram tokenizer
library(factoextra)# for visualization of clusters

# 1. Read data -------------------------------------------------
data <- read.csv("eklipse_gap_tm.csv", 
                 stringsAsFactors = FALSE, 
                 encoding = "UTF-8")

# 2. Tidy preprocessing with tidytext --------------------------

custom_stopwords <- c("future", "should", "could", "efforts", "must",
                        "also", "however", "therefore", "thus", 
                        "recommend", "recommendation", "study", "research",
                        "need", "needed", "important", "key", "urgent")
tidy_gaps <- data %>%
  select(doc_id = 1, text = knowledge_gaps) %>% 
  unnest_tokens(output = "word",
                input = text,
                token = "ngrams",
                n = 5, n_min = 1) %>%
  filter(!str_detect(word, "^[0-9]+$")) %>%                # remove pure numbers
  filter(str_length(word) >= 2) %>%                        # remove very short tokens
  anti_join(stop_words, by = "word") %>%                   # standard English stopwords
  anti_join(tibble(word = custom_stopwords), by = "word") # custom stopwords



# 3. Bag-of-Words: Term Frequency & Document Frequency ---------
term_stats <- tidy_gaps %>%
  count(word, sort = TRUE) %>%
  rename(Frequency = n) %>%
  mutate(Document_Frequency = map_int(word, ~ sum(tidy_recs$word == .x))) %>%
  mutate(Doc_Freq_Percent = round(Document_Frequency / length(unique(tidy_recs$doc_id)) * 100, 2))

# Save full bag-of-words
write_csv(term_stats, "knowledge_gap_bagofwords.csv")

cat("Top 20 expressions:\n")
print(head(term_stats, 20))

# 4. Document-Term Matrix for Clustering -----------------------
# Create DTM (sparse matrix)
dtm <- tidy_recs %>%
  count(doc_id, word) %>%
  cast_dtm(document = doc_id, term = word, value = n)

# Convert to regular matrix for clustering
dtm_matrix <- as.matrix(dtm)

# Remove very rare terms to reduce noise
dtm_matrix <- dtm_matrix[, colSums(dtm_matrix) >= 3]

# 5. Cluster Analysis ------------------------------------------

# --- A. Hierarchical Clustering on Terms  ---
# Transpose so terms are rows
term_dtm <- t(dtm_matrix)

# Use cosine similarity 
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
