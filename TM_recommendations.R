# =============================================
#  Bag-of-Words + N-gram Analysis Recommendations
# =============================================

library(tidyverse)
library(tidytext)
library(RWeka)        
library(dplyr)
library(stringr)

# 1. Read data -------------------------------------------------
data <- read.csv("eklipse_rec_tm.csv", 
                 stringsAsFactors = FALSE, 
                 encoding = "UTF-8")
#"eklipse_rec_tm.csv" had a single column : recommendations
# 2. Tokenization with 1-5 n-grams -----------------------------


# Custom n-gram tokenizer function
ngram_tokenizer <- function(x, n_min = 1, n_max = 5) {
  NGramTokenizer(x, Weka_control(min = n_min, max = n_max))
}

# Unnest tokens (using tidytext + custom tokenizer)
tidy_tokens <- data %>%
  mutate(id = row_number()) %>%                    # document id
  unnest_tokens(output = "term", 
                input = recommendations,
                tokenizer = function(x) ngram_tokenizer(x, 1, 5)) %>%
  filter(str_detect(term, "[a-z]")) %>%            # keep only terms with letters
  anti_join(stop_words, by = c("term" = "word"))   

custom_stopwords <- c("recommend", "should", "could", "may", "must", 
                      "also", "however", "therefore", "thus", "etc","recommend", "recommendation", "study", "research",
                      "need", "needed", "important", "key","urgent")

tidy_tokens <- tidy_tokens %>%
  filter(!term %in% custom_stopwords)

# 3. Calculate Term Frequency & Document Frequency -------------
term_stats <- tidy_tokens %>%
  count(term, sort = TRUE) %>%
  rename(Frequency = n)

document_stats <- tidy_tokens %>%
  group_by(term) %>%
  summarise(Document_Frequency = n_distinct(id), .groups = "drop") %>%
  mutate(Doc_Freq_Percent = round(Document_Frequency / nrow(data) * 100, 2))

# Combine both
results <- term_stats %>%
  left_join(document_stats, by = "term") %>%
  select(term, Frequency, Document_Frequency, Doc_Freq_Percent) %>%
  arrange(desc(Frequency))

# =============================================
#  Bag-of-Words Clustering
# =============================================
library(tidytext)
library(widyr)       
library(cluster)      # for hierarchical clustering

# Create DTM (sparse matrix) from top terms
top_terms <- results %>%
  filter(Document_Frequency >= 5) %>%     # adjust threshold
  slice_head(n = 800) %>%                  # adjust as needed
  pull(term)

dtm_sparse <- tidy_tokens %>%
  filter(term %in% top_terms) %>%
  count(id, term) %>%
  cast_sparse(row = id, column = term, value = n)

# Convert to dense matrix for clustering (if not too big)
dtm_matrix <- as.matrix(dtm_sparse)

# Hierarchical Clustering
set.seed(42)
dist_matrix <- dist(dtm_matrix, method = "euclidean")
hc <- hclust(dist_matrix, method = "ward.D2")

# Cut into 40 clusters (based on hc branches)
clusters <- cutree(hc, k = 40)

# Add cluster to original data
data_with_clusters <- data %>%
  mutate(id = row_number(),
         theme_cluster = clusters[id])

# View most common terms per cluster 
cluster_terms <- tidy_tokens %>%
  left_join(data_with_clusters %>% select(id, theme_cluster), by = "id") %>%
  count(theme_cluster, term, sort = TRUE) %>%
  group_by(theme_cluster) %>%
  slice_head(n = 15) %>%
  ungroup()

write_csv(cluster_terms, "cluster_top_terms_recs.csv")
write_csv(data_with_clusters, "recommendations_with_40_themes.csv")

