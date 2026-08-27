# =============================================
#  Bag-of-Words + N-gram Analysis Recommendations
#  (fixed for reproducibility)
# =============================================


library(tidyverse)
library(tidytext)

set.seed(42)  # ensure clustering step is reproducible

# -------------------------------------------------
# 1. Read data
# -------------------------------------------------
# The file has a single column of free-text recommendations. Most rows are
# NOT quoted and contain internal commas, so we deliberately do NOT use a
# comma-delimited reader. Each recommendation is confirmed to occupy exactly
# one physical line, so reading line-by-line and dropping the header is the
# robust approach here.
raw_lines <- read_lines("eklipse_rec_tm.csv")  # read_lines auto-detects the UTF-8 BOM

data <- tibble(
  recommendations = raw_lines[-1]                 # drop header row
) %>%
  mutate(
    recommendations = str_trim(recommendations),
    # Strip one layer of leading/trailing double quotes for the small
    # subset of rows that were quoted in the source file, so quoting is
    # not treated as literal text.
    recommendations = str_remove(recommendations, '^"'),
    recommendations = str_remove(recommendations, '"$')
  ) %>%
  filter(recommendations != "")                   # drop blank lines

cat("Rows read:", nrow(data), "\n")

# -------------------------------------------------
# 2. Tokenization with 1-5 n-grams
# -------------------------------------------------
# Pure-R n-gram tokenizer via tidytext (no RWeka/Java dependency).
ngram_range <- 1:5

tidy_tokens <- map_dfr(ngram_range, function(n) {
  data %>%
    mutate(id = row_number()) %>%
    unnest_tokens(
      output = term,
      input  = recommendations,
      token  = "ngrams",
      n      = n
    )
}) %>%
  filter(str_detect(term, "[a-z]")) %>%             # keep only terms with letters
  filter(!is.na(term))

# Remove terms whose FIRST word is a standard stop word, matching the
# intent of the original single-word anti_join(stop_words) filter when
# applied across multi-word n-grams.
first_word <- word(tidy_tokens$term, 1)
tidy_tokens <- tidy_tokens[!(first_word %in% stop_words$word), ]

custom_stopwords <- c("recommend", "should", "could", "may", "must",
                      "also", "however", "therefore", "thus", "etc",
                      "recommend", "recommendation", "study", "research",
                      "need", "needed", "important", "key", "urgent")

tidy_tokens <- tidy_tokens %>%
  filter(!term %in% custom_stopwords)

# -------------------------------------------------
# 3. Calculate Term Frequency & Document Frequency
# -------------------------------------------------
term_stats <- tidy_tokens %>%
  count(term, sort = TRUE) %>%
  rename(Frequency = n)

document_stats <- tidy_tokens %>%
  group_by(term) %>%
  summarise(Document_Frequency = n_distinct(id), .groups = "drop") %>%
  mutate(Doc_Freq_Percent = round(Document_Frequency / nrow(data) * 100, 2))

results <- term_stats %>%
  left_join(document_stats, by = "term") %>%
  select(term, Frequency, Document_Frequency, Doc_Freq_Percent) %>%
  arrange(desc(Frequency))

# =============================================
#  Bag-of-Words Clustering
# =============================================
top_terms <- results %>%
  filter(Document_Frequency >= 5) %>%     # adjust threshold
  slice_head(n = 800) %>%                 # adjust as needed
  pull(term)

dtm_sparse <- tidy_tokens %>%
  filter(term %in% top_terms) %>%
  count(id, term) %>%
  cast_sparse(row = id, column = term, value = n)

dtm_matrix <- as.matrix(dtm_sparse)

# Hierarchical Clustering
set.seed(42)
dist_matrix <- dist(dtm_matrix, method = "euclidean")
hc <- hclust(dist_matrix, method = "ward.D2")

clusters <- cutree(hc, k = 40)

data_with_clusters <- data %>%
  mutate(id = row_number()) %>%
  # documents that produced zero surviving tokens (e.g. filtered out
  # entirely by stopwords) are not in dtm_matrix; join instead of
  # positional indexing so those rows get NA rather than a wrong cluster.
  left_join(
    tibble(id = as.integer(names(clusters)), theme_cluster = clusters),
    by = "id"
  )

cluster_terms <- tidy_tokens %>%
  left_join(data_with_clusters %>% select(id, theme_cluster), by = "id") %>%
  filter(!is.na(theme_cluster)) %>%
  count(theme_cluster, term, sort = TRUE) %>%
  group_by(theme_cluster) %>%
  slice_head(n = 15) %>%
  ungroup()

write_csv(cluster_terms, "cluster_top_terms_recs.csv")
write_csv(data_with_clusters, "recommendations_with_40_themes.csv")
