# --- Load required packages ---
library(tm)
library(caret)

# --- Step 1: Read CSV file ---
data <- read.csv("eklipse_recs_analysis.csv", stringsAsFactors = FALSE)

# --- Step 3: Create a text corpus ---
corpus <- VCorpus(VectorSource(data$text))

# --- Step 4: Clean the text ---
corpus <- tm_map(corpus, content_transformer(tolower))      # lowercase
corpus <- tm_map(corpus, removePunctuation)                 # remove punctuation
corpus <- tm_map(corpus, removeNumbers)                     # remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("english")) # remove stopwords
corpus <- tm_map(corpus, stripWhitespace)                   # trim spaces

# --- Step 5: Create a Document-Term Matrix (Bag of Words) ---
dtm <- DocumentTermMatrix(corpus)

# --- Step 6: Convert to data frame ---
bag_of_words <- as.data.frame(as.matrix(dtm))

# Optional: Add back row IDs if you want to keep track of rows
bag_of_words$id <- 1:nrow(bag_of_words)

# --- Step 7: Check structure ---
str(bag_of_words)

# --- Step 8: Optionally create a training partition using caret ---
set.seed(123)
trainIndex <- createDataPartition(bag_of_words$id, p = 0.8, list = FALSE)
trainData <- bag_of_words[trainIndex, ]
testData  <- bag_of_words[-trainIndex, ]


cat("✅ Bag of Words matrix created with", ncol(bag_of_words)-1, "terms\n")

## ------for knowledge gaps ----

# --- Step 1: Read CSV file ---
data <- read.csv("eklipse_gaps_analysis.csv", stringsAsFactors = FALSE)

# --- Step 3: Create a text corpus ---
corpus <- VCorpus(VectorSource(data$text))

# --- Step 4: Clean the text ---
corpus <- tm_map(corpus, content_transformer(tolower))      # lowercase
corpus <- tm_map(corpus, removePunctuation)                 # remove punctuation
corpus <- tm_map(corpus, removeNumbers)                     # remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("english")) # remove stopwords
corpus <- tm_map(corpus, stripWhitespace)                   # trim spaces

# --- Step 5: Create a Document-Term Matrix (Bag of Words) ---
dtm <- DocumentTermMatrix(corpus)

# --- Step 6: Convert to data frame ---
bag_of_words <- as.data.frame(as.matrix(dtm))

# Optional: Add back row IDs if you want to keep track of rows
bag_of_words$id <- 1:nrow(bag_of_words)

# --- Step 7: Check structure ---
str(bag_of_words)

# --- Step 8: Optionally create a training partition using caret ---
set.seed(123)
trainIndex <- createDataPartition(bag_of_words$id, p = 0.8, list = FALSE)
trainData <- bag_of_words[trainIndex, ]
testData  <- bag_of_words[-trainIndex, ]


cat("✅ Bag of Words matrix created with", ncol(bag_of_words)-1, "terms\n")

#_____________________________________________________________________________
### code for expressions n-word phrases

# --- for recommendations ---

# --- Load packages ---
library(tm)
library(caret)
library(RWeka)  # for n-gram tokenization

# --- Step 1: Read CSV ---
data <- read.csv("eklipse_recs_analysis.csv", stringsAsFactors = FALSE)
data$text <- paste(data$recommendations, data$gaps, sep = " ")

# --- Step 2: Create corpus ---
corpus <- VCorpus(VectorSource(data$text))

# --- Step 3: Clean text ---
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)

# --- Step 4: Define n-gram tokenizer ---
# for expressions (1–3 word sequences)
ngramTokenizer <- function(x) {
  RWeka::NGramTokenizer(x, RWeka::Weka_control(min = 1, max = 3))
}

# --- Step 5: Create Document-Term Matrix with n-grams ---
dtm <- DocumentTermMatrix(corpus, control = list(tokenize = ngramTokenizer))

# --- Step 6: Convert to data frame ---
bag_of_ngrams <- as.data.frame(as.matrix(dtm))
bag_of_ngrams$id <- 1:nrow(bag_of_ngrams)

# --- Step 7: Partition  ---
set.seed(123)
trainIndex <- createDataPartition(bag_of_ngrams$id, p = 0.8, list = FALSE)
trainData <- bag_of_ngrams[trainIndex, ]
testData  <- bag_of_ngrams[-trainIndex, ]


cat("✅ N-gram Bag of Words created with", ncol(bag_of_ngrams)-1, "features\n")

# --- for gaps ---


# --- Step 1: Read CSV ---
data <- read.csv("eklipse_gaps_analysis.csv", stringsAsFactors = FALSE)
data$text <- paste(data$recommendations, data$gaps, sep = " ")

# --- Step 2: Create corpus ---
corpus <- VCorpus(VectorSource(data$text))

# --- Step 3: Clean text ---
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)

# --- Step 4: Define n-gram tokenizer ---
# for expressions (1–3 word sequences)
ngramTokenizer <- function(x) {
  RWeka::NGramTokenizer(x, RWeka::Weka_control(min = 1, max = 3))
}

# --- Step 5: Create Document-Term Matrix with n-grams ---
dtm <- DocumentTermMatrix(corpus, control = list(tokenize = ngramTokenizer))

# --- Step 6: Convert to data frame ---
bag_of_ngrams <- as.data.frame(as.matrix(dtm))
bag_of_ngrams$id <- 1:nrow(bag_of_ngrams)

# --- Step 7: Partition  ---
set.seed(123)
trainIndex <- createDataPartition(bag_of_ngrams$id, p = 0.8, list = FALSE)
trainData <- bag_of_ngrams[trainIndex, ]
testData  <- bag_of_ngrams[-trainIndex, ]


cat("✅ N-gram Bag of Words created with", ncol(bag_of_ngrams)-1, "features\n")
