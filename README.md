Overview

This repository contains the R scripts and supporting materials used to perform the text-mining component of a scoping review on policy recommendations and research gaps at the intersection of biodiversity, pandemics, and One Health.
The workflow applies a bag-of-words (document-term matrix) approach using the R packages tm and caret to systematically identify and prioritise recurring keywords and multi-word expressions extracted from the literature.

Repository Structure
1. R scripts used for text preprocessing and generation of the document-term matrix (bag-of-words), including n-gram analysis.
2. Example xslx file with a portion of the raw extracted data
3. Example CSV files replicating the structure of the original dataset (columns: recommendations, gaps). These are sample datasets only, provided to allow code execution.

Data Availability
The original dataset consists of verbatim extracted text segments (policy-relevant content and knowledge gaps) from 425 articles. Due to file size and data-sharing constraints, the full dataset is not hosted in this repository.
To ensure reproducibility:
1. Sample datasets are provided that mirror the structure and format of the original data.
2. The full dataset can be made available upon reasonable request to the authors.

Methods Summary
The analysis follows a two-step approach:
- Data Extraction (manual)
1. Text was extracted verbatim from included studies by reviewers using a standardised template.
2. Extracted content includes sentences or short paragraphs describing policy recommendations and research gaps.
- Text Mining (computational)
1. Text preprocessing (lowercasing, removal of punctuation, numbers, and stopwords)
2. Generation of a document-term matrix (bag-of-words)
3. Inclusion of multi-word expressions (n-grams)
4. Frequency-based identification of key terms
- Synthesis (manual)
1. The most frequent terms and expressions were reviewed by the authors
2. Grouped into broader thematic categories
3. Consolidated into final policy recommendations and research gaps

⚠️ Note: The thematic grouping and final synthesis were conducted manually and are not automated in the R scripts.
