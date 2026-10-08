# Turkish Statistical Language Modeling with N-Grams

A Python implementation of **character-based and syllable-based statistical language models** for Turkish, using N-gram probability estimation, Good-Turing-inspired smoothing, perplexity evaluation, and text generation.

## Overview

This project explores how different tokenization approaches affect statistical language modeling in Turkish, an agglutinative language with complex morphological structures.

Two language models were developed from scratch: one operating at the **character level** and the other at the **syllable level**. Each model implements unigram, bigram, and trigram representations to learn patterns from a Turkish Wikipedia corpus.

The models are evaluated using perplexity scores and randomly generated text to compare their ability to capture linguistic patterns, predict token sequences, and produce meaningful Turkish text.

## Key Features

- **Character & Syllable Tokenization:** Custom preprocessing functions, including rule-based Turkish syllabification using regular expressions.
- **N-Gram Language Modeling:** Unigram, bigram, and trigram frequency and probability estimation using NLTK and Python dictionaries.
- **Probability Smoothing:** Good-Turing-inspired smoothing to address sparse data and unseen token sequences.
- **Perplexity Evaluation:** Quantitative comparison of model performance on held-out test data.
- **Text Generation:** Probabilistic sequence generation based on learned N-gram distributions.
- **Comparative Analysis:** Evaluation of how tokenization granularity and context length influence language modeling performance.

## Methodology

The project follows a statistical NLP pipeline:

1. **Data Preprocessing:** Clean the Turkish Wikipedia corpus, remove HTML formatting, and split the data into 95% training and 5% testing.
2. **Tokenization:** Convert the corpus into character-level and syllable-level sequences, preserving word and sentence boundaries.
3. **Model Training:** Construct unigram, bigram, and trigram frequency tables for both tokenization approaches.
4. **Smoothing:** Apply frequency-based smoothing to handle unseen N-grams.
5. **Evaluation:** Calculate perplexity scores on test sentences.
6. **Text Generation:** Generate token sequences using the most probable N-gram continuations.

## Results & Findings

The experiments demonstrated the importance of context length and tokenization strategy in statistical language modeling.

- **Unigram models** produced highly repetitive and largely meaningless sequences due to their lack of contextual information.
- **Bigram models** captured local token relationships more effectively but struggled with linguistic coherence.
- **Trigram models** generally achieved lower perplexity and generated more recognizable linguistic patterns.
- **Character-based models** often achieved lower perplexity, while **syllable-based models** produced more recognizable Turkish word fragments and occasionally meaningful phrases.

Overall, the syllable-based trigram model provided the most promising qualitative text-generation results, highlighting the potential benefits of linguistically informed tokenization for morphologically rich languages.

*Note: Perplexity scores across different tokenization units are not directly comparable because the prediction units differ. The results also reflect the limitations of the smoothing and probability estimation methods implemented in this project.*

## Technologies Used

- **Python** — Core implementation
- **NLTK** — N-gram extraction and language processing
- **Regular Expressions (re)** — Text processing and Turkish syllabification
- **Collections (Counter)** — N-gram frequency counting
- **html2text** — HTML cleaning

## Getting Started

### Installation

Install the required Python packages:

```bash
pip install nltk html2text
```

### Dataset

The project uses a Turkish Wikipedia corpus. Download and prepare the corpus, then place the input text file at:

```text
data/wiki_00
```

The dataset is not included in the repository.

### Running the Project

From the project directory, execute:

```bash
python main.py
```

By default, the program processes the entire corpus. To reduce execution time or memory consumption, modify the following line in `main.py`:

```python
preprocess(percentage=100)
```

For example, `percentage=10` processes approximately 10% of the corpus.

The program generates separate output files for each model, containing perplexity evaluations and generated text samples.

## Project Background

Developed as part of **CSE484 – Introduction to Natural Language Processing** at Gebze Technical University (Fall 2024–2025).

The project provided hands-on experience in implementing statistical language models, probability estimation, smoothing techniques, language-specific preprocessing, and quantitative NLP evaluation.

A detailed technical report documenting the implementation, experimental results, and comparative analysis is included in the project files.
