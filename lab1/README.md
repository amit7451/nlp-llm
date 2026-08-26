# Lab 1: Foundations of Natural Language Processing & Large Language Models

A rigorous, end-to-end pedagogical laboratory notebook covering the essential mathematical foundations and practical implementations of modern Natural Language Processing (NLP) leading up to Large Language Models (LLMs).

---

## 📚 Laboratory Syllabus & Core Modules

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      NLP & LLM Foundations Pipeline                      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
   ┌─────────────────────────────────┴─────────────────────────────────┐
   │ 1. Text Processing & Classical Feature Engineering                │
   │    • Regex Cleaning & Contraction Normalization                   │
   │    • Stemming (Porter) vs Lemmatization (WordNet + POS)           │
   │    • Sentence Segmentation & POS Tagging                          │
   │    • Named Entity Recognition (NLTK & spaCy en_core_web_sm)       │
   │    • Context-Free Grammar Parsing & Vectorization (BoW, TF-IDF)   │
   └─────────────────────────────────┬─────────────────────────────────┘
                                     │
   ┌─────────────────────────────────┴─────────────────────────────────┐
   │ 2. Modern Subword Tokenization Algorithms                         │
   │    • Word-level & Character-level tradeoffs                       │
   │    • Byte Pair Encoding (BPE) — Full Algorithm from Scratch       │
   │    • WordPiece (Likelihood-based scoring with '##' prefix)        │
   │    • SentencePiece & Unigram Language Model (Viterbi EM pruning)  │
   │    • Special Tokens, Attention Masks & Out-Of-Vocabulary Handling │
   └─────────────────────────────────┬─────────────────────────────────┘
                                     │
   ┌─────────────────────────────────┴─────────────────────────────────┐
   │ 3. Static Distributed Representations (Word Embeddings)           │
   │    • One-Hot Sparsity vs Dense Continuous Embeddings              │
   │    • Word2Vec (Continuous Bag-of-Words & Skip-gram Architecture)  │
   │    • GloVe (Global Vectors from Co-occurrence Statistics)         │
   │    • FastText (Character N-gram Embeddings for Unseen/OOV Words)  │
   └─────────────────────────────────┬─────────────────────────────────┘
                                     │
   ┌─────────────────────────────────┴─────────────────────────────────┐
   │ 4. Contextual Embeddings & Deep Sequence Representations          │
   │    • Polysemy & Limitations of Static Embeddings                  │
   │    • ELMo (Learned Layer-Weighted BiLSTM Representations)         │
   │    • BERT / RoBERTa (Bidirectional Self-Attention Mechanism)     │
   │    • Masked Language Modeling (MLM 80/10/10 Masking Strategy)     │
   └─────────────────────────────────┬─────────────────────────────────┘
                                     │
   ┌─────────────────────────────────┴─────────────────────────────────┐
   │ 5. Probabilistic Language Modeling & Text Generation              │
   │    • N-gram Maximum Likelihood Estimation (MLE)                   │
   │    • Smoothing Techniques: Laplace Add-k & Katz Backoff           │
   │    • Autoregressive Decoding & Temperature-controlled Sampling    │
   │    • Perplexity Evaluation on Held-out Sequences                  │
   └─────────────────────────────────┬─────────────────────────────────┘
                                     │
   ┌─────────────────────────────────┴─────────────────────────────────┐
   │ 6. Quantitative Evaluation Metrics                                │
   │    • Exact Match (EM), Precision, Recall, Macro/Micro F1-Score    │
   │    • N-gram Overlap Metrics: BLEU (Precision) & ROUGE (Recall)    │
   │    • Embedding-based Semantic Evaluation: BERTScore               │
   │    • Intrinsic vs Extrinsic Evaluation Paradigms                  │
   └───────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Environment Setup & Installation

### Option 1: Using the Shared Workspace Virtual Environment

From the root directory `nlp_llm`:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### Option 2: Standalone Isolated Lab Environment

If you prefer to maintain a dedicated environment specifically for Lab 1:

```bash
# Navigate to lab1 folder
cd lab1

# Create isolated virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install lab dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLP models (automatically downloaded on first notebook run as well)
python -m spacy download en_core_web_sm
```

---

## 🚀 Running the Notebook

1. **Launch Jupyter Notebook / JupyterLab**:
   ```bash
   jupyter notebook lab1.ipynb
   ```
2. **Register the Environment Kernel (Optional)**:
   ```bash
   python -m ipykernel install --user --name nlp-llm-lab1 --display-name "Python (NLP Lab 1)"
   ```
3. Open `lab1.ipynb` in your browser or IDE (VS Code, Cursor, Antigravity) and select the `Python (NLP Lab 1)` or `.venv` kernel.

---

## 📦 File Inventory

| File | Description |
|---|---|
| [`lab1.ipynb`](./lab1.ipynb) | Complete laboratory notebook with theory, formulas, diagrams, and runnable code |
| [`requirements.txt`](./requirements.txt) | Pinpoint dependencies for all NLP & ML packages used in Lab 1 |
| [`README.md`](./README.md) | Laboratory guide and conceptual documentation |

---

## 🔬 Key Libraries Used

- **`nltk`** (v3.8+): Tokenization, POS tagging, WordNet lemmatizer, Porter stemmer, BLEU metric, CFG chart parser.
- **`scikit-learn`** (v1.3+): CountVectorizer (BoW), TfidfVectorizer, classification metrics (Precision, Recall, F1).
- **`gensim`** (v4.3+): Word2Vec (CBOW, Skip-Gram) and FastText subword embeddings.
- **`sentencepiece`** (v0.2+): Unigram language model and BPE tokenizer framework.
- **`spacy`** (v3.7+): Pretrained statistical & transformer NLP pipelines (`en_core_web_sm`), named entity recognition and displaCy visualization.
- **`rouge-score`** (v0.1.2+): ROUGE-1, ROUGE-2, ROUGE-L summarization overlap evaluation.
- **`numpy` & `pandas`**: Vector computations, cosine similarity, transition probability matrices, structured displays.
