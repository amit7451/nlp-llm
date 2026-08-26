# Lab 1: Foundations of Natural Language Processing & Large Language Models

A complete, self-contained laboratory module covering the mathematical foundations, classical feature engineering, modern tokenizers from scratch, static and contextual embeddings, statistical language modeling, and evaluation metrics.

---

## 📂 Laboratory Structure

```
lab1/
├── .gitignore          # Lab 1 specific ignore rules (.venv, checkpoints, caches)
├── lab1.ipynb          # Verified end-to-end laboratory notebook (theory + runnable code)
├── requirements.txt    # Standalone dependencies for Lab 1
├── setup_env.bat       # One-click Windows setup script
├── setup_env.sh        # One-click Linux / macOS setup script
└── README.md           # Lab 1 documentation & syllabus
```

---

## 📚 Laboratory Syllabus & Modules

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

## 🛠️ Independent Environment Setup

Each lab in this repository is completely isolated. To set up and run Lab 1:

### Quick Setup (One-Click)

- **Windows**: Double-click or run `setup_env.bat`
- **Linux / macOS**: Run `./setup_env.sh` (or `bash setup_env.sh`)

### Manual Setup

1. **Open terminal in `lab1/`**:
   ```bash
   cd lab1
   ```

2. **Create isolated virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate environment**:
   ```bash
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1

   # Windows (Command Prompt)
   .\.venv\Scripts\activate.bat

   # Linux / macOS
   source .venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Download spaCy language model & register kernel**:
   ```bash
   python -m spacy download en_core_web_sm
   python -m ipykernel install --user --name nlp-llm-lab1 --display-name "Python (NLP-LLM Lab 1)"
   ```

---

## 🚀 Running the Notebook

```bash
jupyter notebook lab1.ipynb
```
Select the **`Python (NLP-LLM Lab 1)`** kernel or the local **`lab1/.venv`** interpreter in your IDE.

---

## 🔬 Core Dependencies

- **`nltk`** (v3.8+): Tokenization, POS tagging, WordNet lemmatizer, Porter stemmer, BLEU metric, CFG chart parser.
- **`scikit-learn`** (v1.3+): CountVectorizer (BoW), TfidfVectorizer, classification metrics (Precision, Recall, F1).
- **`gensim`** (v4.3+): Word2Vec (CBOW, Skip-Gram) and FastText subword embeddings.
- **`sentencepiece`** (v0.2+): Unigram language model and BPE tokenizer framework.
- **`spacy`** (v3.7+): Pretrained NLP pipelines (`en_core_web_sm`), named entity recognition and displaCy visualization.
- **`rouge-score`** (v0.1.2+): ROUGE-1, ROUGE-2, ROUGE-L summarization overlap evaluation.
- **`numpy` & `pandas`**: Vector computations, cosine similarity, transition probability matrices, structured displays.
- **`ipykernel` & `jupyter`**: Interactive notebook runtime and execution kernel.
