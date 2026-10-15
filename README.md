# Natural Language Processing & Large Language Models (`nlp-llm`)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8+-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end, rigorous, hands-on repository for mastering **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** — spanning statistical linguistics, tokenizer algorithms from scratch, distributed embeddings, Transformer architectures, autoregressive pretraining, parameter-efficient fine-tuning (PEFT/LoRA), Retrieval-Augmented Generation (RAG), and post-training alignment (RLHF/DPO).

---

## 🏛️ Unified Repository Architecture

To provide a consistent, high-performance, and reproducible machine learning environment across all laboratories, this repository utilizes a **unified global virtual environment** (`.venv`) at the repository root:

```
nlp_llm/
│
├── .venv/                      # Unified project-wide virtual environment
├── requirements.txt            # Master NLP & LLM dependencies
├── setup_env.bat               # Windows one-click environment setup
├── setup_env.sh                # Linux/macOS one-click environment setup
├── .gitignore                  # Global ignore rules
├── README.md                   # Master NLP & LLM guide, curriculum & lab summaries
│
├── lab1/                       # Lab 1: Foundations of NLP & Language Modeling
│   ├── lab1.ipynb              # Executed laboratory notebook
│   └── README.md               # Detailed Lab 1 syllabus & theoretical guide
│
├── lab2/                       # Lab 2: Word Embeddings & Vector Spaces (Word2Vec & GloVe)
│   ├── lab2.ipynb              # Comprehensive Word2Vec & GloVe notebook
│   └── README.md               # Lab 2 theoretical foundations & guide
│
└── labX/                       # Future labs (Transformers, Pretraining, PEFT/LoRA, RAG, RLHF)
```

A single shared Jupyter kernel (**`Python (NLP-LLM)`**) is registered globally, powering all notebooks without redundant packages or duplicate multi-gigabyte environments.

---

## 🗺️ Master Curriculum: The Path from Classical NLP to Modern LLMs

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           THE NLP & LLM LEARNING JOURNEY                         │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ 1. Classical NLP & Tokenization (Lab 1)                                        │
 │    • Text cleaning, regex, POS tagging, NER, constituency parsing             │
 │    • Tokenization algorithms from scratch: BPE, WordPiece, SentencePiece/Unigram│
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ 2. Vector Spaces & Embeddings (Lab 1)                                         │
 │    • Static representations: One-Hot, Word2Vec (CBOW/Skip-gram), GloVe, FastText│
 │    • Contextual representations: Polysemy, ELMo BiLSTMs, BERT self-attention  │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ 3. Language Modeling Foundations & Metrics (Lab 1)                            │
 │    • N-gram MLE, Laplace smoothing, Katz backoff, autoregressive decoding     │
 │    • Quantitative evaluation: BLEU, ROUGE, BERTScore, Exact Match, Perplexity │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ 4. Deep Transformer Architecture (Lab 2)                                      │
 │    • Scaled dot-product attention, multi-head attention, positional encodings │
 │    • Encoder-only (BERT), Decoder-only (GPT), and Encoder-Decoder (T5)         │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ 5. Pretraining & Autoregressive LLMs (Lab 3)                                  │
 │    • Dataset tokenization, causal mask collation, cross-entropy training loop │
 │    • KV caching, flash attention, RoPE positional embeddings, temperature     │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ 6. Parameter-Efficient Fine-Tuning / PEFT (Lab 4)                             │
 │    • Low-Rank Adaptation (LoRA), QLoRA 4-bit quantization, prefix tuning      │
 │    • Instruction dataset formatting (Alpaca/ShareGPT) & SFT Trainer           │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ 7. Retrieval-Augmented Generation / RAG (Lab 5)                               │
 │    • Dense vector embeddings, chunking strategies, vector databases (Chroma/FAISS)
 │    • Hybrid search (BM25 + Dense), cross-encoder re-ranking, query expansion  │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ 8. LLM Alignment, Evaluation & Production (Lab 6)                             │
 │    • Direct Preference Optimization (DPO), RLHF, PPO, Constitutional AI       │
 │    • LLM-as-a-Judge, benchmark evaluation (MMLU, GSM8K, HumanEval)            │
 └───────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Laboratory Index & Summaries

### 📖 [Lab 1: Foundations of Natural Language Processing & Language Modeling](./lab1)
- **Status**: ✅ **Complete & Verified**
- **Core Topics**:
  1. **Text Processing Pipeline**: Regex normalization, contraction expanding, Porter stemming vs WordNet lemmatization, NLTK sentence/word segmentation, POS tagging, Named Entity Recognition with NLTK & spaCy (`en_core_web_sm`), Context-Free Grammar (CFG) parsing, Bag-of-Words & TF-IDF vectorization.
  2. **Modern Subword Tokenization**: Trade-offs between word, char, and subword granularity; **Byte Pair Encoding (BPE) from scratch**; **WordPiece** scoring; **SentencePiece & Unigram Language Model** with EM pruning; Special tokens (`[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`), Attention masks, Out-Of-Vocabulary handling.
  3. **Static Word Embeddings**: One-hot sparsity limits; **Word2Vec (Skip-gram and CBOW)**; **GloVe** co-occurrence factorization; **FastText** character n-grams for out-of-vocabulary word embeddings.
  4. **Contextual Embeddings**: The polysemy problem; **ELMo** learned layer-weighted BiLSTM representations; **BERT / RoBERTa** bidirectional self-attention mechanisms; **Masked Language Modeling (MLM 80/10/10 rule)**.
  5. **Language Modeling**: Bigram Maximum Likelihood Estimation; Laplace add-$k$ smoothing; Katz backoff; **Autoregressive decoding** with temperature; Perplexity computation on held-out corpora.
  6. **Evaluation Metrics**: Exact Match (EM), Precision, Recall, F1; **BLEU** (precision-oriented n-gram overlap); **ROUGE-1/2/L** (recall-oriented summarization metrics); **BERTScore** (semantic similarity via embeddings).
- **Notebook**: [`lab1/lab1.ipynb`](./lab1/lab1.ipynb)
- **Guide**: [`lab1/README.md`](./lab1/README.md)

---

### 📖 [Lab 2: Word Embeddings & Vector Spaces (Word2Vec & GloVe)](./lab2)
- **Status**: ✅ **Complete & Verified**
- **Core Topics**:
  1. **Corpus Preprocessing with NLTK**: Sentence tokenization, lowercase normalization, stopword filtering, and semantic domain structuring.
  2. **Word2Vec Deep Dive (Gensim)**: Continuous Bag-of-Words (CBOW) vs. Skip-gram with Negative Sampling (SGNS); semantic similarity; vector analogies ($\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$).
  3. **GloVe from First Principles**: Distance-weighted co-occurrence matrix $X$, power-law weighting function $f(X)$, and custom gradient descent optimization on the weighted least-squares loss.
  4. **Matrix Factorization with Scikit-Learn**: Positive Pointwise Mutual Information (PPMI) and TruncatedSVD for count-based distributed representations.
  5. **Comparative Analysis & 2D Visualization**: Cosine similarity benchmarking (`sklearn.metrics.pairwise.cosine_similarity`) and PCA projection (`sklearn.decomposition.PCA` + `matplotlib`).
- **Notebook**: [`lab2/lab2.ipynb`](./lab2/lab2.ipynb)
- **Guide**: [`lab2/README.md`](./lab2/README.md)

---

### 📖 [Lab 3: Transformer Fine-Tuning & Sentiment Analysis with DistilBERT](./lab3)
- **Status**: ✅ **Complete & Verified**
- **Core Topics**:
  1. **Knowledge Distillation & Model Architecture**: Student-Teacher architecture, DistilBERT vs BERT parameter reduction (66M vs 110M), 6-layer Transformer encoder, hidden representation geometry.
  2. **Data Cleansing & Sentiment Normalization**: `Twitter_Data.csv` auditing, missing value sanitization, 0-indexed label transformation (-1 Negative, 0 Neutral, 1 Positive), stratified subsampling controls.
  3. **Subword Tokenization Pipeline**: Fast WordPiece tokenization (`distilbert-base-uncased`), special tokens (`[CLS]`, `[SEP]`, `[PAD]`), attention masks, and dynamic padding collation with `DataCollatorWithPadding`.
  4. **Transfer Learning via Hugging Face Trainer**: Stratified 80/10/10 train-val-test split, AdamW optimizer with linear warmup, mixed-precision FP16 GPU acceleration, best checkpoint selection based on validation Macro F1.
  5. **Quantitative Model Evaluation**: Test set benchmarking, per-class precision/recall/F1 via `classification_report`, raw and normalized confusion matrix heatmaps (`seaborn` / `matplotlib`).
  6. **Serialization & Production Inference**: Model/tokenizer checkpoint export, high-level Hugging Face `pipeline` deployment, and interactive real-time prediction with softmax class probabilities.
- **Notebooks**: [`lab3/lab3.ipynb`](./lab3/lab3.ipynb) (Comprehensive) | [`lab3/lab31.ipynb`](./lab3/lab31.ipynb) (Code-only)
- **Guide**: [`lab3/README.md`](./lab3/README.md)

---

### 📖 [Lab 4: Parameter-Efficient Fine-Tuning (PEFT & LoRA)](./lab4)
- **Status**: 🚧 *Upcoming*
- **Focus Areas**: Supervised Fine-Tuning (SFT), LoRA rank decomposition matrices, QLoRA 4-bit normal float (NF4) quantization, memory optimization with Unsloth / Hugging Face PEFT.

---

### 📖 [Lab 5: Retrieval-Augmented Generation (RAG) Systems](./lab5)
- **Status**: 🚧 *Upcoming*
- **Focus Areas**: Semantic chunking, dense vector embeddings, vector indices (HNSW / FAISS), hybrid lexical & semantic retrieval, cross-encoder rerankers, hallucination mitigation.

---

### 📖 [Lab 6: LLM Alignment, Evaluation & Post-Training](./lab6)
- **Status**: 🚧 *Upcoming*
- **Focus Areas**: Direct Preference Optimization (DPO), Reinforcement Learning from Human Feedback (RLHF), LLM-as-a-Judge evaluation, automated benchmarking suites.

---

## ⚡ Quickstart: Global Environment Setup

The entire repository runs from the root virtual environment:

```bash
# 1. Clone the repository
git clone https://github.com/amit7451/nlp-llm.git
cd nlp-llm

# 2. Automated setup (Windows)
.\setup_env.bat

# Or Automated setup (Linux / macOS)
./setup_env.sh
```

### Manual Setup

```bash
# Create & activate environment
python -m venv .venv
.\.venv\Scripts\activate       # Windows PowerShell: .\.venv\Scripts\Activate.ps1
# source .venv/bin/activate    # Linux / macOS

# Install requirements & register global kernel
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m ipykernel install --user --name nlp-llm --display-name "Python (NLP-LLM)"

# Launch any laboratory notebook
jupyter notebook lab1/lab1.ipynb
# or
jupyter notebook lab2/lab2.ipynb
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
