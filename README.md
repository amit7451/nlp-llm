# Natural Language Processing & Large Language Models (`nlp-llm`)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8+-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end, rigorous, hands-on repository for mastering **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** — spanning statistical linguistics, tokenizer algorithms from scratch, distributed embeddings, Transformer architectures, autoregressive pretraining, parameter-efficient fine-tuning (PEFT/LoRA), Retrieval-Augmented Generation (RAG), and post-training alignment (RLHF/DPO).

---

## 🏛️ Modular Repository Architecture

To maintain strict isolation, reproducibility, and prevent dependency version conflicts across diverse machine learning toolchains, **every laboratory in this repository is completely self-contained**:

```
nlp_llm/
│
├── .gitignore                  # Global ignore rules (covers all lab sub-environments)
├── README.md                   # Global NLP & LLM guide, curriculum & lab summaries
│
├── lab1/                       # Lab 1: Foundations of NLP & LLMs
│   ├── .gitignore              # Lab 1 isolated ignore rules
│   ├── lab1.ipynb              # Complete, executed laboratory notebook
│   ├── requirements.txt        # Dedicated dependencies for Lab 1
│   ├── setup_env.bat           # Windows one-click environment setup
│   ├── setup_env.sh            # Linux/macOS one-click environment setup
│   └── README.md               # Detailed Lab 1 syllabus & theoretical guide
│
├── lab2/                       # Lab 2: Transformers from Scratch (Planned)
│   ├── .gitignore
│   ├── requirements.txt
│   └── ...
│
└── labX/                       # Future labs (Autonomous & Self-Contained)
```

Each lab directory maintains:
- Its own isolated virtual environment (`labX/.venv/`).
- Its own pinned dependencies (`requirements.txt`).
- Its own local `.gitignore` and OS setup automation scripts (`setup_env.bat` / `setup_env.sh`).
- Its own comprehensive theoretical documentation and syllabus (`README.md`).

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

### 📖 [Lab 2: Transformer Architecture from Scratch](./lab2)
- **Status**: 🚧 *Upcoming*
- **Focus Areas**: Scaled dot-product attention, multi-head attention mechanisms, sinusoidal & rotary positional embeddings (RoPE), feed-forward sublayers, pre-LN vs post-LN residual connections, causal masking for autoregressive decoders.

---

### 📖 [Lab 3: Pretraining & Causal Language Modeling](./lab3)
- **Status**: 🚧 *Upcoming*
- **Focus Areas**: Building a custom GPT-style decoder, training dataset pipelines, tokenized chunk collation, next-token cross-entropy loss, KV cache acceleration, top-$k$ / top-$p$ (nucleus) sampling.

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

## ⚡ Quickstart: Working with Any Lab

Each lab is standalone. To work on any lab (e.g. `lab1`):

```bash
# 1. Clone the repository
git clone https://github.com/amit7451/nlp-llm.git
cd nlp-llm

# 2. Enter the target lab directory
cd lab1

# 3. Quick setup (Windows)
.\setup_env.bat

# Or Quick setup (Linux / macOS)
./setup_env.sh
```

### Manual Setup for Any Lab

```bash
cd lab1
python -m venv .venv

# Activate:
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install requirements & register kernel
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m ipykernel install --user --name nlp-llm-lab1 --display-name "Python (NLP-LLM Lab 1)"

# Run Notebook
jupyter notebook lab1.ipynb
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
