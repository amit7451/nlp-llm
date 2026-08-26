# Natural Language Processing & Large Language Models (`nlp-llm`)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-154f30?style=for-the-badge)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8+-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

A structured, modular repository for hands-on **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** laboratories. Each lab is self-contained with its own isolated environment specifications, theoretical documentation, and executable Jupyter notebooks.

---

## 📂 Repository Architecture

```
nlp_llm/
│
├── .gitignore              # Standard multi-OS & Python/ML gitignore
├── README.md               # Root repository overview & quickstart
│
└── lab1/                   # Lab 1: NLP Foundations & Language Modeling
    ├── lab1.ipynb          # Verified end-to-end laboratory notebook
    ├── requirements.txt    # Dedicated dependencies for Lab 1
    └── README.md           # Syllabus, theoretical overview & run guide
```

---

## 🗺️ Laboratory Curriculum Roadmap

| Lab | Topic | Focus Areas | Status |
|:---:|---|---|:---:|
| **01** | [**Foundations of NLP & LLMs**](./lab1) | Text cleaning, POS tagging, NER, BPE & SentencePiece tokenization from scratch, Word2Vec, GloVe, FastText, ELMo, BERT bidirectional self-attention, N-gram LMs, Perplexity, BLEU/ROUGE/BERTScore | ✅ **Complete** |
| **02** | **Transformer Architecture from Scratch** | Multi-head self-attention, positional encoding, encoder-decoder transformers, causal masking | 🚧 *Upcoming* |
| **03** | **Pretraining & Causal Language Modeling** | Custom autoregressive pretraining, tokenizers, data collation, training loops | 🚧 *Upcoming* |
| **04** | **Parameter-Efficient Fine-Tuning (PEFT)** | LoRA, QLoRA, Prefix Tuning, Adapters on instruction datasets | 🚧 *Upcoming* |
| **05** | **Retrieval-Augmented Generation (RAG)** | Dense vector retrievers, vector databases, rerankers, grounded generation | 🚧 *Upcoming* |
| **06** | **LLM Alignment & Evaluation** | RLHF, DPO, PPO, LLM-as-a-Judge, benchmark evaluation (MMLU, GSM8K) | 🚧 *Upcoming* |

---

## ⚡ Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/amit7451/nlp-llm.git
cd nlp-llm
```

### 2. Set Up Virtual Environment

You can create an environment at the root level or independently inside each lab folder:

#### **Root Level Environment (Recommended)**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Upgrade pip and install Lab 1 requirements
python -m pip install --upgrade pip
pip install -r lab1/requirements.txt
```

#### **Per-Lab Isolated Environment**
```bash
cd lab1
python -m venv .venv

# Activate:
.\.venv\Scripts\Activate.ps1   # (Windows)
source .venv/bin/activate       # (Linux/macOS)

pip install -r requirements.txt
```

---

## 📓 Running the Notebooks

1. **Register the Jupyter Kernel**:
   ```bash
   python -m ipykernel install --user --name nlp-llm-lab1 --display-name "Python (NLP-LLM Lab1)"
   ```
2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```
3. Open `lab1/lab1.ipynb` and ensure the `Python (NLP-LLM Lab1)` or active `.venv` kernel is selected.

---

## 🛠️ Tech Stack & Dependencies

- **Data Processing**: `numpy`, `pandas`
- **Classical NLP & Linguistics**: `nltk`, `spacy`, `scikit-learn`
- **Subword Tokenization**: `sentencepiece`, `regex`
- **Embeddings & Vector Spaces**: `gensim` (Word2Vec, FastText)
- **Evaluation & Benchmarks**: `rouge-score`, `scikit-learn`
- **Interactive Runtime**: `jupyter`, `ipykernel`, `notebook`

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
