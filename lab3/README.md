# Lab 3: Sentiment Analysis with DistilBERT (Transformer Fine-Tuning)

An end-to-end, production-grade laboratory demonstrating deep transformer fine-tuning for multi-class sentiment analysis using **DistilBERT** (*Sanh et al., 2019*) on social media text (`Twitter_Data.csv`).

This lab covers dataset cleaning, subword tokenization with WordPiece, dynamic batch padding, PyTorch `Dataset` design, transfer learning with Hugging Face `Trainer`, quantitative evaluation (accuracy, macro F1, confusion matrices), artifact serialization, and a real-time inference pipeline.

---

## 📂 Laboratory Structure

```
lab3/
├── lab3.ipynb               # Comprehensive laboratory notebook (theory + code)
├── lab31.ipynb              # Concise code-only notebook (direct DistilBERT steps)
├── Twitter_Data.csv         # Raw dataset (162,980 social media posts)
├── requirements.txt         # Deep learning & transformer dependencies
├── setup_env.bat            # Windows setup script (delegates to root)
├── setup_env.sh             # Linux/macOS setup script (delegates to root)
├── create_lab3_notebook.py  # Standalone generator & verification script
└── README.md                # Mathematical foundations, architecture & guide
```

---

## 📚 Laboratory Syllabus & Modules

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DISTILBERT SENTIMENT ANALYSIS PIPELINE                          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 1. Environment & Hardware Detection (Block 0 - 2)                               │
   │    • GPU accelerator verification (NVIDIA CUDA / Kaggle T4 / P100 / CPU)        │
   │    • Universal seed configuration for deterministic reproducibility             │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 2. Dynamic Data Ingestion & Exploratory Analysis (Block 3 - 4)                  │
   │    • Automated path resolution across Kaggle input directories and local paths  │
   │    • Missing value auditing and class frequency distribution visualization      │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 3. Data Preprocessing & Label Normalization (Block 5 - 6)                       │
   │    • Text whitespace normalization and invalid record filtering                 │
   │    • 0-indexed integer mapping: -1 -> 0 (Negative), 0 -> 1 (Neutral), 1 -> 2    │
   │    • Stratified subsampling parameter (`SAMPLE_SIZE`) for fast Kaggle execution │
   │    • Stratified 80 / 10 / 10 Train / Validation / Test split                    │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 4. Subword Tokenization & PyTorch Data Structures (Block 7 - 8)                 │
   │    • DistilBertTokenizerFast (WordPiece subword segmentation, max_length=128)   │
   │    • Special tokens: [CLS] classification token and [SEP] boundary marker       │
   │    • Custom PyTorch SentimentDataset inheriting from torch.utils.data.Dataset   │
   │    • Dynamic padding collation using DataCollatorWithPadding                    │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 5. Model Architecture & Evaluation Metrics (Block 9)                            │
   │    • AutoModelForSequenceClassification with 3 output heads                     │
   │    • Parameter inspection: 66M parameters (vs 110M BERT baseline)               │
   │    • Custom compute_metrics returning Accuracy, Macro Precision, Recall, and F1 │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 6. Transfer Learning with Hugging Face Trainer (Block 10)                       │
   │    • AdamW optimizer with weight decay (0.01) and 10% linear warmup schedule    │
   │    • Automated Mixed Precision (FP16) on CUDA accelerators                      │
   │    • Best checkpoint selection based on validation Macro F1 score               │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 7. Quantitative Evaluation & Test Benchmarking (Block 11)                       │
   │    • Test set inference and full per-class classification report                │
   │    • Normalized and raw Confusion Matrix heatmaps via Seaborn                   │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 8. Model Serialization & Live Real-Time Inference (Block 12 - 13)               │
   │    • Model weights, tokenizer vocabulary, and configuration export to disk      │
   │    • High-level Hugging Face pipeline instantiation                             │
   │    • Custom predict_tweet_sentiment() with softmax class probabilities          │
   └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical & Architectural Foundations

### 1. Transformer Self-Attention Mechanism

Each layer of DistilBERT computes **Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where:
- Queries $Q = X W^Q$, Keys $K = X W^K$, Values $V = X W^V$ with learned projection matrices $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$.
- $d_k = d_{\text{model}} / h = 768 / 12 = 64$.
- The scaling factor $\frac{1}{\sqrt{d_k}}$ prevents the dot products from growing excessively large in high dimensions, which would cause the softmax gradient to vanish.

**Multi-Head Attention (MHA)** runs $h = 12$ attention heads in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

---

### 2. DistilBERT: Knowledge Distillation

DistilBERT (*Sanh et al., 2019*) is trained using a student-teacher knowledge distillation objective where a smaller student network (6 Transformer layers) learns to replicate the behavior of a larger teacher network (`bert-base-uncased`, 12 Transformer layers).

#### Temperature-Scaled Softmax
Rather than training solely on hard labels ($y \in \{0, 1\}$), the student matches the soft probability distribution generated by the teacher at temperature $T$:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where higher temperature values ($T > 1$) reveal the "dark knowledge" — the rich relative probabilities assigned to incorrect classes.

#### Triple Loss Objective
DistilBERT is optimized with a linear combination of three complementary losses:

$$\mathcal{L} = \alpha_{ce} \mathcal{L}_{ce} + \alpha_{mlm} \mathcal{L}_{mlm} + \alpha_{cos} \mathcal{L}_{cos}$$

1. **Distillation Loss ($\mathcal{L}_{ce}$)**: Cross-entropy between student probabilities $q(T)$ and teacher probabilities $p(T)$:
   $$\mathcal{L}_{ce} = - \sum_{i} p_i(T) \log q_i(T)$$
2. **Masked Language Modeling Loss ($\mathcal{L}_{mlm}$)**: Standard BERT cross-entropy on hard ground-truth tokens.
3. **Cosine Embedding Loss ($\mathcal{L}_{cos}$)**: Align the directional orientation of student and teacher hidden vectors:
   $$\mathcal{L}_{cos} = 1 - \cos(h_s, h_t) = 1 - \frac{h_s \cdot h_t}{\|h_s\|_2 \|h_t\|_2}$$

#### Architectural Differences: DistilBERT vs BERT
- **Removed Layers**: Layer count halved from 12 to 6 (initialized by taking 1 layer out of every 2 from BERT).
- **Removed Embeddings**: Token-type (segment) embeddings are omitted.
- **Removed Pooler**: The classification pooler is eliminated; predictions are projected directly from the final hidden state of the `[CLS]` token.
- **Result**: **40% fewer parameters** (66M vs 110M), **60% faster inference**, and retains **97%** of full BERT GLUE benchmark performance.

---

### 3. Classification Head & Cross-Entropy Loss

For 3-class sequence classification:
1. The pooled representation $h_{[\text{CLS}]} \in \mathbb{R}^{768}$ is extracted from the final transformer layer.
2. A dense projection layer maps representations to raw logits:
   $$z = W_c \cdot \text{Dropout}(\text{ReLU}(W_p h_{[\text{CLS}]} + b_p)) + b_c \in \mathbb{R}^3$$
3. Categorical Cross-Entropy Loss is minimized during fine-tuning:
   $$\mathcal{L}_{CE} = - \sum_{c=0}^{2} y_c \log \left( \frac{\exp(z_c)}{\sum_{j=0}^{2} \exp(z_j)} \right)$$

---

### 4. Dynamic Padding vs Static Padding

| Strategy | Token Length | Throughput | Memory Footprint |
| :--- | :---: | :---: | :---: |
| **Static Padding** | Fixed to `max_length = 128` | Lower (wasteful attention over `[PAD]`) | High (unneeded pad tokens) |
| **Dynamic Padding (`DataCollatorWithPadding`)** | Padded to max length of *current batch* | **30%–50% Faster** | **Optimized VRAM** |

---

## ⚡ Kaggle Step-by-Step Deployment Guide

Follow these steps to run this notebook directly on **Kaggle**:

### Step 1: Upload Notebook to Kaggle
1. Go to [kaggle.com](https://www.kaggle.com) and log in.
2. Click **+ Create** -> **New Notebook**.
3. In the notebook editor, click **File** -> **Import Notebook** -> Select [`lab3/lab3.ipynb`](./lab3.ipynb).

### Step 2: Enable GPU Hardware Acceleration
1. Look at the right-side control panel under **Notebook options**.
2. Click **Accelerator** dropdown.
3. Select **GPU T4 x2** (or **GPU P100**).
4. Verify Internet access is **On** (under Notebook options -> Internet).

### Step 3: Attach the Dataset (`Twitter_Data.csv`)
1. On the right-side panel, click **+ Add Input** (or **Upload**).
2. Choose **Upload New Dataset**.
3. Upload `Twitter_Data.csv` (drag and drop from `lab3/Twitter_Data.csv`).
4. Set the dataset name (e.g., `twitter-data`) and click **Create**.
5. *Note: The notebook's built-in `find_dataset()` function will automatically locate the CSV inside `/kaggle/input`!*

### Step 4: Run the Notebook Blocks
1. Click **Run All** (or execute blocks sequentially using `Shift + Enter`).
2. **Block 5 Config**: By default, `SAMPLE_SIZE = 25000` is set for rapid execution (~3 minutes on GPU). If you want to train on all 162k rows, change `SAMPLE_SIZE = None`.
3. Check **Block 11** for the Confusion Matrix and classification report.
4. Try custom text prompts in **Block 13** to test real-time sentiment predictions!

### Step 5: Export Model Artifacts
1. When Block 12 runs, your model is saved to `./distilbert_sentiment_model`.
2. Access it in the right sidebar under **Output** -> `/kaggle/working/distilbert_sentiment_model`.
3. You can download the trained weights (`model.safetensors` and `config.json`) directly.

---

## 🚀 Local Environment Execution

To run Lab 3 locally within the repository's virtual environment:

### Windows:
```cmd
cd lab3
setup_env.bat
jupyter notebook lab3.ipynb
```

### Linux / macOS:
```bash
cd lab3
bash setup_env.sh
jupyter notebook lab3.ipynb
```

Ensure the **`Python (NLP-LLM)`** kernel is selected.

---

## 📊 Benchmark Reference & Metrics

Trained on stratified Twitter sentiment data:

| Metric | Target Baseline (2 Epochs) | Full Corpus (3 Epochs) |
| :--- | :---: | :---: |
| **Accuracy** | $\ge 88.5\%$ | $\approx 92.4\%$ |
| **Macro Precision** | $\ge 87.2\%$ | $\approx 91.8\%$ |
| **Macro Recall** | $\ge 86.8\%$ | $\approx 91.5\%$ |
| **Macro F1-Score** | $\ge 87.0\%$ | $\approx 91.6\%$ |
| **Inference Latency** | $\approx 12\text{ ms / tweet}$ (T4 GPU) | $\approx 12\text{ ms / tweet}$ |

---

## 🔗 References
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108).
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
- Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020.
