import json
import os

cells = []

def add_markdown(source_text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.splitlines()]
    })

def add_code(source_text):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in source_text.splitlines()]
    })

# ==============================================================================
# Cell 1: Header & Theoretical Foundations
# ==============================================================================
add_markdown("""# Natural Language Processing & Large Language Models
## Lab 3: Sentiment Analysis with DistilBERT (Transformer Fine-Tuning)

---

### 📌 Laboratory Overview & Objectives
Welcome to **Lab 3**. In this lab, we build an end-to-end, production-grade text classification system using **DistilBERT** (*Sanh et al., 2019*) to classify public social sentiment into **Negative (-1)**, **Neutral (0)**, and **Positive (1)** categories.

This notebook is designed to be **complete, modular, and self-contained**, optimized for running either locally or in cloud accelerators like **Kaggle** (GPU T4/P100) or **Google Colab**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DISTILBERT SENTIMENT ANALYSIS PIPELINE                          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 1. Environment & Hardware Detection (CUDA / GPU T4 / P100 / CPU)                │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 2. Dynamic Dataset Ingestion & Exploratory Analysis (Twitter_Data.csv)          │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 3. Data Cleansing, 0-Indexed Label Mapping & Stratified Splitting (80/10/10)    │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 4. Subword Tokenization with WordPiece (DistilBertTokenizerFast)                │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 5. PyTorch Dataset & Dynamic Padding Collation (DataCollatorWithPadding)        │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 6. Model Instantiation (AutoModelForSequenceClassification: 3 classes)          │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 7. Transfer Learning via Hugging Face Trainer (AdamW, FP16, Linear Warmup)      │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 8. Quantitative Evaluation (Macro F1, Precision, Recall, Confusion Matrix)      │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │ 9. Model Serialization & Live Real-Time Inference Pipeline                      │
   └─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 🧠 Why DistilBERT?
Standard **BERT** (`bert-base-uncased`, 110M parameters) delivers exceptional language representations but is computationally heavy for real-time inference and training on constrained GPUs. **DistilBERT** applies **Knowledge Distillation** during pre-training:
- **40% fewer parameters** (66M vs 110M) by retaining 6 Transformer layers instead of 12.
- **60% faster inference** latency while preserving **97%** of BERT's performance on the GLUE benchmark.
- Uses a triple distillation loss:
  $$\\mathcal{L} = \\alpha_{ce} \\mathcal{L}_{ce} + \\alpha_{mlm} \\mathcal{L}_{mlm} + \\alpha_{cos} \\mathcal{L}_{cos}$$
  where $\\mathcal{L}_{ce}$ aligns the student's soft output probabilities with the teacher's temperature-scaled softmax:
  $$p_i = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$$

---

### ⚡ Running in Kaggle
1. **Enable GPU**: On Kaggle, open the right sidebar -> **Notebook options** -> **Accelerator** -> Select **GPU T4 x2** or **GPU P100**.
2. **Add Data**: Ensure `Twitter_Data.csv` is uploaded to the notebook inputs. This notebook includes an **automated path resolver** that finds the dataset automatically regardless of your input directory structure.""")

# ==============================================================================
# Block 0: Dependency Installation
# ==============================================================================
add_markdown("""---
## Block 0: Dependency Installation & Environment Check

If running on a fresh environment, uncomment and execute the pip installation command below. On Kaggle and Colab, PyTorch and Hugging Face are generally pre-installed.""")

add_code("""# Uncomment the command below if dependencies are not already installed:
# !pip install -q transformers datasets accelerate scikit-learn matplotlib seaborn""")

# ==============================================================================
# Block 1: Imports & Reproducibility
# ==============================================================================
add_markdown("""---
## Block 1: Core Imports & Reproducibility Setup

We import our data science, deep learning, and NLP libraries, and set universal random seeds across Python, NumPy, and PyTorch to guarantee reproducible experimental runs.""")

add_code("""import os
import sys
import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline,
    set_seed
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Set global seed for complete reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)

seed_everything(42)

print(f"PyTorch Version     : {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")
print(f"Pandas Version      : {pd.__version__}")
print(f"NumPy Version       : {np.__version__}")""")

# ==============================================================================
# Block 2: Hardware Accelerator Configuration
# ==============================================================================
add_markdown("""---
## Block 2: Hardware Acceleration Detection

Transformers require significant compute for matrix multiplications during backpropagation. We detect whether an NVIDIA CUDA GPU is available and display device hardware specifications.""")

add_code("""# Detect and configure compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Compute Device: {device}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cuda_arch = torch.cuda.get_device_capability(0)
    print(f"GPU Accelerator : {gpu_name}")
    print(f"Dedicated VRAM  : {vram_gb:.2f} GB")
    print(f"Compute Cap.    : {cuda_arch[0]}.{cuda_arch[1]}")
    print(f"Device Count    : {torch.cuda.device_count()}")
else:
    print("WARNING: CUDA is not available. Running on CPU.")
    print("If on Kaggle: Settings -> Accelerator -> select 'GPU T4 x2' or 'GPU P100'.")""")

# ==============================================================================
# Block 3: Dynamic Dataset Acquisition & Path Resolution
# ==============================================================================
add_markdown("""---
## Block 3: Dynamic Dataset Loading

To avoid path errors across different Kaggle input configurations and local directories, the function below inspects common paths and traverses `/kaggle/input` recursively to locate `Twitter_Data.csv`.""")

add_code("""def find_dataset(filename="Twitter_Data.csv"):
    \"\"\"
    Searches common Kaggle and local paths to locate the target CSV file.
    \"\"\"
    candidate_paths = [
        filename,
        os.path.join(".", filename),
        os.path.join("..", filename),
        os.path.join("lab3", filename),
        os.path.join("..", "lab3", filename),
        os.path.join("/kaggle", "input", filename),
        os.path.join("/kaggle", "input", "twitter-data", filename),
        os.path.join("/kaggle", "input", "twitter-and-reddit-sentimental-analysis-dataset", filename),
        os.path.join("/kaggle", "working", filename)
    ]
    
    # 1. Direct candidate checks
    for path in candidate_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
            
    # 2. Recursive search inside /kaggle/input if running in Kaggle
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, _, files in os.walk(kaggle_input):
            if filename in files:
                return os.path.join(root, filename)
                
    return None

data_path = find_dataset("Twitter_Data.csv")

if data_path:
    print(f"✓ Dataset found at: {data_path}")
    df_raw = pd.read_csv(data_path)
    print(f"Loaded {len(df_raw):,} records with columns: {list(df_raw.columns)}")
else:
    raise FileNotFoundError(
        "Could not locate 'Twitter_Data.csv'. Please upload the dataset to your Kaggle input or current working directory."
    )""")

# ==============================================================================
# Block 4: Exploratory Data Analysis & Text Inspection
# ==============================================================================
add_markdown("""---
## Block 4: Exploratory Data Analysis (EDA)

We audit the dataset for null values, review data types, and inspect the distribution of target sentiment classes:
- **-1**: Negative
- **0**: Neutral
- **1**: Positive""")

add_code("""# Display overview and sample records
print("--- Dataset Shape & Information ---")
print(f"Total Rows: {df_raw.shape[0]:,}, Total Columns: {df_raw.shape[1]}")
print("\\n--- Missing Value Count ---")
print(df_raw.isna().sum())

print("\\n--- First 5 Records ---")
display_fn = globals().get('display', print)
display_fn(df_raw.head())

print("\\n--- Raw Category Frequencies ---")
category_counts = df_raw['category'].value_counts(dropna=False)
print(category_counts)

# Visualize Category Distribution
plt.figure(figsize=(8, 4))
palette = {-1.0: '#e74c3c', 0.0: '#95a5a6', 1.0: '#2ecc71'}
ax = sns.countplot(
    data=df_raw.dropna(subset=['category']), 
    x='category', 
    palette=palette,
    order=[-1.0, 0.0, 1.0]
)
plt.title("Sentiment Category Distribution (Raw Twitter_Data)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Category (-1: Negative, 0: Neutral, 1: Positive)", fontsize=11)
plt.ylabel("Tweet Count", fontsize=11)
for p in ax.patches:
    h = p.get_height()
    if not np.isnan(h) and h > 0:
        ax.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()""")

# ==============================================================================
# Block 5: Cleaning, Label Mapping & Stratified Subsampling
# ==============================================================================
add_markdown("""---
## Block 5: Data Cleaning & Label Mapping

### 1. Label Normalization
PyTorch loss functions (`CrossEntropyLoss`) expect labels to be non-negative integers in the range $[0, C-1]$.
We map:
- ` -1 ` (Negative) $\\rightarrow$ `0`
- `  0 ` (Neutral)  $\\rightarrow$ `1`
- `  1 ` (Positive) $\\rightarrow$ `2`

We also construct explicit bidirectional mapping dictionaries (`ID2LABEL` and `LABEL2ID`) which integrate directly into Hugging Face model configs.

### 2. Configurable Working Subset
The full dataset contains **~162,980** tweets. While training on the full dataset is supported, it can take 20–30 minutes. We provide a configurable `SAMPLE_SIZE` (default: **25,000**), which samples proportionally across all 3 classes for fast experimentation in under 4 minutes. Set `SAMPLE_SIZE = None` to train on the complete corpus.""")

add_code("""# 1. Clean missing values and blank texts
df_clean = df_raw.dropna(subset=['clean_text', 'category']).copy()
df_clean['category'] = df_clean['category'].astype(int)
df_clean['clean_text'] = df_clean['clean_text'].astype(str).str.strip()

# Filter out empty strings
df_clean = df_clean[df_clean['clean_text'].str.len() > 0].reset_index(drop=True)
print(f"Cleaned records: {len(df_clean):,} (dropped {len(df_raw) - len(df_clean):,} invalid rows)")

# 2. Define label mappings for PyTorch and Transformers
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
REVERSE_MAP = {0: -1, 1: 0, 2: 1}

ID2LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
LABEL2ID = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

df_clean['label'] = df_clean['category'].map(LABEL_MAP)

# --------------------------------------------------------------------------
# Configuration: Training Subset Size
# --------------------------------------------------------------------------
# Set SAMPLE_SIZE = 25000 for rapid iteration (~3 mins on T4 GPU)
# Set SAMPLE_SIZE = None to train on the entire 162k dataset (~20 mins on T4 GPU)
SAMPLE_SIZE = 25000  

if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df_clean):
    print(f"\\nCreating stratified sample of {SAMPLE_SIZE:,} tweets for efficient training...")
    df_working = df_clean.groupby('label', group_keys=False).apply(
        lambda group: group.sample(
            int(np.round(SAMPLE_SIZE * len(group) / len(df_clean))),
            random_state=42
        )
    ).reset_index(drop=True)
else:
    print(f"\\nUsing full corpus of {len(df_clean):,} tweets...")
    df_working = df_clean.copy()

print(f"Working DataFrame Shape: {df_working.shape}")
print("\\nClass Distribution in Working Set:")
for label_id, name in ID2LABEL.items():
    count = (df_working['label'] == label_id).sum()
    pct = (count / len(df_working)) * 100
    print(f"  Class {label_id} ({name:<8}): {count:>6,} tweets ({pct:.2f}%)")""")

# ==============================================================================
# Block 6: Stratified Train / Validation / Test Splitting
# ==============================================================================
add_markdown("""---
## Block 6: Stratified Train / Validation / Test Splitting

We perform a **Stratified 80 / 10 / 10 Split**:
- **Train Set (80%)**: Used by the optimizer to update model weights.
- **Validation Set (10%)**: Monitored after each epoch for learning rate scheduling and early stopping/best model checkpoint selection.
- **Test Set (10%)**: Held out completely to assess generalizability on unseen data.""")

add_code("""# Step 1: Split into 80% Train and 20% Temp (Val + Test)
train_df, temp_df = train_test_split(
    df_working,
    test_size=0.20,
    random_state=42,
    stratify=df_working['label']
)

# Step 2: Split Temp evenly into 10% Validation and 10% Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df['label']
)

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train Set      : {len(train_df):>6,} samples ({len(train_df)/len(df_working)*100:.1f}%)")
print(f"Validation Set : {len(val_df):>6,} samples ({len(val_df)/len(df_working)*100:.1f}%)")
print(f"Test Set       : {len(test_df):>6,} samples ({len(test_df)/len(df_working)*100:.1f}%)")""")

# ==============================================================================
# Block 7: Subword Tokenization with DistilBertTokenizerFast
# ==============================================================================
add_markdown("""---
## Block 7: Subword Tokenization with DistilBERT

We instantiate `AutoTokenizer.from_pretrained('distilbert-base-uncased')`.
- DistilBERT uses **WordPiece tokenization** with a 30,522 token vocabulary.
- It injects special tokens:
  - `[CLS]` (ID 101): Classification token whose final hidden state represents the sequence embedding.
  - `[SEP]` (ID 102): Sentence separator token.
  - `[PAD]` (ID 0): Padding token for batch length alignment.
- Tweets are relatively short; setting `MAX_LENGTH = 128` comfortably encapsulates >99% of tweet sequences without wasteful padding computations.""")

add_code("""MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 128

print(f"Loading tokenizer: {MODEL_CHECKPOINT}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# Inspect tokenization on an example tweet
sample_text = train_df['clean_text'].iloc[0]
sample_encoded = tokenizer(sample_text, max_length=MAX_LENGTH, truncation=True)

print("\\n--- Tokenizer Demonstration ---")
print(f"Raw Text       : {sample_text[:85]}...")
print(f"Subword Tokens : {tokenizer.convert_ids_to_tokens(sample_encoded['input_ids'][:12])}")
print(f"Token IDs      : {sample_encoded['input_ids'][:12]}")
print(f"Attention Mask : {sample_encoded['attention_mask'][:12]}")

# Batch tokenization across splits
print("\\nTokenizing Train, Validation, and Test corpora...")
train_encodings = tokenizer(train_df['clean_text'].tolist(), truncation=True, max_length=MAX_LENGTH)
val_encodings = tokenizer(val_df['clean_text'].tolist(), truncation=True, max_length=MAX_LENGTH)
test_encodings = tokenizer(test_df['clean_text'].tolist(), truncation=True, max_length=MAX_LENGTH)
print("✓ Tokenization completed successfully.")""")

# ==============================================================================
# Block 8: PyTorch Dataset & Dynamic Padding Collation
# ==============================================================================
add_markdown("""---
## Block 8: PyTorch Dataset & Dynamic Padding

We implement a custom `SentimentDataset` inheriting from `torch.utils.data.Dataset`.
Instead of static padding upfront (which pads every sequence in the dataset to `MAX_LENGTH = 128`), we utilize `DataCollatorWithPadding`.
- **Dynamic Padding**: Batches are padded dynamically to the longest sequence *in that specific mini-batch*.
- This drastically reduces unnecessary tensor computations and accelerates training throughput by ~30–50%.""")

add_code("""class SentimentDataset(Dataset):
    \"\"\"
    PyTorch Dataset wrapper around tokenized Hugging Face encodings and labels.
    \"\"\"
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_df['label'])
val_dataset = SentimentDataset(val_encodings, val_df['label'])
test_dataset = SentimentDataset(test_encodings, test_df['label'])

# Dynamic collator pads inputs dynamically to each batch's maximum sequence length
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(f"Constructed PyTorch Datasets:")
print(f"  • Train Samples : {len(train_dataset):,}")
print(f"  • Val Samples   : {len(val_dataset):,}")
print(f"  • Test Samples  : {len(test_dataset):,}")""")

# ==============================================================================
# Block 9: Model Initialization & Metrics Formulation
# ==============================================================================
add_markdown("""---
## Block 9: Model Instantiation & Evaluation Metrics

We load `AutoModelForSequenceClassification` configured with `num_labels=3`, passing `id2label` and `label2id`.

### Metric Definition
In sentiment analysis, accuracy alone can be misleading due to class imbalance. We monitor:
- **Accuracy**: Overall fraction of correct predictions.
- **Macro F1-Score**: Unweighted mean of F1 scores across Negative, Neutral, and Positive classes (ensures underrepresented classes are prioritized equally).
- **Macro Precision & Recall**: Provides insight into false positives vs false negatives.""")

add_code("""print(f"Loading pretrained architecture: {MODEL_CHECKPOINT} with 3 output heads...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# Inspect parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Model Parameters     : {total_params:,}")
print(f"Trainable Parameters       : {trainable_params:,}")

def compute_metrics(eval_pred):
    \"\"\"
    Calculates quantitative evaluation metrics for validation/test phases.
    \"\"\"
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='macro', 
        zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': float(acc),
        'f1_macro': float(f1),
        'precision_macro': float(precision),
        'recall_macro': float(recall)
    }""")

# ==============================================================================
# Block 10: Training Configuration & Execution via Hugging Face Trainer
# ==============================================================================
add_markdown("""---
## Block 10: Training Configuration & Execution

We configure `TrainingArguments` optimized for Kaggle GPU environments:
- **Learning Rate (`2e-5`)**: Standard learning rate for fine-tuning pre-trained transformer backbones.
- **Weight Decay (`0.01`)**: Regularizes weights to prevent overfitting.
- **Epochs (`2` or `3`)**: Sufficient for convergence without catastrophic forgetting.
- **Mixed Precision (`fp16=True`)**: Uses 16-bit floating-point numbers on CUDA, doubling compute speed and halving VRAM usage.
- **`load_best_model_at_end=True`**: Automatically reverts to the best checkpoint based on validation **Macro F1**.""")

add_code("""# Detect compatibility for evaluation strategy across transformers versions
eval_strategy_key = "eval_strategy" if hasattr(TrainingArguments, "eval_strategy") else "evaluation_strategy"

training_kwargs = {
    "output_dir": "./distilbert_checkpoints",
    "num_train_epochs": 2,                       # 2 epochs provide high convergence with minimal compute
    "per_device_train_batch_size": 32 if torch.cuda.is_available() else 16,
    "per_device_eval_batch_size": 32 if torch.cuda.is_available() else 16,
    "warmup_ratio": 0.1,                         # 10% warmup schedule for AdamW
    "weight_decay": 0.01,                        # L2 penalty
    "learning_rate": 2e-5,                       # Recommended transfer learning rate
    "logging_dir": "./logs",
    "logging_steps": 50,
    "save_strategy": "epoch",
    eval_strategy_key: "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1_macro",
    "greater_is_better": True,
    "save_total_limit": 1,                       # Conserve Kaggle disk storage
    "fp16": torch.cuda.is_available(),           # Mixed precision if GPU is active
    "report_to": "none",                         # Disable wandb/mlflow prompts
    "seed": 42
}

training_args = TrainingArguments(**training_kwargs)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("🚀 Starting DistilBERT fine-tuning pipeline...")
train_output = trainer.train()

print("\\n✓ Fine-tuning completed!")
print(f"Total Training Runtime : {train_output.metrics.get('train_runtime', 0):.2f} seconds")
print(f"Final Training Loss    : {train_output.training_loss:.4f}")""")

# ==============================================================================
# Block 11: Quantitative Evaluation & Test Set Benchmarking
# ==============================================================================
add_markdown("""---
## Block 11: Quantitative Evaluation & Error Analysis

We now evaluate our fine-tuned DistilBERT model on the **held-out test set** (never seen during training or validation).
We compute:
1. Aggregate metrics (**Accuracy, Macro F1, Precision, Recall**).
2. Per-class metrics via `classification_report`.
3. Normalized and raw **Confusion Matrices** via Seaborn heatmaps.""")

add_code("""# 1. Run evaluation on test dataset
print("--- Held-Out Test Set Metrics ---")
test_eval = trainer.evaluate(test_dataset)
for key, val in test_eval.items():
    if any(k in key for k in ["loss", "accuracy", "f1", "precision", "recall"]):
        print(f"  • {key:<20}: {val:.4f}")

# 2. Extract predictions
predictions_raw = trainer.predict(test_dataset)
y_pred = np.argmax(predictions_raw.predictions, axis=-1)
y_true = np.array(test_dataset.labels)

# 3. Detailed classification report
target_labels = [ID2LABEL[i] for i in range(3)]
print("\\n--- Per-Class Classification Report ---")
print(classification_report(y_true, y_pred, target_names=target_labels, digits=4))

# 4. Confusion Matrix Heatmaps
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Raw count confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_labels, yticklabels=target_labels, ax=ax1)
ax1.set_title("Test Confusion Matrix (Absolute Counts)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Predicted Class", fontsize=11)
ax1.set_ylabel("True Class", fontsize=11)

# Normalized confusion matrix
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', xticklabels=target_labels, yticklabels=target_labels, ax=ax2)
ax2.set_title("Test Confusion Matrix (Normalized Accuracy)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Predicted Class", fontsize=11)
ax2.set_ylabel("True Class", fontsize=11)

plt.tight_layout()
plt.show()""")

# ==============================================================================
# Block 12: Model Serialization & Artifact Export
# ==============================================================================
add_markdown("""---
## Block 12: Model Serialization & Artifact Export

We save the fine-tuned model weights, tokenizer configuration, and vocabulary to disk. In Kaggle, files written to `./` or `/kaggle/working` can be downloaded directly from the Kaggle file explorer or committed as model artifacts.""")

add_code("""EXPORT_PATH = "./distilbert_sentiment_model"
print(f"Exporting model and tokenizer to: {EXPORT_PATH}")

# Save PyTorch weights and Hugging Face configuration
model.save_pretrained(EXPORT_PATH)
tokenizer.save_pretrained(EXPORT_PATH)

print("\\nSaved Artifacts:")
for item in sorted(os.listdir(EXPORT_PATH)):
    item_path = os.path.join(EXPORT_PATH, item)
    size_mb = os.path.getsize(item_path) / (1024 * 1024)
    print(f"  ✓ {item:<28} ({size_mb:>6.2f} MB)")""")

# ==============================================================================
# Block 13: End-to-End Real-Time Inference Pipeline
# ==============================================================================
add_markdown("""---
## Block 13: Live Inference Pipeline & Interactive Testing

We demonstrate production deployment using Hugging Face's high-level `pipeline` API and a dedicated inference function that extracts full softmax probability distributions for custom text inputs.""")

add_code("""# Instantiate inference pipeline from exported directory
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=EXPORT_PATH,
    tokenizer=EXPORT_PATH,
    device=0 if torch.cuda.is_available() else -1
)

def predict_tweet_sentiment(text):
    \"\"\"
    Performs inference on a text string and prints softmax confidence scores.
    \"\"\"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    if torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_class_id = int(np.argmax(probabilities))
        
    predicted_label = ID2LABEL[pred_class_id]
    confidence = probabilities[pred_class_id]
    
    print(f"\\nInput Tweet: '{text}'")
    print(f"Prediction : {predicted_label} (Confidence: {confidence:.2%})")
    print("Class Probabilities:")
    for cid, cname in ID2LABEL.items():
        bar = "█" * int(probabilities[cid] * 25)
        print(f"  {cname:<10}: {probabilities[cid]:>6.2%} | {bar}")

# Test suite with realistic samples
benchmark_samples = [
    "Fantastic performance by the team! The new economic reforms are yielding phenomenal results.",
    "The official press release announced the parliament meeting scheduled for 10 AM tomorrow.",
    "Very disappointed with the poor infrastructure, delayed flights, and terrible customer service.",
    "A remarkable milestone achieved today, congratulations to everyone involved!",
    "The ministry held a review meeting regarding national railway network maintenance."
]

print("=== Running Live Sentiment Predictions ===")
for sample in benchmark_samples:
    predict_tweet_sentiment(sample)""")

# ==============================================================================
# Block 14: Laboratory Summary, Theoretical Insights & Best Practices
# ==============================================================================
add_markdown("""---
## Block 14: Laboratory Summary & Key Insights

### 📊 Model Architecture Comparison

| Model Architecture | Parameters | Transformer Layers | Hidden Dimension | Relative Speed | GLUE Score Retention |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BERT-Base** | 110M | 12 | 768 | $1.0\\times$ (Baseline) | $100\\%$ |
| **DistilBERT** | **66M** | **6** | **768** | **$\\approx 1.6\\times$ Faster** | **$\\approx 97\\%$** |
| **RoBERTa-Base** | 125M | 12 | 768 | $0.9\\times$ | $103\\%$ |
| **TinyBERT (4L)** | 14.5M | 4 | 312 | $\\approx 7.5\\times$ Faster | $\\approx 92\\%$ |

---

### 💡 Key Takeaways
1. **Knowledge Distillation in NLP**: DistilBERT leverages soft probability matching (Student-Teacher distillation) during pre-training, dropping token-type embeddings and the pooler while halving the layer count.
2. **Dynamic Padding Efficiency**: Using `DataCollatorWithPadding` instead of static maximum padding prevents wasted compute on short sequences, leading to significant batch throughput gains.
3. **Macro F1 vs Accuracy**: Social media datasets often exhibit significant class imbalances. Tracking Macro F1 ensures high sensitivity across both majority and minority classes.
4. **Learning Rate Sensitivity**: Fine-tuning pre-trained transformer backbones requires small learning rates (e.g., $2 \\times 10^{-5}$ to $5 \\times 10^{-5}$) with linear warmup schedules to protect pre-trained weights from gradient instability.

---
**Lab 3 Complete! You are ready to export this notebook to Kaggle or continue exploring advanced prompt tuning and quantization.**""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = "lab3/lab3.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Successfully generated {output_path} with {len(cells)} cells.")
