# Lab 2: Word Embeddings & Vector Spaces (Word2Vec & GloVe)

An in-depth, hands-on laboratory exploring distributed word representations, shifting from sparse lexical counts to dense semantic vector spaces using **Word2Vec** (Mikolov et al., 2013), **GloVe** (Pennington et al., 2014), and **Count-based Matrix Factorization** with `scikit-learn`, `gensim`, and `nltk`.

---

## 📂 Laboratory Structure

```
lab2/
├── lab2.ipynb          # End-to-end, executed laboratory notebook
└── README.md           # Theoretical foundations, derivations & guide
```

> **Note**: This lab runs within the unified repository-wide virtual environment (`nlp_llm/.venv`). Activate it via `.\.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/macOS).

---

## 📚 Laboratory Syllabus & Modules

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED WORD EMBEDDINGS PIPELINE                 │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
    ┌────────────────────────────────┴────────────────────────────────┐
    │ 1. Corpus Preprocessing & Vocabulary Structuring (NLTK)         │
    │    • Sentence segmentation & Word tokenization                  │
    │    • Punctuation normalization & Stopword filtering             │
    │    • Domain-structured corpus design (Royalty, Fruits, Gender)  │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
    ┌────────────────────────────────┴────────────────────────────────┐
    │ 2. Word2Vec Architectures: CBOW vs Skip-Gram (Gensim)           │
    │    • Continuous Bag-of-Words (Context -> Center)                │
    │    • Skip-Gram with Negative Sampling (Center -> Context)       │
    │    • Negative sampling distribution P_n(w) ~ f(w)^(3/4)         │
    │    • Frequent word subsampling P(drop) = 1 - sqrt(t / f(w))     │
    │    • Linear vector analogies: king - man + woman = queen        │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
    ┌────────────────────────────────┴────────────────────────────────┐
    │ 3. GloVe from First Principles: Global Co-occurrence Factorization
    │    • Distance-weighted co-occurrence matrix X_ij                │
    │    • Soft-capping weighting function f(X_ij) = (X_ij / x_max)^a │
    │    • Weighted Least Squares Objective & Gradient Descent        │
    │    • Dual representations: v_w = w_w + w_tilde_w                │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
    ┌────────────────────────────────┴────────────────────────────────┐
    │ 4. Matrix Factorization Baseline via Scikit-Learn (PPMI + SVD)   │
    │    • Positive Pointwise Mutual Information (PPMI) construction  │
    │    • TruncatedSVD for low-rank dense vector extraction          │
    │    • Connecting GloVe to classical Latent Semantic Analysis     │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
    ┌────────────────────────────────┴────────────────────────────────┐
    │ 5. Comparative Evaluation & 2D Embedding Visualization (sklearn)│
    │    • Pairwise cosine similarity benchmarking across models      │
    │    • Dimensionality reduction with PCA (Principal Components)   │
    │    • Geometric cluster plots & vector analogy displacement visualization
    └─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Foundations

### 1. Word2Vec: Continuous Bag-of-Words (CBOW) & Skip-Gram

#### Continuous Bag-of-Words (CBOW)
Predicts the center word $w_t$ given a window of surrounding context words $w_{t-c}, \dots, w_{t+c}$:

$$\bar{v} = \frac{1}{2c} \sum_{-c \le j \le c, j \neq 0} v_{w_{t+j}}$$

The probability of the target word is computed via softmax over the vocabulary $V$:

$$P(w_t \mid \text{context}) = \frac{\exp({v'_{w_t}}^\top \bar{v})}{\sum_{w \in V} \exp({v'_w}^\top \bar{v})}$$

#### Skip-Gram with Negative Sampling (SGNS)
Predicts each context word $w_c$ given the center word $w_t$. Because a full softmax is computationally prohibitive ($O(|V|)$ per step), **Negative Sampling** reformulates the problem as multiple binary classifications:

$$\mathcal{L}_{\text{SGNS}} = \log \sigma({v'_{w_c}}^\top v_{w_t}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-{v'_{w_i}}^\top v_{w_t}) \right]$$

where:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.
- $k$ is the number of negative samples per positive pair (typically 5–20 for small corpora).
- $P_n(w) \propto f(w)^{3/4}$ is the smoothed unigram noise distribution, dampening high-frequency words and boosting rare terms.

---

### 2. GloVe: Global Vectors for Word Representation

GloVe bridges global count-based matrix factorization (like SVD) and local window-based predictive models (like Word2Vec).

#### Ratio of Co-occurrence Probabilities
Let $P_{ik} = P(k \mid i) = \frac{X_{ik}}{X_i}$ be the probability that word $k$ appears in the context of word $i$. Pennington et al. demonstrated that semantic meaning is captured by the **ratio** of co-occurrence probabilities:

$$\frac{P_{ik}}{P_{jk}} = \frac{P(\text{solid} \mid \text{ice})}{P(\text{solid} \mid \text{steam})} \gg 1, \quad \frac{P(\text{gas} \mid \text{ice})}{P(\text{gas} \mid \text{steam})} \ll 1, \quad \frac{P(\text{water} \mid \text{ice})}{P(\text{water} \mid \text{steam})} \approx 1$$

To encode this ratio linearly in vector space:

$$w_i^\top \tilde{w}_k + b_i + \tilde{b}_k = \log X_{ik}$$

#### Weighted Least-Squares Objective
GloVe solves this system via a weighted least squares regression:

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

where the weighting function $f(x)$ caps extremely frequent words to prevent them from dominating:

$$f(x) = \begin{cases} \left( \frac{x}{x_{\max}} \right)^\alpha & \text{if } x < x_{\max} \\ 1.0 & \text{otherwise} \end{cases}$$

Standard parameters: $x_{\max} = 100$, $\alpha = 0.75$.

Final word vectors are obtained by summing the center and context vectors:

$$v_w = w_w + \tilde{w}_w$$

---

### 3. Positive Pointwise Mutual Information (PPMI) & TruncatedSVD

As a classical count-based baseline, we compute the PMI matrix:

$$\text{PMI}(i, j) = \log_2 \frac{P(i, j)}{P(i) P(j)} = \log_2 \frac{X_{ij} \cdot \sum_{a,b} X_{ab}}{\sum_b X_{ib} \sum_a X_{aj}}$$

To eliminate negative infinities for unobserved pairs:

$$\text{PPMI}(i, j) = \max(\text{PMI}(i, j), 0)$$

Applying `sklearn.decomposition.TruncatedSVD` decomposes $\text{PPMI} \approx U \Sigma V^\top$, yielding low-rank embeddings $W = U \Sigma^{1/2}$.

---

## 🚀 Quickstart & Execution

```bash
# From repository root:
.\.venv\Scripts\activate       # Windows
# or source .venv/bin/activate # Linux/macOS

# Launch notebook
jupyter notebook lab2/lab2.ipynb
```
Select the **`Python (NLP-LLM)`** kernel in the notebook interface.
