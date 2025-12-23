

# Transformer Language Model – Theory and Implementation

## 1. Introduction

The Transformer architecture, introduced by Vaswani et al. in **“Attention is All You Need” (2017)**, revolutionized sequence modeling in NLP. Unlike RNNs or LSTMs, Transformers do **not require sequential computation** and can process sequences in parallel using **self-attention**, enabling **long-range dependency modeling**, faster training, and scalability.

This project implements a Transformer **from scratch** in PyTorch, with fully **custom layers** including `Linear`, `Dropout`, `ReLU`, `LayerNorm`, `Softmax`, and optionally `Embedding`. It is designed for **language modeling**, predicting the next token in a sequence and generating coherent text.

---

## 2. Transformer Architecture Overview

A Transformer consists of **stacked decoder blocks** (for language modeling). The main components:

```
Input Tokens --> Embedding --> Positional Encoding
       --> [ Transformer Block x N ]
       --> LayerNorm --> Linear --> Softmax --> Predicted Tokens
```

Each **Transformer Block** consists of:

```
      ┌──────────────────────────────┐
x --->│ LayerNorm ──> MultiHeadAttention ──┐
      │                                  │
      │                                  ▼
      │                     Residual Connection (+)
      │                                  │
      │ LayerNorm ──> FeedForward ───────┘
      │
      │
      ▼
  Output to next block
```

---

## 3. Token Embeddings

* **Purpose:** Convert discrete tokens into continuous dense vectors.
* **Shape:** `(vocab_size, d_model)`
* **Project Implementation:** `nn.Embedding` or `MyEmbedding`.
* **Formula:**

$$
E_t = \text{Embedding}(token_t) \in \mathbb{R}^{d_{model}}
$$

---

## 4. Positional Encoding

Since Transformers **do not know token order**, we encode positions:

$$
PE_{(pos, 2i)} = \sin \left(\frac{pos}{10000^{2i/d_{model}}} \right), \quad
PE_{(pos, 2i+1)} = \cos \left(\frac{pos}{10000^{2i/d_{model}}} \right)
$$

* **Adds unique position info** to embeddings.
* **Project Implementation:** Precompute `position_encoding_lookup_table` tensor.

---

## 5. Scaled Dot-Product Attention

* Computes weighted sum of **values (V)** using **queries (Q) and keys (K)**:

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

* **Masking:** Prevents attending to future tokens (causal) and optionally ignores padding tokens.
* **Dropout:** Applied on attention weights.

---

### 5.1 Multi-Head Attention

* Multiple attention heads capture **different relationships**:

```
        Q, K, V
          │
    ┌─────┴─────┐
    │ head 1     │
    │ head 2     │
    │ ...        │
    └─────┬─────┘
          │
      Concat --> Linear --> Output
```

* **Project:** `num_heads = 4`, `head_size = d_model / num_heads`.

---

## 6. Feed-Forward Network (FFN)

* Position-wise FFN in each block:

$$
FFN(x) = \text{Dropout}(\text{ReLU}(x W_1 + b_1) W_2 + b_2)
$$
* Expands `d_model` → `4*d_model` → back to `d_model`.
* Introduces **non-linearity** and increases **capacity**.

---

## 7. Layer Normalization

* Stabilizes training by normalizing across features:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta
$$

* Applied **before attention and FFN** (Pre-LN).

---

## 8. Residual Connections

* Help **gradient flow**:

$$
x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))
$$

* Essential for deep Transformer blocks.

---

## 9. Dropout

* Randomly drops neurons with probability `p` to prevent overfitting.
* Remaining values are **scaled**:

$$
x_{\text{drop}} = \frac{x \odot mask}{1-p}
$$

---

## 10. Output Layer

* Linear projection `d_model` → `vocab_size`.
* Produces logits for each token.
* **Cross-Entropy Loss** used for training:

$$
\text{Loss} = - \sum_i y_i \log(\text{Softmax}(logits_i))
$$

---

## 11. Text Generation

* Autoregressive: predicts one token at a time.
* Steps:

  1. Crop input to `context_length`.
  2. Forward pass → logits.
  3. Softmax → probabilities.
  4. Sample next token.
  5. Append → repeat for `max_new_tokens`.

---

## 12. Custom Implementations

| Layer/Function | Custom Class  |
| -------------- | ------------- |
| Linear         | `MyLinear`    |
| Dropout        | `MyDropout`   |
| ReLU           | `MyReLU`      |
| LayerNorm      | `MyLayerNorm` |
| Softmax        | `MySoftmax`   |
| Embedding      | `MyEmbedding` |

* Ensures **full control and transparency**.

---

## 13. Hyperparameters – Comprehensive Table

| Hyperparameter | Meaning / Role          | Effect of Increasing                                   | Effect of Decreasing        | Recommended Range                                     |
| -------------- | ----------------------- | ------------------------------------------------------ | --------------------------- | ----------------------------------------------------- |
| batch_size     | Sequences per batch     | Stable gradients, faster convergence, more memory      | Noisy gradients, slower     | Small: 4–16<br>Medium: 32–128<br>Large: 256+          |
| context_length | Tokens per sequence     | Capture longer dependencies, more memory & computation | Short-term learning, faster | Small: 16–64<br>Medium: 128–512<br>Large: 1024–2048   |
| d_model        | Embedding & hidden size | More capacity, slower training                         | Less expressive, faster     | Small: 64–128<br>Medium: 256–512<br>Large: 768–2048   |
| num_blocks     | Transformer layers      | Deeper, more expressive                                | Shallower, faster           | Small: 2–6<br>Medium: 6–12<br>Large: 12–96            |
| num_heads      | Attention heads         | Capture more relationships                             | Fewer heads, less capacity  | Small: 2–4<br>Medium: 8<br>Large: 12–16               |
| learning_rate  | Optimizer step size     | Faster learning, may diverge                           | Slower, may get stuck       | 1e-4 – 1e-3 (AdamW)                                   |
| dropout        | Regularization          | Prevents overfitting, slower                           | Risk of overfitting, faster | 0.1–0.3                                               |
| max_iters      | Training iterations     | Better convergence                                     | Underfitting                | Small: 500–5000<br>Medium: 10k–50k<br>Large: millions |
| eval_interval  | Eval frequency          | More frequent validation                               | Less frequent validation    | 50–500 steps                                          |
| eval_iters     | Eval averaging batches  | Stable validation                                      | Less stable                 | 10–50                                                 |

---

## 14. Hyperparameter Step-by-Step Explanation

1. **`batch_size`** – Larger → stable, faster convergence; Smaller → less memory, noisy gradients.
2. **`context_length`** – Larger → long-range dependencies, more memory; Smaller → short-term, faster.
3. **`d_model`** – Larger → more expressive, slower; Smaller → less expressive, faster.
4. **`num_blocks`** – More → deeper understanding, slower; Fewer → shallow, faster.
5. **`num_heads`** – More → captures multiple relationships; Fewer → less capacity.
6. **`learning_rate`** – Too high → unstable; Too low → slow training.
7. **`dropout`** – More → regularization; Less → may overfit.
8. **`max_iters`** – More → better convergence; Less → underfitting.
9. **`eval_interval`** – More frequent → slower training, early detection of overfitting; Less frequent → faster training.
10. **`eval_iters`** – Larger → stable evaluation; Smaller → noisier evaluation.

---

## 15. Flowcharts and Diagrams

### 15.1 Transformer Block Flow

```
          Input x
             │
         LayerNorm
             │
   ┌─────────┴─────────┐
   │   MultiHeadAttention │
   └─────────┬─────────┘
             │
       Residual Add (+)
             │
         LayerNorm
             │
          FeedForward
             │
       Residual Add (+)
             │
          Output
```

### 15.2 Multi-Head Attention

```
         Input x
            │
     ┌──────┴──────┐
     │ Head 1       │
     │ Head 2       │
     │ ...          │
     └──────┬──────┘
            │
        Concatenate
            │
         Linear
            │
          Output
```

### 15.3 Autoregressive Generation

```
Prompt --> Crop to context_length --> Forward Pass --> Softmax --> Sample next token
        --> Append to prompt --> Repeat for max_new_tokens
```

### 15.4 Causal Masking in Attention

```
Query positions:   0 1 2 3
Key positions:
0 -> ✔
1 -> ✔ ✔
2 -> ✔ ✔ ✔
3 -> ✔ ✔ ✔ ✔
```

### 15.5 Residual + LayerNorm Overview

```
      x_in
        │
   LayerNorm
        │
   SubLayer (MHA / FFN)
        │
   +------+
   | Add  |
   +------+
        │
      x_out
```

---

## 16. Initialization of Parameters

* Proper initialization avoids vanishing/exploding gradients.
* **Common methods:**

  * Xavier/Glorot for Linear layers.
  * Kaiming/He for ReLU-based layers.
* Ensure your custom `MyLinear` and `MyEmbedding` layers are initialized accordingly.

---

## 17. Optimizer and Learning Rate Scheduling

* **AdamW:** Combines Adam optimizer with decoupled weight decay.
* **Optional Scheduler:** Linear warm-up + cosine decay for stable training.
* Update rule:

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$


---

## 18. Gradient Clipping

* Prevents exploding gradients in deep Transformers.
* Typical usage:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 19. Loss Smoothing (Optional)

* Label smoothing prevents overconfident predictions:

$$
\text{LS Loss} = (1 - \epsilon) \cdot \text{CE Loss} + \epsilon / N
$$

Where `N = vocab_size`, `ε ≈ 0.1`.

---

## 20. Evaluation Metrics

* **Perplexity:** Standard metric for language models.
$$
Perplexity = exp(Loss)
$$

Lower perplexity → better predictive performance.

---

## 21. Training Tips

* **Mixed Precision (FP16):** Reduces memory usage and speeds up training.
* **Gradient Accumulation:** Simulates large batch sizes.
* **Early Stopping:** Stop training if validation loss doesn’t improve.

---

## 22. Scaling Considerations

* **Depth vs Width:** Deeper → more abstraction; Wider → more features.
* **Context Length:** Longer context captures more dependencies but increases memory (O(T²) in attention).
* **Batch Size:** Large batch + gradient accumulation + learning rate scaling improves stability.

---

## 23. References

* Vaswani et al., *Attention is All You Need*, 2017: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

---

## 24. Conclusion

This project demonstrates a **fully modular Transformer language model** from scratch:

* **Custom layers** for learning the inner workings.
* **Attention-based architecture** capable of long-range dependency modeling.
* **Step-by-step hyperparameter understanding** for better experimentation.
* **Autoregressive generation** to produce text sequences.
* **Complete training best practices** for initialization, optimization, masking, evaluation, and scaling.

With proper hyperparameter tuning, it can scale from **toy datasets** to **medium NLP tasks**, while providing **full transparency** of the underlying computations.


---