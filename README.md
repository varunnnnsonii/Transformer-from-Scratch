


# Transformer‑from‑Scratch

A **from‑scratch implementation of an autoregressive Transformer language model** in PyTorch — no reliance on high‑level libraries like Hugging Face Transformers.  
Designed as a **learning project** to understand how a Transformer language model (similar to GPT) works internally, including **multi‑head self‑attention, positional encodings, training loop, and text generation**.

---

## 🚀 Project Overview

This repository contains a complete implementation of a Transformer‑style **autoregressive language model** built manually in PyTorch, inspired by foundational concepts from the original Transformer architecture (“Attention Is All You Need”) and simplified implementations (e.g., nanoGPT examples). 

The model is trained on a plain‑text dataset (e.g., a sales textbook) and learns to predict the next token given a context window (`context_length`). Once trained, it can generate coherent text continuations autoregressively.

---

## 📦 Repository Structure

```

Transformer-from-Scratch/
├── model.py                # Main file of the project
├── README.md               # Project overview
├── THEORY.md               # Mathematical + conceptual theory
├── .gitignore
│
├── data/
│   ├── input.txt
│   └── sales_textbook.txt
│
├── functions/              # Custom neural-network primitives
│   ├── F_softmax/
│   │   └── softmax.py
│   ├── nn_Linear/
│   │   └── linear.py
│   ├── nn_Embedding/
│   │   └── embedding.py
│   ├── nn_LayerNorm/
│   │   └── layernorm.py
│   ├── nn_Dropout/
│   │   └── dropout.py
│   └── nn_ReLU/
│       └── relu.py
│
└── transformer/
    └── model.py            # Transformer architecture


````

> *Note:* Some scripts may be combined in your current setup (e.g., training and model classes in a single file). You can split them into separate modules for clarity.

---

## 🧠 Key Features

✔️ **From‑scratch implementation** of core Transformer components  
✔️ **Multi‑head self‑attention** with causal masking  
✔️ **Feed‑forward neural networks** with residual connections  
✔️ **Sinusoidal positional encoding**  
✔️ **Autoregressive text generation**  
✔️ Training and validation on custom datasets  

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/varunnnnsonii/Transformer‑from‑Scratch.git
cd Transformer‑from‑Scratch
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical `requirements.txt`:

```
torch
tiktoken
requests
```

---

## 📘 Usage

### 📥 Prepare Dataset

The project expects a text corpus in:

```
data/sales_textbook.txt
```

If not present, the training script will automatically download it from HuggingFace.

---

### 📊 Train Model

To train the model, run:

```bash
python model.py
```

This script will:

✔ Load and tokenize the text
✔ Split into training/validation sets
✔ Train the Transformer model with AdamW
✔ Print train/validation loss periodically
✔ Save model checkpoint (`model‑ckpt.pt`)


---

## 🧩 How It Works — Concept Breakdown

### 🔹 1. Tokenization

Uses the TikToken tokenizer (`cl100k_base`) to convert raw text into token IDs.
These tokens are then converted to PyTorch tensors for batching and context windows.

---

### 🔹 2. Positional Encoding

Implements **sinusoidal positional encodings** to provide the model with token order information — a requirement since Transformers do not inherently model order. ([Wikipedia][1])

---

### 🔹 3. Attention & Multi‑Head Attention

Attention computes similarity between queries & keys, then weights values accordingly.
Multi‑head attention runs several attention “heads” in parallel to capture diverse patterns.

---

### 🔹 4. Transformer Blocks

Each block has:

* LayerNorm
* Multi‑head self‑attention with causal masking
* Feed‑forward network
* Skip residual connections

This structure enables efficient learning and stable gradients.

---

### 🔹 5. Autoregressive Generation

During inference:

1. Take prompt tokens
2. Crop to model’s context length
3. Compute logits
4. Sample next token from softmax distribution
5. Append and repeat

---

## 📈 Training Details

| Hyperparameter   | Value |
| ---------------- | ----- |
| Batch Size       | 4     |
| Context Length   | 16    |
| Model Dimension  | 64    |
| Number of Blocks | 8     |
| Attention Heads  | 4     |
| Dropout          | 0.1   |
| Learning Rate    | 1e‑3  |
| Max Iterations   | 5000  |
| Eval Interval    | 50    |

---

## 📌 Tips & Best Practices

✅ **Save checkpoints frequently** to avoid losing training progress
✅ **Use GPU (CUDA)** if available for faster training
✅ Batch size and context length can be increased for richer context learning

---

## 🧪 Results & Examples

After training, an example generation might look like:

```
The salesperson to identify the other cost savings interaction …
```

(Your output will vary with model performance and training length.)

---

## 📜 License

Licensed under MIT License.
Feel free to use and improve this implementation.

---

## 🙌 Acknowledgements

Inspired by many educational transformer‑from‑scratch projects and tutorials that aim to demystify transformer internals. 

---

## 🤝 Contributing

Contributions are welcome!
Feel free to:

* Add CLI flags for hyperparameters
* Split scripts into modules
* Add logging & visualization of training
* Introduce more advanced sampling (top‑k / top‑p)

---
