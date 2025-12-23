import os
import requests
import math
import tiktoken
import torch
from torch import nn

# Instead of redefining them, import your custom implementations
from functions.nn_Linear.linear import MyLinear
from functions.nn_Dropout.dropout import MyDropout
from functions.nn_ReLU.relu import MyReLU
from functions.nn_LayerNorm.layernorm import MyLayerNorm
from functions.F_softmax.softmax import MySoftmax 
from functions.nn_Embedding.embedding import MyEmbedding 

# ==========================
# Hyperparameters
# ==========================
batch_size = 4
context_length = 16
d_model = 64
num_blocks = 8
num_heads = 4
learning_rate = 1e-3
dropout = 0.1
max_iters = 300
eval_interval = 50
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# ==========================
# Load data
# ==========================
if not os.path.exists('data/sales_textbook.txt'):
    os.makedirs('data', exist_ok=True)
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text) + 1
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]

# ==========================
# Custom Embedding
# ==========================
# class MyEmbedding(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
#     def forward(self, x):
#         return self.weight[x]

# ==========================
# Feed Forward
# ==========================
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            MyLinear(d_model, d_model * 4),
            MyReLU(),
            MyLinear(d_model * 4, d_model),
            MyDropout(dropout)
        )
    def forward(self, x):
        return self.ffn(x)

# ==========================
# Attention
# ==========================
class Attention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key_layer = MyLinear(d_model, head_size, bias=False)
        self.query_layer = MyLinear(d_model, head_size, bias=False)
        self.value_layer = MyLinear(d_model, head_size, bias=False)
        self.tril = torch.tril(torch.ones(context_length, context_length)).to(device)
        self.dropout_layer = MyDropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)
        weights = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = MySoftmax(dim=-1)(weights)
        weights = self.dropout_layer(weights)
        return weights @ v

# ==========================
# MultiHead Attention
# ==========================
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Attention(head_size) for _ in range(num_heads)])
        self.projection_layer = MyLinear(d_model, d_model)
        self.dropout_layer = MyDropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out

# ==========================
# Transformer Block
# ==========================
class TransformerBlock(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.ln1 = MyLayerNorm(d_model)
        self.ln2 = MyLayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model // num_heads)
        self.ffn = FeedForward()
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ==========================
# Transformer Language Model
# ==========================
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = MyEmbedding(max_token_value + 1, d_model)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(num_heads) for _ in range(num_blocks)])
        self.ln_final = MyLayerNorm(d_model)
        self.out_layer = MyLinear(d_model, max_token_value)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Positional Encoding
        pos_enc = torch.zeros(context_length, d_model, device=device)
        pos = torch.arange(0, context_length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        x = self.token_embedding(idx) + pos_enc[:T]
        x = self.transformer_blocks(x)
        x = self.ln_final(x)
        logits = self.out_layer(x)

        if targets is not None:
            # Custom cross entropy loss
            log_probs = MySoftmax(dim=-1)(logits).log()
            loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).mean()
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -context_length:]
            logits, _ = self(idx_crop)
            logits_last = logits[:, -1, :]
            probs = MySoftmax(dim=-1)(logits_last)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================
# Initialize
# ==========================
model = TransformerLanguageModel().to(device)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(0, len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in idxs]).to(device)
    y = torch.stack([data[i+1:i+context_length+1] for i in idxs]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tracked_losses = []

for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters-1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print(f"Step {step}, Train Loss: {losses['train']:.3f}, Val Loss: {losses['valid']:.3f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save model
torch.save(model.state_dict(), 'shakespere-model-ckpt.pt')

# Generate
model.eval()
start = "The salesperson"
start_ids = encoding.encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
y = model.generate(x, max_new_tokens=100)
print('---------------')
generated_text = encoding.decode(y[0].tolist())
with open("generated.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)

print(encoding.decode(y[0].tolist()).encode('utf-8', errors='ignore').decode('utf-8'))

print('---------------')
