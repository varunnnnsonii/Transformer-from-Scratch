import torch
from torch import nn
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
    def forward(self, x):
        return self.weight[x]
