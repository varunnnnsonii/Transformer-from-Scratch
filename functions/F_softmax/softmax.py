import torch
from torch import nn

class MySoftmax(nn.Module):
    def __init__(self, dim=-1):
        """
        Custom Softmax activation function.

        Args:
            dim (int): The dimension along which softmax will be computed.
        """
        super(MySoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Subtract max for numerical stability
        x_max = x.max(dim=self.dim, keepdim=True).values
        x_exp = torch.exp(x - x_max)
        return x_exp / x_exp.sum(dim=self.dim, keepdim=True)
