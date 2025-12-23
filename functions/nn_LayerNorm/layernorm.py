import torch
from torch import nn

class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Custom Layer Normalization.

        Args:
            normalized_shape (int or tuple): The shape of the normalized dimensions
            eps (float): Small number to prevent division by zero
            elementwise_affine (bool): If True, learnable scale (gamma) and shift (beta) are used
        """
        super(MyLayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(*normalized_shape))
            self.beta = nn.Parameter(torch.zeros(*normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):
        # Compute mean and variance along the normalized dimensions
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable scale and shift if needed
        if self.elementwise_affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm
