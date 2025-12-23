import torch
from torch import nn

class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        Custom Dropout layer.
        
        Args:
            p (float): Probability of dropping a unit. 0 <= p < 1
        """
        super(MyDropout, self).__init__()
        if p < 0 or p >= 1:
            raise ValueError("Dropout probability must be in [0, 1).")
        self.p = p

    def forward(self, x):
        if self.training:
            # Generate a mask with 0s and 1s
            mask = (torch.rand_like(x) > self.p).float()
            # Scale the remaining values
            return mask * x / (1.0 - self.p)
        else:
            # During evaluation, do nothing
            return x
