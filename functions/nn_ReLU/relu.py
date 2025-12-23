import torch
from torch import nn

class MyReLU(nn.Module):
    def __init__(self, inplace=False):
        """
        Custom ReLU activation function.

        Args:
            inplace (bool): If True, modifies the input directly (like nn.ReLU(inplace=True))
        """
        super(MyReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.clamp_(min=0)
        else:
            return x.clamp(min=0)
