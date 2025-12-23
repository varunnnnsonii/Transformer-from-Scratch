import torch
import math
from torch import nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        A custom implementation of nn.Linear.

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            bias (bool): Whether to include a bias term
        """
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weight and bias similar to PyTorch nn.Linear
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Forward pass: y = xW^T + b
        """
        return input.matmul(self.weight.t()) + (self.bias if self.bias is not None else 0)

# if __name__ == "__main__":
#     batch_size = 4
#     in_features = 3
#     out_features = 2

#     layer = MyLinear(in_features, out_features)
#     x = torch.randn(batch_size, in_features)
#     y = layer(x)
#     print("Input:", x)
#     print("Output:", y)
