import torch
from torch.nn.functional import gelu

x = torch.tensor([1.0, 2.0, 3.0])
print(gelu(x))  # Test the GELU activation function