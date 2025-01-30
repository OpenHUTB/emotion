import torch
from torch.nn.functional import gelu

x = torch.tensor([1.0, 2.0, 3.0])
print(gelu(x))  # 测试 GELU 函数
