import torch

x = torch.arange(10)

y = torch.randn(10)

xy = torch.cat((x, y), dim=-1)
print(xy)