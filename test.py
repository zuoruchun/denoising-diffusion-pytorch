import torch
import torch.nn as nn
from functools import partial

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.g / (x.norm(2, dim=1, keepdim=True) * self.scale + 1e-8)

class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        print("x.shape:",x.shape)
        x = self.proj(x)
        print("after proj:",x.shape)
        x = self.norm(x)
        print("after norm:",x.shape)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        print("after act:",x.shape)
        x = self.dropout(x)
        print("after dropout:",x.shape)
        return x

if __name__ == "__main__":
    # 创建测试数据
    batch_size = 1
    channels = 3
    height = 256
    width = 256
    x = torch.randn(batch_size, channels, height, width)
    print("Input shape:", x.shape)

    # 测试Block
    block = Block(dim=3, dim_out=64)
    output = block(x)
    print("Output shape:", output.shape)

    # # 测试带scale_shift的情况
    # scale = torch.randn_like(output)
    # shift = torch.randn_like(output)
    # output_with_scale_shift = block(x, (scale, shift))
    # print("Output shape with scale_shift:", output_with_scale_shift.shape)