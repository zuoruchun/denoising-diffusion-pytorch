import torch
import math
from torch.nn import Module

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        print(f"Step 1: emb (log scale factor) = {emb}")
        
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        print(f"Step 2: emb (frequency factors) = {emb}")
        
        # 修改这一步，确保 x 和 emb 的大小匹配
        emb = x[:, None] * emb[None, :]  # 这里x的形状是 (batch_size, seq_length)，emb是(half_dim,)
        print(f"Step 3: emb (scaled positions) = {emb}")
        
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        print(f"Step 4: emb (sin + cos) = {emb}")
        
        return emb

# 测试代码
batch_size = 2
seq_length = 5
dim = 6  # 假设嵌入维度是6
x = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)

# 创建 SinusoidalPosEmb 实例并进行前向传播
sinusoidal_pos_emb = SinusoidalPosEmb(dim=dim)
output = sinusoidal_pos_emb(x)

print(f"Final Output (Position Embeddings): {output}")
