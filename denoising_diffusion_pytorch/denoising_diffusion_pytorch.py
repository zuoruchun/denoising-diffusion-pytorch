import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])        # 定义一个元组，包含两个元素，分别是pred_noise和pred_x_start，可以通过ModelPrediction.pred_noise和ModelPrediction.pred_x_start来访问

# helpers functions

def exists(x):  # 判断x是否存在，存在返回True，不存在返回False
    return x is not None

def default(val, d):    # 如果val存在，则返回val，否则返回d
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):  # 将t转换为长度为length的元组
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):  # 判断numer是否能被denom整除
    return (numer % denom) == 0

def identity(t, *args, **kwargs):  # 返回t
    return t

def cycle(dl):  # 无限循环数据加载器
    while True:
        for data in dl:
            yield data      # 生成器， 相当于return data

def has_int_squareroot(num):        # 判断num是否为整数的平方根
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):    # 例如num=10, divisor=3，输出[3, 3, 3, 1]
    groups = num // divisor    # 取整
    remainder = num % divisor   # 取余
    arr = [divisor] * groups    
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):       # 将image转换为img_type类型
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):    # [0, 1] 归一化到 [-1, 1]
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):    # [-1, 1] 归一化到 [0, 1]
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(               # nn.Sequential是一个有序的容器，用于将多个模块按顺序组合在一起,在这里会先执行上采样，然后进行卷积操作
        nn.Upsample(scale_factor = 2, mode = 'nearest'),        # 调用nn.Upsample，将输入的图像放大2倍，使用最近邻插值法
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)   # 调用nn.Conv2d，将输入的图像进行卷积操作，输出通道数为default(dim_out, dim)，卷积核大小为3，填充为1
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),    # b c (h p1) (w p2)  分别指的是batch_size, channels, height, width；这个重排将c->(c p1 p2)，(h p1)->h，(w p2)->w, 由于p1=2, p2=2, 所以通道数增加4倍，宽高减半
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)    # 调用nn.Conv2d，将输入的图像进行卷积操作，输出通道数为default(dim_out, dim)，卷积核大小为1
    )

class RMSNorm(Module):       # 实现一种基于均方根（Root Mean Square）的归一化方法。
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5     # 缩放因子  
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))    # 可训练的参数，用于调整归一化后的结果

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale    # \frac{x_i}{\sqrt{\sum(x_i^2)} + epsilon}

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):         # 正弦位置编码（Sinusoidal Positional Embedding） 模块，用于将离散的时间步（timestep）映射为连续的特征表示。
    def __init__(self, dim, theta = 10000):     # dim: 输入张量的维度，theta: 控制波长分布的常数
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)     
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)       # 生成一组频率值，用于控制正弦函数的波长。torch.arange和python的range类似，都是生成一个从0到half_dim-1的序列
        emb = x[:, None] * emb[None, :]         # x[:, None]指的是在x插入一个新的维度，这里就是在列插入，于是原本的x的形状是(10,)，现在变成了(10, 1)，而emb[None, :]指的是在emb插入一个新的维度，这里就是在行插入，于是原本的emb的形状是(5,)，现在变成了(1, 5)，于是x[:, None] * emb[None, :]的形状是(10, 5)。
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 将正弦和余弦部分拼接在一起，形成最终的正弦位置编码。
        return emb

# 如果使用了随机或者自学习正弦条件
"""
为什么需要拼接：
1. 原始时间步的局限性：
    单独使用原始时间步 x 作为条件信息，可能无法充分表达时间步的复杂关系，尤其是在扩散模型中，时间步的变化对噪声分布的影响是非线性的。
2. 正弦/余弦编码的补充： 
    正弦/余弦编码通过多尺度的频率特征，能够捕捉时间步的周期性变化和非线性关系，但单独使用可能会丢失原始时间步的直接信息。
3. 拼接的优势：
    将两者拼接在一起，既保留了原始时间步的直接信息，又引入了多尺度的频率特征，形成一种 混合编码，能够更全面地表达时间步的信息。
"""
class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)     # 断言dim是否能被2整除，不能整除会报错
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)    # 生成权重参数：先生成half_dim个0-1之间的随机数，默认使用梯度，也就是参数是可训练的，否则参数是不可训练的，即参数是不变的。
        
    def forward(self, x):
        x = rearrange(x, 'b -> b 1')    # 将x的形状从(b,)变为(b, 1)
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi    # 将x与权重相乘，得到频率值
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)    # 将正弦和余弦部分拼接在一起，形成最终的正弦位置编码。
        fouriered = torch.cat((x, fouriered), dim = -1)    # 将x与fouriered拼接在一起，形成最终的正弦位置编码。
        return fouriered
    

# building block modules
# 残差块
class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):     # 初始化模型参数
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)     # 卷积层，输入维度为dim，输出维度为dim_out，卷积核大小为3，填充为1
        self.norm = RMSNorm(dim_out)    # 归一化层，输入维度为dim_out
        self.act = nn.SiLU()    # 激活函数，使用SiLU（Sigmoid Linear Unit）激活函数
        self.dropout = nn.Dropout(dropout)    # 随机失活层，dropout率为dropout

    def forward(self, x, scale_shift = None):       # 调用初始化函数，进行前向传播
        x = self.proj(x)        # 卷积操作
        x = self.norm(x)        # 归一化操作

        if exists(scale_shift):  # 如果存在缩放和偏移，则将缩放和偏移应用到x上
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)    # 使用SiLU激活函数
        return self.dropout(x)    # 使用dropout

# 残差块
class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None     # 如果存在时间嵌入，则使用时间嵌入，先执行SiLU激活函数，然后执行线性层，输出维度为dim_out * 2

        self.block1 = Block(dim, dim_out, dropout = dropout)    # 利用Block，初始化block1   
        self.block2 = Block(dim_out, dim_out)    # 利用Block，初始化block2
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()    # 如果输入维度不等于输出维度，则使用1x1卷积，否则使用恒等映射

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):       # mlp的存在: 取决于创建实例化ResnetBlock时是否传入time_emb_dim。
            time_emb = self.mlp(time_emb)           # 将时间嵌入通过mlp，得到缩放和偏移
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')    # 将时间嵌入的形状从(b, c)变为(b, c, 1, 1)
            scale_shift = time_emb.chunk(2, dim = 1)    # 将时间嵌入的形状从(b, c, 1, 1)变为(b, c//2, 1, 1)和(b, c//2, 1, 1)

        h = self.block1(x, scale_shift = scale_shift)    # 调用block1，输入x，scale_shift为scale_shift

        h = self.block2(h)       # 调用block2，输入h

        return h + self.res_conv(x)    # 返回h + res_conv(x)

# 线性注意力通过减少计算复杂度来加速注意力计算，同时保持对全局信息的捕捉能力。
## 在下面每一步的步骤后面注释一下当前变量的维度
class LinearAttention(Module):
    def __init__(
        self,
        dim,    # 输入维度
        heads = 4,    # 注意力头的数量，多头注意力机制允许模型同时关注不同的特征子空间。
        dim_head = 32,    # 注意力头的维度
        num_mem_kv = 4    # 记忆键值对的数量
    ):
        super().__init__()
        self.scale = dim_head ** -0.5    # 用于缩放注意力分数，防止点积过大导致梯度不稳定。
        self.heads = heads    # 注意力头的数量
        hidden_dim = dim_head * heads    # 计算多头注意力的总维度

        self.norm = RMSNorm(dim)    # 对输入特征进行 RMS 归一化，稳定训练过程。

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))    # 定义可学习的记忆键值对。这些键值对与输入无关，用于增强注意力的表达能力。mem_kv:(2, heads, dim_head, num_mem_kv)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)    # 将输入特征投影为查询（Q）、键（K）和值（V）。输出通道数为 hidden_dim * 3，因为 Q、K、V 各占一部分。经过to_qkv之后张量的变化：(b, c, h, w)  --to_qkv--> (b, hidden_dim * 3, h, w)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),    # 将多头注意力结果投影回原始维度
            RMSNorm(dim)    # 对输出特征进行 RMS 归一化，稳定训练过程。
        )

    def forward(self, x):
        b, c, h, w = x.shape    # x:(b, c, h, w)

        x = self.norm(x)    # 归一化处理。x:(b, c, h, w)  --norm--> x:(b, c, h, w)

        qkv = self.to_qkv(x).chunk(3, dim = 1)    # x:(b, c, h, w)  --to_qkv--> qkv:(b, hidden_dim * 3, h, w) --chunk--> qkv:(b, hidden_dim, h, w) * 3
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)    # 对qkv进行rearrange操作。qkv:(b, hidden_dim, h, w) * 3  --rearrange--> q:(b, heads, dim_head, h*w) ， k:(b,heads,  dim_head, h*w) ， v:(b, heads, dim_head, h*w)
        """
        map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])-------> 计算[1+2, 3+4, 5+6, 7+8, 9+10]
        rearrange(t, 'h (c n) -> h c n', c = 2, n = 3)-------> 将t的形状从(h, c*n)变为(h,2,3)
        repeat(t, 'h c n -> b h c n', b = b)-------> 将t的形状从(h, c, n)变为(b, h, c, n);其中b是repeat的次数
        在下面的repeat中，mem_kv是(2, heads, dim_head, num_mem_kv)，这个repeat是对(heads, dim_head, num_mem_kv进行repeat)，因此会出现两个张量，一个是mk，一个是mv，mk和mv的形状都是(b, heads, dim_head, num_mem_kv)
        """
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)    # 对mem_kv的(heads, dim_head, num_mem_kv)进行repeat操作。mem_kv:(2, heads, dim_head, num_mem_kv)  --repeat--> mk:(b, heads, dim_head, num_mem_kv) ， mv:(b, heads, dim_head, num_mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))    # 将mk和k拼接成k，将mv和v拼接成v。mk:(b, heads, dim_head, num_mem_kv) + k:(b, heads, dim_head, h*w) = k:(b, heads, dim_head, h*w+num_mem_kv)

        q = q.softmax(dim = -2)    # 对查询进行 softmax 操作，计算注意力分数。q:(b, heads, dim_head, h*w)
        k = k.softmax(dim = -1)    # 对键进行 softmax 操作，计算注意力分数。k:(b, heads, dim_head, h*w+num_mem_kv)

        q = q * self.scale    # 缩放注意力分数，防止点积过大导致梯度不稳定。  q:(b, heads, dim_head, h*w)
        """
        基于爱因斯坦求和约定（Einstein summation convention）。它通过字符标签来灵活地进行张量的维度缩并、转置和乘法等操作。
        根据 einsum 的规则，会自动对所有出现在输入标签中但未出现在输出标签中的维度进行求和。

        """
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)    # 对两个张量的最后一个维度（n）进行点积，并对结果进行求和。维度变化：(b, heads, dim_head, h*w+num_mem_kv) * (b, heads, dim_head, w+num_mem_kv) = (b, heads, dim_head, dim_head)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)    # 对两个张量的第三个维度（d）进行点积，并对结果进行求和。维度变化：(b, heads, dim_head, dim_head) * (b, heads, dim_head, w) = (b, heads, dim_head, w)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)    # 维度变化：(b, heads, dim_head, w)  --rearrange--> (b, heads*dim_head, h, w)
        return self.to_out(out)    # 将输出投影回原始维度，并进行 RMS 归一化，稳定训练过程。  out:(b, heads*dim_head, h, w)  --to_out--> out:(b, c, h, w)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads  # 注意力头的数量
        hidden_dim = dim_head * heads    # 计算多头注意力的总维度

        self.norm = RMSNorm(dim)    # 对输入特征进行 RMS 归一化，稳定训练过程。
        self.attend = Attend(flash = flash)    # 初始化Attend类，用于注意力机制

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))    # 定义可学习的记忆键值对。这些键值对与输入无关，用于增强注意力的表达能力。
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)    # 将输入特征投影为查询（Q）、键（K）和值（V）。输出通道数为 hidden_dim * 3，因为 Q、K、V 各占一部分。
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)    # 将多头注意力结果投影回原始维度

    def forward(self, x):
        b, c, h, w = x.shape    # 获取输入张量的形状x:(b, c, h, w)

        x = self.norm(x)    # 对输入特征进行 RMS 归一化，稳定训练过程。x:(b, c, h, w)  --norm--> x:(b, c, h, w)

        qkv = self.to_qkv(x).chunk(3, dim = 1)      # x:(b, c, h, w)  --to_qkv--> qkv:(b, hidden_dim * 3, h, w) --chunk--> qkv:(b, hidden_dim, h, w) * 3
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)    # 对qkv进行rearrange操作。qkv:(b, hidden_dim, h, w) * 3  --rearrange--> q:(b, heads, h*w, dim_head) ， k:(b,heads,  h*w, dim_head) ， v:(b, heads, h*w, dim_head)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)    # 对mem_kv的(heads, num_mem_kv, dim_head)进行repeat操作。mem_kv:(2, heads, num_mem_kv, dim_head)  --repeat--> mk:(b, heads, num_mem_kv, dim_head) ， mv:(b, heads, num_mem_kv, dim_head)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))    # 将mk和k拼接成k，将mv和v拼接成v。mk:(b, heads, num_mem_kv, dim_head) + k:(b, heads, h*w, dim_head) = k:(b, heads, h*w+num_mem_kv, dim_head)

        out = self.attend(q, k, v)    # 调用Attend类，进行注意力机制。q:(b, heads, h*w, dim_head) ， k:(b, heads, h*w+num_mem_kv, dim_head) ， v:(b, heads, h*w+num_mem_kv, dim_head)  --attend--> out:(b, heads, h*w, dim_head)        （这个没看）

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)    # 将out的形状从(b, heads, h*w, dim_head)变为(b, heads*dim_head, h, w)
        return self.to_out(out)    # 将输出投影回原始维度，并进行 RMS 归一化，稳定训练过程。  out:(b, heads*dim_head, h, w)  --to_out--> out:(b, c, h, w)

# model

class Unet(Module):     # 定义Unet模型，继承自Module，Moudle来自torch.nn
    def __init__(
        self,
        dim,        # 模型维度
        init_dim = None,  # 初始维度  
        out_dim = None,    # 输出维度   
        dim_mults = (1, 2, 4, 8),    # 维度倍数
        channels = 3,    # 通道数
        self_condition = False,    # 是否使用自条件，即使用前一步的预测作为额外输入
        learned_variance = False,    # 是否学习噪声方差
        learned_sinusoidal_cond = False,    # 是否使用学习到的正弦条件
        random_fourier_features = False,    # 是否使用随机傅里叶特征    
        learned_sinusoidal_dim = 16,    # 学习到的正弦条件的维度
        sinusoidal_pos_emb_theta = 10000,    # 正弦位置嵌入的theta
        dropout = 0.,    # dropout率，防止过拟合
        attn_dim_head = 32,    # 注意力头的维度
        attn_heads = 4,    # 注意力头的数量
        full_attn = None,    # 默认情况下，仅对最内层使用全注意力
        flash_attn = False    # 是否使用flash注意力
    ):
        super().__init__()      # 调用父类Module的初始化方法

        # determine dimensions 确定尺寸

        self.channels = channels    # 通道数，将输入图像的通道数信息传递给模型
        self.self_condition = self_condition    # 是否使用自条件
        input_channels = channels * (2 if self_condition else 1)    # 输入通道数，如果使用自条件，则有2个通道，一个是输入图像，另一个是前一步的预测

        init_dim = default(init_dim, dim)    # 初始维度
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)    # 初始化模型的卷积层，输入通道数为input_channels，输出通道数为init_dim，卷积核大小为7，填充为3
        # 卷积核就是移动的矩阵，padding就是填充，padding=3，就是左右上下各填充3个像素
        # 输入通道是3，输出通道是64：通过对输入的三个通道的加权输出一个特征，64个不同权重得到64个特征，一般默认是使用nn默认的Conv2d

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]    # 输入[64 , *map(lambda m: 64  * m, (1, 2, 4, 8))]，输出[64, 64, 128, 256, 512]
        in_out = list(zip(dims[:-1], dims[1:]))    # dims = [64, 64, 128, 256, 512]  ---->  [(64, 64), (64, 128), (128, 256), (256, 512)]

        # time embeddings 时间嵌入

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features     # 是否使用随机或学习到的正弦条件

        # 如果使用随机或学习到的正弦条件，则使用随机或学习到的正弦条件（这个没看呢）
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        # 时间映射网络（这个没看）
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,       
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:       # 输出(False, False, False, True),默认仅在最高分辨率最低的深层使用完整注意力，以平衡计算效率和模型性能。
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)     # 输出dim_mults的长度
        full_attn  = cast_tuple(full_attn, num_stages)  # 将full_attn转换为长度为num_stages的元组，比如输入full_attn = 4，num_stages=3，输出(4, 4, 4)
        attn_heads = cast_tuple(attn_heads, num_stages)   # 将attn_heads转换为长度为num_stages的元组
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)  # 将attn_dim_head转换为长度为num_stages的元组

        assert len(full_attn) == len(dim_mults)     # 断言full_attn和dim_mults的长度相同,不同会报错

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)  # 用于部分应用（partially apply）一个函数或类构造函数，固定某些参数，生成新的可调用对象
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        """
        若 flash_attn=True，则启用 Flash Attention
        attn_layer = FullAttention(dim=64, heads=8)
        等价于
        attn_layer = Attention(dim=64, heads=8, flash=True)
        """

        # layers
        """
        下采样：通过降低特征图的分辨率（如高度和宽度减半），同时增加通道数，提取更高层次的语义信息。
        上采样：通过增加特征图的分辨率（如高度和宽度加倍），同时减少通道数，恢复到原始分辨率，同时保留更多细节。
        示例说明：
        假设输入图像的分辨率为 256x256，通道数为 3（RGB图像），UNet 的下采样和上采样路径如下：

        下采样路径  -------- 压缩，忽略细节
        层级 0：
        输入：[B, 3, 256, 256]
        下采样：[B, 64, 128, 128]

        层级 1：
        输入：[B, 64, 128, 128]
        下采样：[B, 128, 64, 64]

        层级 2：
        输入：[B, 128, 64, 64]
        下采样：[B, 256, 32, 32]

        上采样路径  -------- 解压，恢复细节
        层级 2：
        输入：[B, 256, 32, 32]
        上采样：[B, 128, 64, 64]

        层级 1：
        输入：[B, 128, 64, 64]
        上采样：[B, 64, 128, 128]

        层级 0：
        输入：[B, 64, 128, 128]
        上采样：[B, 3, 256, 256]

        跨层连接
        上采样路径的每一层会与下采样路径的对应层进行特征拼接（torch.cat），融合低层次和高层次特征。
        """
        
        self.downs = ModuleList([])     # 定义一个ModuleList，用于存储下采样块
        self.ups = ModuleList([])       # 定义一个ModuleList，用于存储上采样块
        num_resolutions = len(in_out)   # 输出in_out的长度,总下采样次数

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)  # 判断是否是最后一层,is_last是bool类型，判断ind是否大于等于num_resolutions - 1

            attn_klass = FullAttention if layer_full_attn else LinearAttention  # 如果layer_full_attn为True，则使用FullAttention，否则使用LinearAttention

            # 将resnet_block、attn_klass、Downsample或nn.Conv2d添加到self.downs下采样列表中
            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),   # 输入维度为dim_in，输出维度为dim_in
                resnet_block(dim_in, dim_in),   # 输入维度为dim_in，输出维度为dim_in
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),   # 注意力层，输入维度为dim_in，注意力头的维度为layer_attn_dim_head，注意力头的数量为layer_attn_heads
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)  # 如果is_last为False，则使用下采样，否则使用卷积层
            ]))

        # 处理中间层
        mid_dim = dims[-1]  # 获取最深层的特征维度
        self.mid_block1 = resnet_block(mid_dim, mid_dim)    # 看完resnet_block的代码再说
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention  # 如果layer_full_attn为True，则使用FullAttention，否则使用LinearAttention

            # 将resnet_block、attn_klass、Upsample或nn.Conv2d添加到self.ups上采样列表中
            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)  # 如果需要学习方差，则输出维度为channels * 2，否则为channels
        self.out_dim = default(out_dim, default_out_dim)  # 输出维度，如果out_dim为None，则输出default_out_dim，否则输出out_dim

        # 处理最终层
        self.final_res_block = resnet_block(init_dim * 2, init_dim)  # 
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)  # 

    """
    属性的调用：
    如果method是一个属性，则可以直接调用，不需要加括号
    如果method是一个方法，则需要加括号
    """
    @property   # 定义一个属性，用于获取下采样因子
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)  # 下采样因子，2的幂次，len(self.downs) - 1表示下采样次数
    
    # 定义一个方法，用于前向传播
    def forward(self, x, time, x_self_cond = None):     # 参数分别是输入图像x，时间time，自条件x_self_cond
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        # 断言，确保输入的维度是下采样因子的倍数，否则会报错
        if self.self_condition:     # 如果使用自条件，则将自条件和输入图像拼接
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))     # 如果输入的自条件为None，则生成一个和输入图像x相同大小的全0张量，否则输入x_self_cond，也就是上一步预测结果
            x = torch.cat((x_self_cond, x), dim = 1)     # 将自条件和输入图像拼接，dim=1表示在通道维度上拼接

        x = self.init_conv(x)   # 初始化卷积层，输入图像x，输出卷积后的图像x
        r = x.clone()    # 克隆输入图像x，用于存储中间结果  

        t = self.time_mlp(time)    # 时间嵌入，输入时间time，输出时间嵌入t

        h = []    # 定义一个列表，用于存储中间结果

        # 遍历下采样块
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)    # block在前面已经实例化了，这里是调用block了。而ResnetBlock直接调用了forward，这是因为nn.Module 实现了 __call__ ，方法__call__ 方法内部会调用 forward
            h.append(x)    # 将 x 添加到 h 列表中，用于存储中间结果

            x = block2(x, t)    # 输入 x 通过第二个残差块 block2，同时注入时间条件 t
            x = attn(x) + x    # 对 x 进行注意力机制 attn，然后将结果与 x 相加
            h.append(x)    # 将 x 添加到 h 列表中，用于存储中间结果

            x = downsample(x)   # 下采样，输入x，输出下采样后的图像x

        x = self.mid_block1(x, t)    # 输入x和时间t，调用forward方法，同时嵌入时间t
        x = self.mid_attn(x) + x    # 对x进行注意力机制，然后将结果与x相加
        x = self.mid_block2(x, t)    # 输入x和时间t，调用forward方法，同时嵌入时间t

        # 遍历上采样块
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)    # 将x和h列表中的最后一个元素拼接，dim=1表示在通道维度上拼接
            x = block1(x, t)    # 输入x和时间t，调用forward方法，同时嵌入时间t

            x = torch.cat((x, h.pop()), dim = 1)    # 将x和h列表中的最后一个元素拼接，dim=1表示在通道维度上拼接
            x = block2(x, t)    # 输入x和时间t，调用forward方法，同时嵌入时间t
            x = attn(x) + x    # 对x进行注意力机制，然后将结果与x相加

            x = upsample(x)    # 上采样，输入x，输出上采样后的图像x

        x = torch.cat((x, r), dim = 1)    # 将原始图像r和上采样后的图像x拼接，dim=1表示在通道维度上拼接

        x = self.final_res_block(x, t)    # 输入x和时间t，调用forward方法，同时嵌入时间t
        return self.final_conv(x)    # 输入x，输出卷积后的图像x

# gaussian diffusion trainer class

def extract(a, t, x_shape):     # 用于从张量a中提取特定索引t对应的值，并将其重塑为可与x_shape广播的形状
    b, *_ = t.shape     # 将batch_size赋值给b，其余赋值给*_
    out = a.gather(-1, t)       # 从维度-1（最后一列）按照索引t来提取a的元素
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))     # 如果x_shape是(b, c, h, w)，则返回的形状是(b, 1, 1, 1)

def linear_beta_schedule(timesteps):    # 扩散模型通过逐步向数据添加噪声（正向过程）并学习如何逐步去除噪声（反向过程）来生成数据。噪声调度决定了在每个时间步添加多少噪声。
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001     # 正向过程中添加的噪声的初始值
    beta_end = scale * 0.02         # 正向过程中添加的噪声的结束值
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)       # timesteps = 1000,那么返回0.0001到0.02之间的1000个数

def cosine_beta_schedule(timesteps, s = 0.008):     # 与线性调度不同，余弦调度通过余弦函数来调整噪声的添加量，使得噪声的添加过程更加平滑，并且在训练的早期和晚期阶段更加稳定。
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps      # 生成从 0 到 timesteps 的均匀分布的时间点，并将其归一化到 [0, 1] 区间。
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2      # s 是一个小的偏移量，用于调整余弦函数的形状，使其在早期阶段更加平滑。
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]                 # 归一化余弦函数的值，使其在 t = 0 时为 1。
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])              # 计算每个时间点的噪声添加量，即每个时间点的 beta 值。计算公式是beta_t = 1-alpha_t/alpha_{t-1}
    return torch.clip(betas, 0, 0.999)      # 将beta值限制在 [0, 0.999] 区间, 防止噪声添加量过大。clip函数的作用是将输入张量的元素限制在指定范围内，比如将小于0的值设置为0，大于1的值设置为1。

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):           # 通过 Sigmoid 函数来调整噪声的添加量
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps      # 生成从 0 到 timesteps 的均匀分布的时间点，并将其归一化到 [0, 1] 区间。
    v_start = torch.tensor(start / tau).sigmoid()           # 通过 sigmoid 函数将起始值 start 缩放到 [0, 1] 区间。
    v_end = torch.tensor(end / tau).sigmoid()               # 通过 sigmoid 函数将结束值 end 缩放到 [0, 1] 区间。
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)       # 使用 Sigmoid 函数计算累积的 alpha 值
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]         # 归一化 alpha 值，使其在 t = 0 时为 1。
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])      # 计算每个时间点的噪声添加量，即每个时间点的 beta 值。计算公式是beta_t = 1-alpha_t/alpha_{t-1}
    return torch.clip(betas, 0, 0.999)                          # 将beta值限制在 [0, 0.999] 区间, 防止噪声添加量过大。clip函数的作用是将输入张量的元素限制在指定范围内，比如将小于0的值设置为0，大于1的值设置为1。

# 负责管理噪声的添加和去除过程
class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        immiscible = False
    ):
        super().__init__()          # 继承父类Module的初始化方法
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)        # 断言，如果模型的通道数不等于输出维度，则报错
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond           # 断言，如果模型有random_or_learned_sinusoidal_cond属性，则报错

        self.model = model      # 传入的模型

        self.channels = self.model.channels     # 传入模型的通道数
        self.self_condition = self.model.self_condition     # 传入模型是否使用自条件

        if isinstance(image_size, int):     # 检查image_size是否是整数
            image_size = (image_size, image_size)           # 如果是整数，则将image_size转换为元组
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'       # 判断image_size是否是元组或列表，且长度为2
        self.image_size = image_size    # 传入的图像大小

        self.objective = objective      # 传入的目标pred_v

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # 使用不同的调度函数
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)       # 通过调度函数计算beta值,schedule_fn_kwargs是一个字典，用于传递给调度函数的参数。这里是一个空字典

        alphas = 1. - betas     # 计算alpha值，alpha_t = 1 - beta_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)       # 计算累积alpha值，即alpha的_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)    # 计算alpha_t-1, 通过删除最后一个元素，然后在前面填充一个1

        timesteps, = betas.shape        # 获取beta的形状,注意beta必须是一维的
        self.num_timesteps = int(timesteps)     # 转换成整数

        # sampling related parameters   相关采样参数

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # 如果指定了sampling_timesteps，则使用指定的值，否则使用timesteps，作为采样时间步数

        assert self.sampling_timesteps <= timesteps         # 断言，采样时间步数必须小于等于总时间步数
        self.is_ddim_sampling = self.sampling_timesteps < timesteps     # 判断是否使用DDIM采样，即采样时间步数小于总时间步数时使用DDIM采样，否则使用正常采样（扩散过程）
        self.ddim_sampling_eta = ddim_sampling_eta      # DDIM采样的eta值,当eta的值为0时，采样将不带有随机性。注意DDIM是一种确定性的采样方法，它通过在扩散过程中的每个时间步中添加一个小的随机扰动来实现。

        # helper function to register buffer from float64 to float32    将 float64 类型的数据转换为 float32，然后注册为 torch.nn.Module 的缓冲区,以便在训练过程中被保存和加载，但不会被优化器更新。

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))   # 创建匿名函数，将val转换为float32类型，然后注册为torch.nn.Module的缓冲区。name是缓冲区的名称，val是缓冲区的值

        register_buffer('betas', betas)     # 注册beta值为缓冲区
        register_buffer('alphas_cumprod', alphas_cumprod)   # 注册累积alpha值为缓冲区
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  # 注册前一个累积alpha值为缓冲区

        # calculations for diffusion q(x_t | x_{t-1}) and others    扩散 q(x_t | x_{t-1}) 和其他计算

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))      # 注册累积alpha值的平方根为缓冲区
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))       # 注册1减去累积alpha值的平方根为缓冲区
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))         # 注册1减去累积alpha值的对数为缓冲区
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))           # 注册累积alpha值的倒数的平方根为缓冲区
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))     # 注册累积alpha值的倒数减1的平方根为缓冲区

        # calculations for posterior q(x_{t-1} | x_t, x_0)      后验 q(x_{t-1} | x_t, x_0) 的计算

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)         # 注册后验方差为缓冲区，计算公式是：beta_t * (1. - alpha_{t-1}) / (1. - alpha_t)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)       上面的等式等于 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)       # 注册后验方差为缓冲区

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain    下面的对数计算被截断，因为扩散链的开始处后验方差为0

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))      # 注册截断后的后验对数方差为缓冲区,为了防止出现负无穷大的情况，将后验方差限制在1e-20以上
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))    # 注册后验均值系数1为缓冲区，计算公式是：beta_t * sqrt(alpha_{t-1}) / (1. - alpha_t)
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))    # 注册后验均值系数2为缓冲区，计算公式是：(1. - alpha_{t-1}) * sqrt(alpha_t) / (1. - alpha_t)

        # immiscible diffusion

        self.immiscible = immiscible        # 是否使用不可混合扩散

        # offset noise strength - in blogpost, they claimed 0.1 was ideal       偏移噪声强度 - 在博客文章中，他们声称0.1是理想的

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight        衍生损失权重
        # snr - signal noise ratio      信噪比

        snr = alphas_cumprod / (1 - alphas_cumprod)     # 计算信噪比，信噪比是alpha的累积乘积除以1减去alpha的累积乘积

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()     # 克隆信噪比
        if min_snr_loss_weight:     # 如果最小信噪比损失权重为True
            maybe_clipped_snr.clamp_(max = min_snr_gamma)       # 限制信噪比的最大值为min_snr_gamma

        if objective == 'pred_noise':       # 如果目标是预测噪声
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':        # 如果目标是预测图像起始
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':         # 如果目标是预测v
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False         数据的自动归一化[0, 1] -> [-1, 1] - 可以通过将其设置为False来关闭

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity        # 如果auto_normalize为True，则使用normalize_to_neg_one_to_one，否则使用identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity       # 如果auto_normalize为True，则使用unnormalize_to_zero_to_one，否则使用identity

    @property
    def device(self):   # 获取betas使用的设备，gpu或cpu
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):      # 从噪声恢复原始图像, 即去噪. x_t表示t时刻的图像，t表示时间步，noise表示噪声
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -       
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise     # 正向过程公式 
        )       # sqrt_recip_alphas_cumprod: 用于恢复信号. sqrt_recipm1_alphas_cumprod: 用于去噪
    """
    extract函数用于从张量a中提取特定索引t对应的值，并将其重塑为可与x_shape广播的形状。
    """

    def predict_noise_from_start(self, x_t, t, x0):     # 从原始图像预测噪声
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)     # 通过正向过程求解噪声
        )

    def predict_v(self, x_start, t, noise):         # 从噪声预测v, v是一个参数化的噪声
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):      # 从v预测原始图像
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):         # 计算后验分布
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 通过模型预测噪声、图像起始或v，x_self_cond是自条件，clip_x_start表示是否裁剪x_start，rederive_pred_noise表示是否重新计算预测噪声      
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)        # 这个model是前面GasussianDiffusion类中传入的模型 (在下面这个output用于不同的目，这个不太懂)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity    # 如果clip_x_start为True，则使用torch.clamp函数，否则使用identity(相当于恒等函数)

        if self.objective == 'pred_noise':      # 如果目标是预测噪声
            pred_noise = model_output       # 预测噪声就是模型的输出
            x_start = self.predict_start_from_noise(x, t, pred_noise)       # 从噪声恢复原始图像
            x_start = maybe_clip(x_start)       # 如果clip_x_start为True，则裁剪x_start

            if clip_x_start and rederive_pred_noise:        # 如果clip_x_start为True且rederive_pred_noise为True
                pred_noise = self.predict_noise_from_start(x, t, x_start)       # 重新计算预测噪声

        elif self.objective == 'pred_x0':       # 如果目标是预测原始图像
            x_start = model_output      # 预测图像起始就是模型的输出
            x_start = maybe_clip(x_start)       
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)     # 输出预测噪声和图像起始，可以通过ModelPrediction.pred_x_start和输出预测噪声和图像起始，可以通过ModelPrediction.pred_noise调用他

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):      # 计算模型的均值和方差
        preds = self.model_predictions(x, t, x_self_cond)       # 通过模型预测噪声、图像起始或v
        x_start = preds.pred_x_start

        if clip_denoised:       # 如果clip_denoised为True，则裁剪去噪后的图像
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)        # 通过后验分布计算模型的均值和方差
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# dataset classes

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer:
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
