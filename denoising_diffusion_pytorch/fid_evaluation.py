import math
import os

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm


def num_to_groups(num, divisor):         # 将num分成divisor个组，例如num=10，divisor=3，则返回[3,3,4]
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 计算FID分数的类
class FIDEvaluation:
    def __init__(
        self,
        batch_size,      # 批量大小
        dl,              # 数据加载器
        sampler,         # 采样器
        channels=3,      # 通道数
        accelerator=None, # 加速器
        stats_dir="./results", # 统计目录
        device="cuda",   # 设备
        num_fid_samples=50000, # 样本数量
        inception_block_idx=2048, # inception block索引
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print      # 如果加速器为None，则使用print，否则使用加速器的print
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM     # 断言inception block索引在InceptionV3.BLOCK_INDEX_BY_DIM中
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]   # 获取inception block索引
        self.inception_v3 = InceptionV3([block_idx]).to(device)          # 创建inception v3模型
        self.dataset_stats_loaded = False    # 数据集统计是否加载

    def calculate_inception_features(self, samples):     # 计算inception特征
        if self.channels == 1:   # 如果通道数为1，则将样本扩展为3通道
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)         # 将样本扩展为3通道，直接复制样本

        self.inception_v3.eval()     # 设置inception v3模型为评估模式
        features = self.inception_v3(samples)[0]   # 计算inception特征

        if features.size(2) != 1 or features.size(3) != 1:   # 如果特征的尺寸不是1x1，则将特征调整为1x1
            features = adaptive_avg_pool2d(features, output_size=(1, 1))   # 将特征调整为1x1
        features = rearrange(features, "... 1 1 -> ...")   # 将特征展平
        return features

    def load_or_precalc_dataset_stats(self):     # 加载或预计算数据集统计
        path = os.path.join(self.stats_dir, "dataset_stats")   # 获取数据集统计路径
        try:
            ckpt = np.load(path + ".npz")   # 加载数据集统计
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]   # 获取数据集统计
            self.print_fn("Dataset stats loaded from disk.")   # 打印数据集统计加载成功
            ckpt.close()   # 关闭数据集统计文件
        except OSError:   # 如果数据集统计文件不存在
            num_batches = int(math.ceil(self.n_samples / self.batch_size))   # 计算数据集统计的批次数
            stacked_real_features = []   # 初始化一个空列表，用于存储真实数据集的特征
            self.print_fn(  
                f"Stacking Inception features for {self.n_samples} samples from the real dataset."   # 打印数据集统计加载成功
            )
            for _ in tqdm(range(num_batches)):   # 遍历数据集的每个批次
                try:
                    real_samples = next(self.dl)   # 获取真实数据集的样本
                except StopIteration:   # 如果数据集结束
                    break
                real_samples = real_samples.to(self.device)   # 将真实数据集的样本移动到设备上
                real_features = self.calculate_inception_features(real_samples)   # 计算真实数据集的inception特征
                stacked_real_features.append(real_features)   # 将真实数据集的inception特征添加到列表中
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()   # 将真实数据集的inception特征展平
            )
            m2 = np.mean(stacked_real_features, axis=0)   # 计算真实数据集的inception特征的均值
            s2 = np.cov(stacked_real_features, rowvar=False)   # 计算真实数据集的inception特征的协方差
            np.savez_compressed(path, m2=m2, s2=s2)   # 保存真实数据集的inception特征的均值和协方差
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")   # 打印数据集统计缓存成功
            self.m2, self.s2 = m2, s2   # 更新数据集统计
        self.dataset_stats_loaded = True   # 设置数据集统计加载状态

    @torch.inference_mode()
    def fid_score(self):     # 计算FID分数
        if not self.dataset_stats_loaded:   # 如果数据集统计未加载
            self.load_or_precalc_dataset_stats()   # 加载或预计算数据集统计
        self.sampler.eval()   # 设置采样器为评估模式
        batches = num_to_groups(self.n_samples, self.batch_size)   # 将样本数量分成批次数
        stacked_fake_features = []   # 初始化一个空列表，用于存储生成数据集的特征
        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."   # 打印生成数据集的特征
        )
        for batch in tqdm(batches):   # 遍历生成数据集的每个批次
            fake_samples = self.sampler.sample(batch_size=batch)   # 获取生成数据集的样本
            fake_features = self.calculate_inception_features(fake_samples)   # 计算生成数据集的inception特征
            stacked_fake_features.append(fake_features)   # 将生成数据集的inception特征添加到列表中
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()   # 将生成数据集的inception特征展平
        m1 = np.mean(stacked_fake_features, axis=0)   # 计算生成数据集的inception特征的均值
        s1 = np.cov(stacked_fake_features, rowvar=False)   # 计算生成数据集的inception特征的协方差

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)   # 计算FID分数
