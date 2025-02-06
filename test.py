import torch
import torch.nn as nn

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DenoisingTest:
    def __init__(self, timesteps=1000):
        # 模拟alphas系数
        betas = torch.linspace(0.0001, 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # 计算需要的系数
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

def main():
    # 创建测试实例
    model = DenoisingTest()
    
    # 创建测试数据
    batch_size = 2
    channels = 3
    height = width = 4  # 使用小尺寸便于展示
    
    # 创建噪声图像和时间步
    x_t = torch.randn(batch_size, channels, height, width)
    t = torch.tensor([100, 500])  # 两个不同的时间步
    noise = torch.randn_like(x_t)
    
    print("输入形状:")
    print(f"x_t shape: {x_t.shape}")
    print(f"t shape: {t.shape}")
    print(f"noise shape: {noise.shape}")
    
    # 执行预测
    x_0_pred = model.predict_start_from_noise(x_t, t, noise)
    print(f"\n预测输出 x_0 shape: {x_0_pred.shape}")
    
    # 展示系数
    print("\n提取的系数:")
    print(f"sqrt_recip_alphas_cumprod: {extract(model.sqrt_recip_alphas_cumprod, t, x_t.shape)}")
    print(f"sqrt_recipm1_alphas_cumprod: {extract(model.sqrt_recipm1_alphas_cumprod, t, x_t.shape)}")

if __name__ == "__main__":
    main()