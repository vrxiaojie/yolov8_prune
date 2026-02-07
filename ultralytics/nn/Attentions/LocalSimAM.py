import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalSimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        """
        LocalSimAM: Local Simple Attention Mechanism
        论文来源: 基于局部特征和特征融合的无人驾驶场景目标检测方法 (纪涛, 等)

        Args:
            e_lambda (float): 正则化项，防止除零错误 (论文中提到为 1e-6，这里作为可配置参数)
        """
        super(LocalSimAM, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        """
        x: 输入特征图, shape [Batch, Channel, Height, Width]
        """
        b, c, h, w = x.size()

        # 论文核心：不再计算全局均值和方差，而是基于 3x3 滑动窗口计算局部统计量
        # n = w_2 = 3 * 3 = 9 (窗口内的像素数量)
        n = 9

        # 为了高效实现滑动窗口统计，可以使用 unfold 或者 AveragePool
        # 使用 AveragePool2d 计算局部均值 mu
        # kernel_size=3, stride=1, padding=1 保证输出尺寸不变
        mu = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # 计算局部方差: E[x^2] - (E[x])^2
        # 先计算 x的平方的局部均值
        x_sq = x.pow(2)
        mu_sq = F.avg_pool2d(x_sq, kernel_size=3, stride=1, padding=1)

        # var = E[x^2] - E[x]^2
        var = mu_sq - mu.pow(2)

        # 论文公式 (1): y_ij 计算
        # 注意：论文公式中分母部分是计算能量函数，SimAM 的推导最终简化为:
        # e_t = (x - mu)^2 / (4 * (var + lambda)) + 0.5
        # 这里的实现基于 SimAM 的快速解法，但统计量 mu 和 var 是局部的

        # 计算能量函数 energy (1/e*)
        # 使得与背景差异大的神经元获得更高的权重
        y = (x - mu).pow(2) / (4 * (var + self.e_lambda)) + 0.5

        # Sigmoid 激活生成注意力权重
        # 论文描述: "通过归一化处理和 Sigmoid 激活函数生成注意力权重"
        att = torch.sigmoid(y)

        # 最终输出: 输入特征图 * 注意力权重
        return x * att


if __name__ == "__main__":
    # 测试代码
    input_tensor = torch.randn(2, 64, 32, 32)
    model = LocalSimAM(e_lambda=1e-6)
    output_tensor = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
