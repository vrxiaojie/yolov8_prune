import torch
from torch import nn

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor  # 分组数，默认为32
        assert channels // self.groups > 0  # 确保通道数能够被分组数整除
        self.softmax = nn.Softmax(-1)  # 定义 Softmax 层，用于最后一维度的归一化
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化，将特征图缩小为1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 自适应平均池化，保留高度维度，将宽度压缩为1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 自适应平均池化，保留宽度维度，将高度压缩为1
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 分组归一化
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)  # 1x1卷积
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)  # 3x3卷积

    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入张量的尺寸：批次、通道、高度、宽度
        group_x = x.reshape(b * self.groups, -1, h, w)  # 将张量按组重构：批次*组数, 通道/组数, 高度, 宽度
        x_h = self.pool_h(group_x)  # 对高度方向进行池化，结果形状为 (b*groups, c//groups, h, 1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 对宽度方向进行池化，并转置结果形状为 (b*groups, c//groups, 1, w)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 将池化后的特征在高度方向拼接后进行1x1卷积
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 将卷积后的特征分为高度特征和宽度特征
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 结合高度和宽度特征，应用分组归一化
        x2 = self.conv3x3(group_x)  # 对重构后的张量应用3x3卷积
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x1 进行自适应平均池化并应用Softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 重构 x2 的形状为 (b*groups, c//groups, h*w)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 进行自适应平均池化并应用Softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 重构 x1 的形状为 (b*groups, c//groups, h*w)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  # 计算权重，并重构为 (b*groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)  # 将权重应用于原始张量，并重构为原始输入形状
