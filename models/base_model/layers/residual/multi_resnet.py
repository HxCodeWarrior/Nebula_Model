# 多层残差链接模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_connection import ResConnect


class MultiResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, kernel_size=3, stride=1, padding=1,
                 use_batchnorm=True, use_activation=True, activation_fn=F.relu, use_residual=True,
                 init_method='kaiming', dropout_prob=0.0, dilation=1, skip_every_n=1):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param num_layers: 残差块的层数
        :param kernel_size: 卷积核大小
        :param stride: 卷积步幅
        :param padding: 卷积填充
        :param use_batchnorm: 是否使用批归一化
        :param use_activation: 是否使用激活函数
        :param activation_fn: 激活函数，默认是ReLU
        :param use_residual: 是否使用残差连接
        :param init_method: 权重初始化方法
        :param dropout_prob: Dropout 概率
        :param dilation: 膨胀卷积
        :param skip_every_n: 每 `skip_every_n` 层跳过连接
        """
        super(MultiResidualBlock, self).__init__()

        self.layers = nn.ModuleList()
        self.skip_every_n = skip_every_n

        # 创建多层残差连接
        for _ in range(num_layers):
            self.layers.append(ResConnect(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                use_batchnorm,
                use_activation,
                activation_fn,
                use_residual,
                init_method,
                dropout_prob,
                dilation))
            in_channels = out_channels  # 每一层的输入通道数是前一层的输出通道数

    def forward(self, x):
        """前向传播"""
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            # 每 `skip_every_n` 层跳过连接
            if self.skip_every_n > 1 and (idx + 1) % self.skip_every_n == 0:
                x = x + self.layers[idx](x)  # 累加输出，形成跳跃连接
        return x

if __name__ == '__main__':
    # 测试多层残差模块
    def test_multi_residual_block():
        # 设置输入的通道数和输出的通道数
        in_channels = 3  # 输入通道数（例如RGB图像）
        out_channels = 64  # 输出通道数
        batch_size = 8  # 测试批次大小
        height, width = 32, 32  # 输入图像的高度和宽度
        num_layers = 5  # 残差块的层数

        # 创建一个随机输入张量，模拟一个批量的图像数据（batch_size x in_channels x height x width）
        x = torch.randn(batch_size, in_channels, height, width)

        # 创建 MultiResidualBlock 实例
        multi_res_block = MultiResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            kernel_size=3,
            stride=1,
            padding=1,
            use_batchnorm=True,
            use_activation=True,
            activation_fn=F.relu,
            use_residual=True,
            init_method='kaiming',
            dropout_prob=0.0,
            dilation=1)

        # 前向传播
        output = multi_res_block(x)

        # 输出结果的形状
        print(f"Output shape: {output.shape}")
        print("Test Pass~")


    test_multi_residual_block()