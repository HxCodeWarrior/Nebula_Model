# 基本残差连接模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResConnect(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_batchnorm=True, use_activation=True, activation_fn=F.relu, use_residual=True,
                 init_method='kaiming', dropout_prob=0.0, dilation=1):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 卷积步幅
        :param padding: 卷积填充
        :param use_batchnorm: 是否使用批归一化
        :param use_activation: 是否使用激活函数
        :param activation_fn: 激活函数，默认是ReLU
        :param use_residual: 是否使用残差连接
        :param init_method: 权重初始化方法，默认是'kaiming'，可选'orthogonal'等
        :param dropout_prob: Dropout 概率，用于正则化
        :param dilation: 卷积的膨胀系数
        """
        super(ResConnect, self).__init__()

        self.use_residual = use_residual
        self.use_batchnorm = use_batchnorm
        self.use_activation = use_activation
        self.activation_fn = activation_fn
        self.dropout_prob = dropout_prob
        self.dilation = dilation

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation=dilation)

        # 批归一化层
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # Dropout层
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        # Shortcut处理：若输入与输出通道数不同，则调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        # 权重初始化
        self._initialize_weights(init_method)

    def _initialize_weights(self, init_method):
        """根据指定的初始化方法初始化卷积层权重"""
        if init_method == 'kaiming':
            init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        elif init_method == 'xavier':
            init.xavier_normal_(self.conv1.weight)
            init.xavier_normal_(self.conv2.weight)
        elif init_method == 'orthogonal':
            init.orthogonal_(self.conv1.weight)
            init.orthogonal_(self.conv2.weight)

        if self.conv1.bias is not None:
            init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            init.zeros_(self.conv2.bias)

    def forward(self, x):
        """前向传播"""
        # 第一层卷积 + 批归一化 + 激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out) if self.use_activation else out

        # Dropout层（若启用）
        out = self.dropout(out)

        # 第二层卷积 + 批归一化
        out = self.conv2(out)
        out = self.bn2(out)

        # 加上shortcut（残差连接）
        if self.use_residual:
            out += self.shortcut(x)

        # 激活函数应用于最终输出
        out = self.activation_fn(out) if self.use_activation else out
        return out

if __name__ == '__main__':
    def test_residual_block():
        # 设置输入的通道数和输出的通道数
        in_channels = 3  # 输入通道数（例如RGB图像）
        out_channels = 64  # 输出通道数
        batch_size = 8  # 测试批次大小
        height, width = 32, 32  # 输入图像的高度和宽度

        # 创建一个随机输入张量，模拟一个批量的图像数据（batch_size x in_channels x height x width）
        x = torch.randn(batch_size, in_channels, height, width)

        # 创建 ResConnect 实例
        residual_block = ResConnect(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=3, stride=1, padding=1, use_batchnorm=True,
                                    use_activation=True, activation_fn=F.relu, use_residual=True,
                                    init_method='kaiming', dropout_prob=0.2, dilation=1)

        # 前向传播
        output = residual_block(x)

        # 输出的形状应该是(batch_size, out_channels, height, width)
        print(f"Output shape: {output.shape}")

        # 测试成功标准：输出的形状是否符合预期
        assert output.shape == (batch_size, out_channels, height, width), "Test failed!"
        print("Test passed!")


    # 执行测试
    test_residual_block()
