# 层归一化实现（Layer Normalization）
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        初始化层归一化层

        Args:
            normalized_shape (int or tuple): 归一化的特征维度数，可以是单个整数或包含多个整数的元组。
            eps (float): 用于数值稳定性的极小值，防止除以零。
            elementwise_affine (bool): 是否学习可训练的缩放系数gamma和偏移量beta。
        """
        super(LayerNorm, self).__init__()

        # 如果输入是一个整数（例如: 输入是2D张量），则扩展为一个元组
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.eps = eps
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine

        # 可学习的缩放系数gamma和偏移量beta
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        """
        前向传播函数，执行层归一化

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, ..., feature_dim)

        Returns:
            Tensor: 归一化后的输出张量，形状与输入相同
        """
        # 计算每个样本的均值和方差
        mean = x.mean(dim=-1, keepdim=True)  # 均值计算
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)  # 方差计算

        # 标准化
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # 如果使用可训练的gamma和beta进行缩放与偏移
        if self.elementwise_affine:
            x_normalized = self.gamma * x_normalized + self.beta

        return x_normalized

    def extra_repr(self):
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class LayerNorm2d(LayerNorm):
    """
    用于二维输入的层归一化，可以处理4D张量（如卷积层输出的特征图）。
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        # `normalized_shape` 这里指的是特征图的高度和宽度
        super(LayerNorm2d, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        """
        对4D张量（例如卷积层的输出）进行归一化，通常用于CNN模型中的卷积层输出。
        """
        # 这里假设输入为 (batch_size, channels, height, width) 的4D张量
        mean = x.mean(dim=[2, 3], keepdim=True)  # 计算空间维度（height, width）上的均值
        variance = ((x - mean) ** 2).mean(dim=[2, 3], keepdim=True)  # 计算空间维度上的方差

        # 标准化
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # 可训练的缩放和偏移
        if self.elementwise_affine:
            # 确保gamma和beta的形状与通道维度匹配
            self.gamma = nn.Parameter(torch.ones((x.shape[1], 1, 1)))
            self.beta = nn.Parameter(torch.zeros((x.shape[1], 1, 1)))
            x_normalized = self.gamma * x_normalized + self.beta

        return x_normalized


class LayerNorm3d(LayerNorm):
    """
    用于三维输入的层归一化，可以处理5D张量（例如3D卷积网络的输出）。
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        # `normalized_shape` 这里指的是3D卷积层的空间维度
        super(LayerNorm3d, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        """
        对5D张量（例如3D卷积层的输出）进行归一化，通常用于3D卷积模型的输出。
        """
        # 输入形状为 (batch_size, channels, depth, height, width)
        mean = x.mean(dim=[2, 3, 4], keepdim=True)  # 计算空间维度（depth, height, width）上的均值
        variance = ((x - mean) ** 2).mean(dim=[2, 3, 4], keepdim=True)  # 计算空间维度上的方差

        # 标准化
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # 可训练的缩放和偏移
        if self.elementwise_affine:
            # 这里gamma和beta是针对每个channel进行的
            self.gamma = nn.Parameter(torch.ones((x.shape[1], 1, 1, 1)))
            self.beta = nn.Parameter(torch.zeros((x.shape[1], 1, 1, 1)))
            x_normalized = self.gamma.view(1, -1, 1, 1, 1) * x_normalized + self.beta.view(1, -1, 1, 1, 1)

        return x_normalized


if __name__ == '__main__':
    # 用法示例
    # 对于1D输入（例如全连接层的输出）
    input_tensor = torch.randn(32, 128)  # 假设batch_size=32, feature_dim=128
    layer_norm = LayerNorm(normalized_shape=128)
    output_tensor = layer_norm(input_tensor)
    print(output_tensor.shape)  # 输出 (32, 128)

    # 对于2D输入（例如卷积层的输出，4D张量）
    input_tensor_2d = torch.randn(32, 64, 128, 128)  # 假设batch_size=32, channels=64, height=128, width=128
    layer_norm_2d = LayerNorm2d(normalized_shape=(128, 128))
    output_tensor_2d = layer_norm_2d(input_tensor_2d)
    print(output_tensor_2d.shape)  # 输出 (32, 64, 128, 128)

    # 对于3D输入（例如3D卷积层的输出，5D张量）
    input_tensor_3d = torch.randn(32, 64, 16, 128, 128)  # 假设batch_size=32, channels=64, depth=16, height=128, width=128
    layer_norm_3d = LayerNorm3d(normalized_shape=(16, 128, 128))
    output_tensor_3d = layer_norm_3d(input_tensor_3d)
    print(output_tensor_3d.shape)  # 输出 (32, 64, 16, 128, 128)
