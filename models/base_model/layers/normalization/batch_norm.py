# 批量归一化
import torch
from .layer_norm import LayerNorm

class BatchNorm(LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, momentum=0.1, elementwise_affine=True):
        """
        初始化批量归一化层

        Args:
            normalized_shape (int or tuple): 归一化的特征维度数，通常是channels数。
            eps (float): 用于数值稳定性的极小值，防止除以零。
            momentum (float): 用于更新均值和方差的动量。
            elementwise_affine (bool): 是否学习可训练的缩放系数gamma和偏移量beta。
        """
        super(BatchNorm, self).__init__(normalized_shape, eps, elementwise_affine)

        self.momentum = momentum
        self.running_mean = torch.zeros(normalized_shape)
        self.running_var = torch.ones(normalized_shape)

    def forward(self, x):
        """
        前向传播函数，执行批量归一化

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, channels, ...)

        Returns:
            Tensor: 归一化后的输出张量，形状与输入相同
        """
        if self.training:
            # 计算当前批次的均值和方差
            mean = x.mean(dim=0, keepdim=True)  # 按batch维度计算均值
            variance = ((x - mean) ** 2).mean(dim=0, keepdim=True)  # 按batch维度计算方差

            # 更新running均值和方差（用于推理时使用）
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance
        else:
            # 推理时使用训练时的均值和方差
            mean = self.running_mean
            variance = self.running_var

        # 标准化
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # 如果使用可训练的gamma和beta进行缩放与偏移
        if self.elementwise_affine:
            x_normalized = self.gamma * x_normalized + self.beta

        return x_normalized


class BatchNorm2d(BatchNorm):
    """
    用于二维输入的批量归一化，通常用于处理4D张量（如卷积层的输出特征图）。
    """

    def __init__(self, normalized_shape, eps=1e-5, momentum=0.1, elementwise_affine=True):
        # normalized_shape通常是通道数
        super(BatchNorm2d, self).__init__(normalized_shape, eps, momentum, elementwise_affine)

        # 初始化 running_mean 和 running_var 为 (channels, 1, 1)
        self.running_mean = torch.zeros(normalized_shape, 1, 1)
        self.running_var = torch.ones(normalized_shape, 1, 1)

    def forward(self, x):
        """
        对4D张量（例如卷积层的输出）进行归一化，通常用于CNN模型中的卷积层输出。
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)

        Returns:
            Tensor: 归一化后的输出张量，形状与输入相同
        """
        if self.training:
            # 计算当前批次的均值和方差
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # 按batch、height、width维度计算均值
            variance = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)  # 按batch、height、width维度计算方差

            # 更新running均值和方差（用于推理时使用）
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance
        else:
            # 推理时使用训练时的均值和方差
            mean = self.running_mean
            variance = self.running_var

            # 标准化
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # 如果使用可训练的gamma和beta进行缩放与偏移
        if self.elementwise_affine:
            x_normalized = self.gamma.view(1, -1, 1, 1) * x_normalized + self.beta.view(1, -1, 1, 1)

        return x_normalized

    def extra_repr(self):
        return (f'normalized_shape={self.normalized_shape}, eps={self.eps}, '
                f'momentum={self.momentum}, elementwise_affine={self.elementwise_affine}')


class BatchNorm3d(BatchNorm):
    """
    用于三维输入的批量归一化，通常用于处理5D张量（如3D卷积网络的输出）。
    """

    def __init__(self, normalized_shape, eps=1e-5, momentum=0.1, elementwise_affine=True):
        # normalized_shape通常是通道数
        super(BatchNorm3d, self).__init__(normalized_shape, eps, momentum, elementwise_affine)

        # 初始化 running_mean 和 running_var 为 (channels, 1, 1, 1)
        self.running_mean = torch.zeros(normalized_shape, 1, 1, 1)
        self.running_var = torch.ones(normalized_shape, 1, 1, 1)

    def forward(self, x):
        """
        对5D张量（例如3D卷积层的输出）进行归一化，通常用于3D卷积模型的输出。
        """
        if self.training:
            # 计算当前批次的均值和方差
            mean = x.mean(dim=(0, 2, 3, 4), keepdim=True)  # 按batch、depth、height、width维度计算均值
            variance = ((x - mean) ** 2).mean(dim=(0, 2, 3, 4), keepdim=True)  # 按batch、depth、height、width维度计算方差

            # 更新running均值和方差（用于推理时使用）
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance
        else:
            # 推理时使用训练时的均值和方差
            mean = self.running_mean
            variance = self.running_var

            # 标准化
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # 如果使用可训练的gamma和beta进行缩放与偏移
        if self.elementwise_affine:
            x_normalized = self.gamma.view(1, -1, 1, 1, 1) * x_normalized + self.beta.view(1, -1, 1, 1, 1)

        return x_normalized



if __name__ == '__main__':
    # 测试 BatchNorm
    def test_batchnorm():
        # 模拟一维输入张量，形状为 (batch_size=4, channels=3)
        x_1d = torch.randn(4, 3)

        # 创建一个 BatchNorm 层，normalized_shape=channels=3
        bn = BatchNorm(normalized_shape=3, eps=1e-5, momentum=0.1, elementwise_affine=True)

        # 测试训练模式
        bn.train()  # 设置为训练模式
        output_train_1d = bn(x_1d)
        print("一维输入 - 训练模式输出形状:", output_train_1d.shape)  # 应该与输入形状相同 (4, 3)

        # 测试推理模式
        bn.eval()  # 设置为推理模式
        output_eval_1d = bn(x_1d)
        print("一维输入 - 推理模式输出形状:", output_eval_1d.shape)  # 应该与输入形状相同 (4, 3)


    # 测试 BatchNorm2d
    def test_batchnorm2d():
        # 模拟输入张量，形状为 (batch_size=4, channels=3, height=5, width=5)
        x_2d = torch.randn(4, 3, 5, 5)

        # 创建一个 BatchNorm2d 层，normalized_shape=channels=3
        bn2d = BatchNorm2d(normalized_shape=3, eps=1e-5, momentum=0.1, elementwise_affine=True)

        # 测试训练模式
        bn2d.train()  # 设置为训练模式
        output_train_2d = bn2d(x_2d)
        print("二维输入 - 训练模式输出形状:", output_train_2d.shape)  # 应该与输入形状相同 (4, 3, 5, 5)

        # 测试推理模式
        bn2d.eval()  # 设置为推理模式
        output_eval_2d = bn2d(x_2d)
        print("二维输入 - 推理模式输出形状:", output_eval_2d.shape)  # 应该与输入形状相同 (4, 3, 5, 5)


    # 测试 BatchNorm3d
    def test_batchnorm3d():
        # 模拟输入张量，形状为 (batch_size=2, channels=3, depth=4, height=5, width=5)
        x_3d = torch.randn(2, 3, 4, 5, 5)

        # 创建一个 BatchNorm3d 层，normalized_shape=channels=3
        bn3d = BatchNorm3d(normalized_shape=3, eps=1e-5, momentum=0.1, elementwise_affine=True)

        # 测试训练模式
        bn3d.train()  # 设置为训练模式
        output_train_3d = bn3d(x_3d)
        print("三维输入 - 训练模式输出形状:", output_train_3d.shape)  # 应该与输入形状相同 (2, 3, 4, 5, 5)

        # 测试推理模式
        bn3d.eval()  # 设置为推理模式
        output_eval_3d = bn3d(x_3d)
        print("三维输入 - 推理模式输出形状:", output_eval_3d.shape)  # 应该与输入形状相同 (2, 3, 4, 5, 5)


    # 执行测试
    print("测试 BatchNorm:")
    test_batchnorm()
    print("\n测试 BatchNorm2d:")
    test_batchnorm2d()
    print("\n测试 BatchNorm3d:")
    test_batchnorm3d()



