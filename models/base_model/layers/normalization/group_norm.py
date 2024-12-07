# 组归一化（Group Normalization）
import torch
from .layer_norm import LayerNorm


class GroupNorm(LayerNorm):
    def __init__(self, num_groups, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        初始化组归一化层

        Args:
            num_groups (int): 特征维度分成多少组进行归一化。
            normalized_shape (int or tuple): 归一化的特征维度数，可以是单个整数或包含多个整数的元组。
            eps (float): 用于数值稳定性的极小值，防止除以零。
            elementwise_affine (bool): 是否学习可训练的缩放系数gamma和偏移量beta。
        """
        super().__init__(normalized_shape, eps, elementwise_affine)

        self.num_groups = num_groups
        self.running_mean = torch.zeros(normalized_shape)
        self.running_var = torch.ones(normalized_shape)

    def _calculate_mean_var(self, x):
        # 输出张量的形状，确保维度符合预期
        print(f"x shape: {x.shape}")

        if x.dim() == 2:
            # 对于 2D 张量，计算沿着 `dim=1` 的均值和方差
            mean = x.mean(dim=1, keepdim=True)
            variance = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        elif x.dim() == 3:
            # 对于 3D 张量，计算沿着 `dim=2, 3` 的均值和方差
            mean = x.mean(dim=2, keepdim=True)
            variance = ((x - mean) ** 2).mean(dim=2, keepdim=True)
        elif x.dim() == 4:
            # 对于 4D 张量，计算沿着 `dim=2, 3` 的均值和方差
            mean = x.mean(dim=[2, 3], keepdim=True)
            variance = ((x - mean) ** 2).mean(dim=[2, 3], keepdim=True)
        elif x.dim() == 5:
            # 对于 5D 张量，计算沿着 `dim=2, 3, 4` 的均值和方差
            mean = x.mean(dim=[2, 3, 4], keepdim=True)
            variance = ((x - mean) ** 2).mean(dim=[2, 3, 4], keepdim=True)
        elif x.dim() == 6:
            # 对于 5D 张量，计算沿着 `dim=2, 3, 4` 的均值和方差
            mean = x.mean(dim=[2, 3, 4, 5], keepdim=True)
            variance = ((x - mean) ** 2).mean(dim=[2, 3, 4, 5], keepdim=True)
        else:
            # 如果是更高维度的输入，需要相应处理
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

        return mean, variance

    def _normalize(self, x, mean, variance):
        """
        对张量进行标准化
        """
        return (x - mean) / torch.sqrt(variance + self.eps)

    def forward(self, x):
        """
        前向传播函数，执行组归一化

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, ..., feature_dim)

        Returns:
            Tensor: 归一化后的输出张量，形状与输入相同
        """
        # 计算分组数量
        batch_size, num_channels = x.shape[0], x.shape[1]
        group_size = num_channels // self.num_groups

        # 将输入张量重塑为 (batch_size, num_groups, group_size, height, width)
        x = x.view(batch_size, self.num_groups, group_size, *x.shape[2:])

        # 计算均值和方差
        mean, variance = self._calculate_mean_var(x)

        # 标准化
        x_normalized = self._normalize(x, mean, variance)

        # 如果使用可训练的gamma和beta进行缩放与偏移
        if self.elementwise_affine:
            x_normalized = self.gamma.view(1, -1, 1, 1, 1) * x_normalized + self.beta.view(1, -1, 1, 1, 1)

        return x_normalized

    def extra_repr(self):
        return (f'num_groups={self.num_groups}, normalized_shape={self.normalized_shape}, '
                f'eps={self.eps}, elementwise_affine={self.elementwise_affine}')


class GroupNorm2d(GroupNorm):
    """
    用于二维输入的组归一化，可以处理4D张量（如卷积层输出的特征图）。
    """

    def __init__(self, num_groups, normalized_shape, eps=1e-5, elementwise_affine=True):
        # `normalized_shape` 这里指的是特征图的高度和宽度
        super(GroupNorm2d, self).__init__(num_groups, normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        """
        对4D张量（例如卷积层的输出）进行组归一化，通常用于CNN模型中的卷积层输出。
        """
        # 输入形状为 (batch_size, channels, height, width) 的4D张量
        batch_size, num_channels, height, width = x.shape
        group_size = num_channels // self.num_groups

        # 将输入张量重塑为 (batch_size, num_groups, group_size, height, width)
        x = x.view(batch_size, self.num_groups, group_size, height, width)

        # 计算均值和方差
        mean, variance = self._calculate_mean_var(x)

        # 标准化
        x_normalized = self._normalize(x, mean, variance)

        # 如果使用可训练的gamma和beta进行缩放与偏移
        if self.elementwise_affine:
            x_normalized = self.gamma.view(1, -1, 1, 1) * x_normalized + self.beta.view(1, -1, 1, 1)

        return x_normalized



class GroupNorm3d(GroupNorm):
    """
    用于三维输入的组归一化，可以处理5D张量（例如3D卷积网络的输出）。
    """

    def __init__(self, num_groups, normalized_shape, eps=1e-5, elementwise_affine=True):
        # `normalized_shape` 这里指的是3D卷积层的空间维度
        super(GroupNorm3d, self).__init__(num_groups, normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        """
        对5D张量（例如3D卷积层的输出）进行组归一化，通常用于3D卷积模型的输出。
        """
        batch_size, num_channels, depth, height, width = x.shape
        group_size = num_channels // self.num_groups

        # 将输入张量重塑为 (batch_size, num_groups, group_size, depth, height, width)
        x = x.view(batch_size, self.num_groups, group_size, depth, height, width)

        # 计算均值和方差
        mean, variance = self._calculate_mean_var(x)

        # 标准化
        x_normalized = self._normalize(x, mean, variance)

        # 如果使用可训练的gamma和beta进行缩放与偏移
        if self.elementwise_affine:
            x_normalized = self.gamma.view(1, -1, 1, 1, 1, 1) * x_normalized + self.beta.view(1, -1, 1, 1, 1, 1)

        return x_normalized



if __name__ == '__main__':
    # 测试 GroupNorm
    def test_groupnorm():
        # 模拟一维输入张量，形状为 (batch_size=4, channels=3)
        x_1d = torch.randn(4, 3)

        # 创建一个 GroupNorm 层，num_groups=1（表示每个通道作为一个组），normalized_shape=3
        gn = GroupNorm(num_groups=1, normalized_shape=3, eps=1e-5, elementwise_affine=True)

        # 测试训练模式
        gn.train()  # 设置为训练模式
        output_train_1d = gn(x_1d)
        print("一维输入 - 训练模式输出形状:", output_train_1d.shape)  # 应该与输入形状相同 (4, 3)

        # 测试推理模式
        gn.eval()  # 设置为推理模式
        output_eval_1d = gn(x_1d)
        print("一维输入 - 推理模式输出形状:", output_eval_1d.shape)  # 应该与输入形状相同 (4, 3)


    # 测试 GroupNorm2d
    def test_groupnorm2d():
        # 模拟输入张量，形状为 (batch_size=4, channels=3, height=5, width=5)
        x_2d = torch.randn(4, 3, 5, 5)

        # 创建一个 GroupNorm2d 层，num_groups=3，normalized_shape=3
        gn2d = GroupNorm2d(num_groups=3, normalized_shape=3, eps=1e-5, elementwise_affine=True)

        # 测试训练模式
        gn2d.train()  # 设置为训练模式
        output_train_2d = gn2d(x_2d)
        print("二维输入 - 训练模式输出形状:", output_train_2d.shape)  # 应该与输入形状相同 (4, 3, 5, 5)

        # 测试推理模式
        gn2d.eval()  # 设置为推理模式
        output_eval_2d = gn2d(x_2d)
        print("二维输入 - 推理模式输出形状:", output_eval_2d.shape)  # 应该与输入形状相同 (4, 3, 5, 5)


    # 测试 GroupNorm3d
    def test_groupnorm3d():
        # 模拟输入张量，形状为 (batch_size=2, channels=3, depth=4, height=5, width=5)
        x_3d = torch.randn(2, 3, 4, 5, 5)

        # 创建一个 GroupNorm3d 层，num_groups=3，normalized_shape=3
        gn3d = GroupNorm3d(num_groups=3, normalized_shape=3, eps=1e-5, elementwise_affine=True)

        # 测试训练模式
        gn3d.train()  # 设置为训练模式
        output_train_3d = gn3d(x_3d)
        print("三维输入 - 训练模式输出形状:", output_train_3d.shape)  # 应该与输入形状相同 (2, 3, 4, 5, 5)

        # 测试推理模式
        gn3d.eval()  # 设置为推理模式
        output_eval_3d = gn3d(x_3d)
        print("三维输入 - 推理模式输出形状:", output_eval_3d.shape)  # 应该与输入形状相同 (2, 3, 4, 5, 5)


    # 执行测试
    print("测试 GroupNorm:")
    test_groupnorm()
    print("\n测试 GroupNorm2d:")
    test_groupnorm2d()
    print("\n测试 GroupNorm3d:")
    test_groupnorm3d()

