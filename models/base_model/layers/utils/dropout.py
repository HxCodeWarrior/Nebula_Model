# Dropout模块（可支持不同类型的Dropout）
import torch
import torch.nn as nn
import torch.cuda

class Dropout(nn.Module):
    def __init__(self,
                 p: float,
                 dropout_type: str = "standard",
                 min_p: float = 0.1,
                 decay_factor: float = 0.99,
                 smooth_factor: float = 0.05):
        """
        初始化Dropout层

        :param p: Dropout概率，即丢弃的比例
        :param dropout_type: Dropout类型，可选的类型有 ('standard', 'spatial', 'dropconnect', 'gaussian', 'alpha', 'adaptive', 'smooth')
        :param min_p: 适用于自适应Dropout的最小丢弃概率
        :param decay_factor: 自适应Dropout中每次更新时Dropout概率的衰减因子
        :param smooth_factor: 平滑衰减Dropout概率时的因子
        """
        super(Dropout, self).__init__()
        self.p = p  # 初始Dropout概率
        self.dropout_type = dropout_type  # 选择的Dropout类型
        self.min_p = min_p  # 自适应Dropout时的最小概率
        self.decay_factor = decay_factor  # 自适应Dropout的衰减因子
        self.smooth_factor = smooth_factor  # 平滑衰减的因子

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，根据选择的Dropout类型执行相应的Dropout操作
        :param x: 输入的张量
        :return: 经过Dropout处理后的张量
        """
        if self.dropout_type == 'standard':
            return self.standard_dropout(x)
        elif self.dropout_type == 'spatial':
            return self.spatial_dropout(x)
        elif self.dropout_type == 'dropconnect':
            return self.dropconnect(x)
        elif self.dropout_type == 'gaussian':
            return self.gaussian_dropout(x)
        elif self.dropout_type == 'alpha':
            return self.alpha_dropout(x)
        elif self.dropout_type == 'adaptive':
            return self.adaptive_dropout(x)
        elif self.dropout_type == 'smooth':
            return self.smooth_dropout(x)
        else:
            raise ValueError(f"Unsupported dropout type: {self.dropout_type}")

    def standard_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准Dropout：随机丢弃部分神经元的输出

        :param x: 输入张量
        :return: 经过标准Dropout处理后的张量
        """
        if self.training:
            # 生成一个与x相同大小的随机掩码，掩码的值大于p的地方保留数据，其余位置丢弃
            mask = torch.rand_like(x) > self.p
            x.mul_(mask)  # 使用掩码进行逐元素相乘，进行Dropout
            return x / (1 - self.p)  # 训练时，对剩余神经元进行扩展，保证期望输出一致
        else:
            return x  # 推理时，直接返回原输入

    def spatial_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        空间Dropout：对卷积层的整个特征图进行丢弃

        :param x: 输入张量，通常是4维张量(batch_size, channels, height, width)
        :return: 经过空间Dropout处理后的张量
        """
        if self.training:
            # 对于每个通道，随机丢弃整个通道
            mask = torch.rand(x.size(1), 1, 1) > self.p  # 只对每个通道的掩码生成随机值
            mask = mask.to(x.device)  # 确保mask在正确的设备上
            x.mul_(mask)  # 使用掩码进行逐元素相乘，进行Dropout
            return x / (1 - self.p)  # 训练时，对剩余神经元进行扩展，保证期望输出一致
        else:
            return x  # 推理时，直接返回原输入

    def dropconnect(self, x: torch.Tensor) -> torch.Tensor:
        """
        DropConnect：随机丢弃神经元的连接（即权重）

        :param x: 输入张量
        :return: 经过DropConnect处理后的张量
        """
        if self.training:
            # 随机生成一个与x相同大小的掩码，用于丢弃输入到权重矩阵的部分连接
            mask = torch.rand_like(x) > self.p
            x.mul_(mask)  # 使用掩码进行逐元素相乘，丢弃部分权重连接
            return x
        else:
            return x  # 推理时，直接返回原输入

    def gaussian_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        高斯Dropout：对神经元的输出加入高斯噪声

        :param x: 输入张量
        :return: 经过高斯噪声Dropout处理后的张量
        """
        if self.training:
            # 生成与x相同大小的高斯噪声，标准差为p
            noise = torch.normal(mean=1.0, std=self.p, size=x.size()).to(x.device)
            return x * noise  # 将高斯噪声与输入进行逐元素相乘
        else:
            return x  # 推理时，直接返回原输入

    def alpha_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alpha Dropout：专门用于tanh激活函数等非线性激活函数的Dropout

        :param x: 输入张量
        :return: 经过Alpha Dropout处理后的张量
        """
        if self.training:
            # 生成一个与x相同大小的随机掩码，掩码值大于p的地方保留数据，其余地方丢弃
            mask = torch.rand_like(x) > self.p
            return torch.where(mask, x, x.new_zeros(x.size()))  # 在丢弃的地方填充为零
        else:
            return x  # 推理时，直接返回原输入

    def adaptive_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        自适应Dropout：根据训练进度动态调整Dropout概率

        :param x: 输入张量
        :return: 经过自适应Dropout处理后的张量
        """
        if self.training:
            # 动态调整p的值，确保其不低于最小概率
            self.p = max(self.min_p, self.p * self.decay_factor)
        return self.standard_dropout(x)  # 使用标准Dropout进行处理

    def smooth_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        平滑Dropout：平滑衰减Dropout概率

        :param x: 输入张量
        :return: 经过平滑衰减Dropout处理后的张量
        """
        if self.training:
            # 平滑衰减p，确保p不低于零
            self.p = max(0.0, self.p - self.smooth_factor)
        return self.standard_dropout(x)  # 使用标准Dropout进行处理

    def _apply_sparse_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        使用稀疏掩码对张量进行内存高效的处理

        :param x: 输入张量
        :param mask: 掩码
        :return: 经过稀疏掩码处理后的张量
        """
        mask = mask.to(x.device)  # 确保mask在正确的设备上
        return torch.mul(x, mask)  # 将x与掩码逐元素相乘

    def apply_sparse_dropconnect(self, x: torch.Tensor) -> torch.Tensor:
        """
        稀疏DropConnect：使用稀疏矩阵进行DropConnect

        :param x: 输入张量
        :return: 经过稀疏DropConnect处理后的张量
        """
        if self.training:
            # 生成稀疏掩码，仅丢弃部分连接
            mask = torch.sparse.FloatTensor(
                torch.randint(0, x.numel(), (2, x.numel() // 2)),
                torch.ones(x.numel() // 2),
                size=torch.Size([x.numel()])
            ).to(x.device)
            return self._apply_sparse_mask(x, mask.view_as(x))  # 对输入应用稀疏掩码
        else:
            return x  # 推理时，直接返回原输入

    def forward_with_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理时的Dropout处理：推理阶段通常不进行Dropout，但可以进行模拟

        :param x: 输入张量
        :return: 经过推理处理后的张量
        """
        if not self.training:
            # 在推理时，可以模拟Dropout效果，通常返回x乘以(1 - p)，即模拟丢弃部分神经元
            if self.dropout_type == 'standard':
                return x * (1 - self.p)
            elif self.dropout_type == 'dropconnect':
                return x * (1 - self.p)
        return self.forward(x)  # 训练阶段使用常规的forward操作




if __name__ == '__main__':
    # 创建一个简单的测试用例
    def test_dropout():
        # 创建一个简单的输入张量 (batch_size=2, channels=3, height=4, width=4)
        x = torch.randn(2, 3, 4, 4)

        # 设置随机种子，以确保测试可重复
        torch.manual_seed(0)

        # 测试标准Dropout
        dropout_standard = Dropout(p=0.5, dropout_type='standard')
        print("标准Dropout:")
        print(dropout_standard(x))  # 应该丢弃50%的神经元输出

        # 测试空间Dropout
        dropout_spatial = Dropout(p=0.5, dropout_type='spatial')
        print("\n空间Dropout:")
        print(dropout_spatial(x))  # 应该丢弃50%的通道

        # 测试DropConnect
        dropout_dropconnect = Dropout(p=0.5, dropout_type='dropconnect')
        print("\nDropConnect:")
        print(dropout_dropconnect(x))  # 应该丢弃50%的权重连接

        # 测试高斯Dropout
        dropout_gaussian = Dropout(p=0.2, dropout_type='gaussian')
        print("\n高斯Dropout:")
        print(dropout_gaussian(x))  # 输出应该加入高斯噪声

        # 测试Alpha Dropout
        dropout_alpha = Dropout(p=0.5, dropout_type='alpha')
        print("\nAlpha Dropout:")
        print(dropout_alpha(x))  # 应该丢弃一些神经元并填充为0

        # 测试自适应Dropout
        dropout_adaptive = Dropout(p=0.5, dropout_type='adaptive', min_p=0.1, decay_factor=0.9)
        print("\n自适应Dropout:")
        print(dropout_adaptive(x))  # 在每次调用时，Dropout概率会衰减

        # 测试平滑Dropout
        dropout_smooth = Dropout(p=0.5, dropout_type='smooth', smooth_factor=0.05)
        print("\n平滑Dropout:")
        print(dropout_smooth(x))  # Dropout概率会平滑衰减

        # 测试推理阶段的Dropout
        dropout_inference = Dropout(p=0.5, dropout_type='standard')
        dropout_inference.eval()  # 设置为推理模式
        print("\n推理阶段的标准Dropout:")
        print(dropout_inference(x))  # 在推理时，Dropout应该不会应用

    test_dropout()

