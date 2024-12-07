# 梯度裁剪（Gradient Clipping，防止梯度爆炸）
import torch
import torch.nn as nn
import torch.optim as optim
import math


class GradientClipping:
    def __init__(self, mode="global", max_norm=1.0, norm_type=2, clip_value=None, adaptive=False, device="cuda"):
        """
        初始化梯度裁剪模块

        :param mode: 裁剪模式，'global' 为全局裁剪，'elementwise' 为逐元素裁剪
        :param max_norm: 最大裁剪范数，超过此范数时会进行裁剪
        :param norm_type: 使用的范数类型，通常为 L2 范数（norm_type=2），也可以选择 L1 范数（norm_type=1）
        :param clip_value: 逐元素裁剪模式下的裁剪值，用于裁剪梯度的上限
        :param adaptive: 是否使用自适应裁剪，随着训练的进行逐步减小裁剪阈值
        :param device: 模型所在的设备 ('cpu' 或 'cuda')
        """
        self.mode = mode
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.clip_value = clip_value
        self.adaptive = adaptive
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clip_gradients(self, model, epoch=None):
        """
        对模型的梯度进行裁剪

        :param model: 需要裁剪梯度的模型
        :param epoch: 当前训练轮次，用于自适应裁剪策略
        """
        # 如果启用了自适应裁剪，根据当前训练轮次调整裁剪阈值
        if self.adaptive and epoch is not None:
            # 自适应裁剪：随着训练轮次增加，逐步减小裁剪阈值
            self.max_norm = self.max_norm / math.sqrt(epoch + 1)

        # 确保所有参数的梯度都在正确的设备上（GPU 或 CPU）
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data.to(self.device)

        # 根据裁剪模式进行不同类型的裁剪
        if self.mode == "global":
            # 全局裁剪：裁剪所有参数的梯度范数
            parameters = [param.grad for param in model.parameters() if param.grad is not None]
            if parameters:
                # 使用 torch.nn.utils.clip_grad_norm_ 进行全局裁剪
                torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)
        elif self.mode == "elementwise":
            # 逐元素裁剪：每个参数的梯度都进行裁剪
            for param in model.parameters():
                if param.grad is not None:
                    # 对每个参数的梯度进行逐元素裁剪，限制在 [-clip_value, clip_value] 范围内
                    param.grad.data = torch.clamp(param.grad.data, -self.clip_value, self.clip_value)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # 清理未使用的临时变量
        torch.cuda.empty_cache()  # 清理 GPU 内存


if __name__ == '__main__':
    # 定义一个简单的神经网络
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 1)  # 10 输入维度，1 输出维度

        def forward(self, x):
            return self.fc(x)


    # 创建训练数据
    x = torch.randn(32, 10)  # 32 个样本，10 个特征
    y = torch.randn(32, 1)  # 32 个目标值

    # 创建模型、优化器和损失函数
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 创建梯度裁剪模块
    grad_clipper = GradientClipping(mode="global", max_norm=1.0, norm_type=2, adaptive=True, device="cuda")

    # 训练循环
    for epoch in range(1000):
        model.train()

        optimizer.zero_grad()  # 清零梯度

        # 正向传播
        output = model(x)
        loss = criterion(output, y)  # 计算损失

        # 反向传播
        loss.backward()

        # 进行梯度裁剪
        grad_clipper.clip_gradients(model, epoch)

        # 优化器更新参数
        optimizer.step()

        # 打印训练过程中的损失
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
