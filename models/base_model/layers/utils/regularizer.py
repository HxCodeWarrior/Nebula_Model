# 正则化模块（L2正则化、L1正则化等）
import torch
import torch.nn as nn
import numpy as np


# 1. L2 + L1 正则化 (ElasticNet)
class ElasticNetRegularization(nn.Module):
    def __init__(self, model, l1_weight=0.01, l2_weight=0.01):
        super(ElasticNetRegularization, self).__init__()
        self.model = model
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, x):
        # 常规前向传播
        return self.model(x)

    def regularization_loss(self):
        l1_loss = 0.0
        l2_loss = 0.0
        for name, param in self.model.named_parameters():
            if 'bias' not in name:  # 忽略偏置项
                l1_loss += torch.norm(param, p=1)
                l2_loss += torch.norm(param, p=2) ** 2
        return self.l1_weight * l1_loss + self.l2_weight * l2_loss


# 2. Dropout 正则化 (动态调整)
class DynamicLayerDropout(nn.Module):
    def __init__(self, model, layer_dropout_rates):
        super(DynamicLayerDropout, self).__init__()
        self.model = model
        self.layer_dropout_rates = layer_dropout_rates

    def forward(self, x):
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Dropout):
                # 根据层的名字或者顺序动态调整 Dropout 比例
                layer.p = self.layer_dropout_rates.get(layer_name, layer.p)
            x = layer(x)
        return x


# 3. Mixup 数据增强
class MixupRegularization:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def mixup_data(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y


# 4. Label Smoothing 正则化
class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.size = size
        self.confidence = 1.0 - smoothing
        self.register_buffer('true_dist', torch.zeros(size))

    def forward(self, x, target):
        if x.size(1) == 2:  # 针对二分类
            true_dist = target.float() * (1 - self.smoothing) + (1 - target).float() * self.smoothing
            loss = -true_dist * torch.log(torch.sigmoid(x))
        else:
            n_class = x.size(1)
            true_dist = self.true_dist.repeat(x.size(0), 1)
            true_dist.fill_(self.smoothing / (n_class - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            loss = -true_dist * torch.log_softmax(x, dim=-1)
        return loss.sum(dim=-1).mean()



# 5. Adaptive Normalization (BatchNorm 或 LayerNorm)
class AdaptiveNormalization(nn.Module):
    def __init__(self, model, use_batch_norm=True, momentum=0.1):
        super(AdaptiveNormalization, self).__init__()
        self.model = model
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(model.input_size, momentum=momentum)
        else:
            self.bn = nn.LayerNorm(model.input_size)

    def forward(self, x):
        x = self.bn(x)
        return self.model(x)


# 综合优化正则化模块
class OptimizedRegularizationModule(nn.Module):
    def __init__(self, model, dropout_rates=None, l1_weight=0.01, l2_weight=0.01, alpha=0.2, use_batch_norm=True):
        super(OptimizedRegularizationModule, self).__init__()
        self.model = model
        self.elasticnet = ElasticNetRegularization(model, l1_weight, l2_weight)
        self.mixup = MixupRegularization(alpha)
        self.batch_norm = AdaptiveNormalization(model, use_batch_norm)
        self.dropout = DynamicLayerDropout(model, dropout_rates or {})

    def forward(self, x, y=None, epoch=None):
        # Dropout 正则化
        x = self.dropout(x)

        # Mixup 数据增强
        if self.training and y is not None:
            x, y = self.mixup.mixup_data(x, y)

        # Batch Normalization
        x = self.batch_norm(x)

        # 常规前向传播
        output = self.model(x)
        return output, y

    def regularization_loss(self):
        # 合并 ElasticNet 正则化损失
        return self.elasticnet.regularization_loss()

    def update_dropout(self, epoch):
        # 动态调整 dropout 比例
        self.dropout.layer_dropout_rates = {k: v * (0.99 ** epoch) for k, v in self.dropout.layer_dropout_rates.items()}

    def early_stop(self, val_loss, val_accuracy=None):
        # 提前停止
        return self.early_stopping(val_loss, val_accuracy)



if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # 假设的简单模型
    class SimpleModel(nn.Module):
        def __init__(self, input_size, output_size):
            super(SimpleModel, self).__init__()
            self.input_size = input_size
            self.fc1 = nn.Linear(input_size, 128)  # 输入大小：input_size，输出大小：128
            self.fc2 = nn.Linear(128, output_size)  # 输入大小：128，输出大小：output_size

        def forward(self, x):
            x = F.relu(self.fc1(x))  # 激活函数
            x = self.fc2(x)  # 输出层
            return x


    # 测试代码
    def test_model():
        # 定义输入数据的尺寸 (batch_size, input_size)
        batch_size = 32
        input_size = 10
        output_size = 2  # 假设我们有两个类别

        # 创建模型
        model = SimpleModel(input_size, output_size)

        # 随机生成输入数据 (batch_size, input_size)
        inputs = torch.randn(batch_size, input_size)

        # 将数据传入模型
        outputs = model(inputs)

        # 打印模型输出
        print("Model outputs:")
        print(outputs)

        # 确认输出的形状是否匹配预期
        print("Output shape:", outputs.shape)  # 应该是 (batch_size, output_size)


    # 执行测试
    if __name__ == "__main__":
        test_model()


