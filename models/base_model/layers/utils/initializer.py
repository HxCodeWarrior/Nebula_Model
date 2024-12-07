# 权重初始化模块（如Xavier、He初始化等）
import math
import torch.nn as nn
import torch.nn.init as init


class OptimizedWeightInitializer:
    def __init__(self):
        pass

    @staticmethod
    def initialize_weights(model, init_type='xavier', gain=1.0, bias_type='zero', activation=None):
        """
        对给定模型的所有层进行初始化，自动检测层类型和激活函数。

        :param model: 要初始化的模型
        :param init_type: 权重初始化方法 ('xavier', 'he', 'normal', 'uniform', 'orthogonal', 'lecun', 'kaiming')
        :param gain: 用于调整初始化的增益参数，通常对Xavier和He初始化有效
        :param bias_type: 偏置初始化方法 ('zero' 或 'constant')
        :param activation: 激活函数类型，用于选择适合的初始化方法（例如'relu', 'sigmoid', 'tanh'等）
        """
        for name, param in model.named_parameters():
            if 'weight' in name:
                # 自动选择初始化策略
                if activation in ['relu', 'leakyrelu']:
                    init_type = 'he'
                elif activation in ['sigmoid', 'tanh']:
                    init_type = 'xavier'
                elif activation == 'selu':
                    init_type = 'lecun'

                # 按初始化类型选择相应的初始化方法
                if init_type == 'xavier':
                    init.xavier_uniform_(param, gain=gain)
                elif init_type == 'he':
                    init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')
                elif init_type == 'normal':
                    init.normal_(param, mean=0.0, std=0.02)
                elif init_type == 'uniform':
                    init.uniform_(param, a=-gain, b=gain)
                elif init_type == 'orthogonal':
                    init.orthogonal_(param, gain=gain)
                elif init_type == 'lecun':
                    init.kaiming_uniform_(param, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
                elif init_type == 'kaiming':
                    init.kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='relu')
                else:
                    raise ValueError(f"Unsupported initialization type: {init_type}")

            # 偏置初始化
            if 'bias' in name:
                if bias_type == 'zero':
                    init.zeros_(param)
                elif bias_type == 'constant':
                    init.constant_(param, 0.1)
                else:
                    raise ValueError(f"Unsupported bias initialization type: {bias_type}")

    @staticmethod
    def initialize_layer(layer, init_type='xavier', gain=1.0, bias_type='zero', activation=None):
        """
        对单个层进行初始化，支持卷积层、线性层、批归一化层等。

        :param layer: 要初始化的层
        :param init_type: 权重初始化方法
        :param gain: 初始化增益
        :param bias_type: 偏置初始化方法
        :param activation: 激活函数类型
        """
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
            # 卷积层初始化
            OptimizedWeightInitializer.initialize_weights(layer, init_type=init_type, gain=gain, bias_type=bias_type, activation=activation)
        elif isinstance(layer, nn.Linear):
            # 全连接层初始化
            OptimizedWeightInitializer.initialize_weights(layer, init_type=init_type, gain=gain, bias_type=bias_type, activation=activation)
        elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
            # 批归一化层（BatchNorm）初始化
            init.normal_(layer.weight, mean=1.0, std=0.02)
            init.zeros_(layer.bias)
        else:
            pass  # 其他类型的层可以按需扩展

    @staticmethod
    def apply_to_model(model, init_type='xavier', gain=1.0, bias_type='zero', activation=None):
        """
        对整个模型进行初始化。

        :param model: 要初始化的模型
        :param init_type: 权重初始化方法
        :param gain: 增益参数
        :param bias_type: 偏置初始化方法
        :param activation: 激活函数类型
        """
        for module in model.modules():
            OptimizedWeightInitializer.initialize_layer(module, init_type, gain, bias_type, activation)


if __name__ == '__main__':
    import torch
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
            self.fc1 = nn.Linear(128 * 6 * 6, 1000)
            self.fc2 = nn.Linear(1000, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)  # Flatten
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    # 实例化模型并初始化
    model = SimpleCNN()

    # 使用He初始化
    initializer = OptimizedWeightInitializer()
    initializer.apply_to_model(model, init_type='he', activation='relu')

    # 打印模型结构以查看初始化情况
    print(model)

