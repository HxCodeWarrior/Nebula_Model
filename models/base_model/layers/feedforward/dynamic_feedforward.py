# 动态前馈网络（根据上下文调整前馈网络）
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, xavier_normal_


class DynamicFeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_fn='ReLU', dropout_rate=0.5,
                 use_batchnorm=True,
                 weight_init='he', gate_type='sigmoid', use_dynamic_activation=True, dynamic_depth=True,
                 dynamic_lr=True):
        """
        动态前馈神经网络
        :param input_size: 输入层特征数量
        :param hidden_layers: 隐藏层大小（List格式）
        :param output_size: 输出层的神经元数量
        :param activation_fn: 激活函数类型，'ReLU', 'LeakyReLU', 'Swish', 'GELU'
        :param dropout_rate: Dropout的比例
        :param use_batchnorm: 是否使用Batch Normalization
        :param weight_init: 权重初始化方法，'he' 或 'xavier'
        :param gate_type: 门控机制的类型，'sigmoid', 'tanh' 等
        :param use_dynamic_activation: 是否动态选择激活函数
        :param dynamic_depth: 是否根据训练阶段动态调整网络深度
        :param dynamic_lr: 是否使用动态学习率调整
        """
        super(DynamicFeedforwardNN, self).__init__()

        # 激活函数映射
        activation_map = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(negative_slope=0.01),
            'Swish': nn.SiLU(),
            'GELU': nn.GELU()
        }
        self.activation_fn = activation_map.get(activation_fn, nn.ReLU())

        # 门控机制
        gate_map = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'none': None
        }
        self.gate = gate_map.get(gate_type, nn.Sigmoid())

        # 网络层（动态调整网络深度）
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))  # 全连接层
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))  # Batch Normalization
            layers.append(self.activation_fn)  # 激活函数
            layers.append(nn.Dropout(p=dropout_rate))  # Dropout层
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, output_size))

        # 组合所有层
        self.network = nn.Sequential(*layers)

        # 权重初始化
        self._initialize_weights(weight_init)

        # 动态学习率和深度
        self.use_dynamic_activation = use_dynamic_activation
        self.dynamic_depth = dynamic_depth
        self.dynamic_lr = dynamic_lr

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据
        :return: 网络的输出
        """
        out = x
        for layer in self.network:
            out = layer(out)
            if isinstance(layer, nn.Linear) and self.gate is not None:
                out = out * self.gate(out)  # 门控机制：动态调整信息流
            if self.use_dynamic_activation:
                out = self._dynamic_activation(out)  # 动态激活函数选择
        return out

    def _dynamic_activation(self, x):
        """
        动态激活函数选择
        :param x: 输入数据
        :return: 激活后的输出
        """
        # 根据输入的统计数据动态选择激活函数
        if torch.mean(x) > 0.5:
            return torch.relu(x)  # 使用ReLU
        else:
            return torch.sigmoid(x)  # 使用Sigmoid

    def _initialize_weights(self, init_type='he'):
        """
        权重初始化
        :param init_type: 权重初始化类型（'he' 或 'xavier'）
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'he':
                    kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)