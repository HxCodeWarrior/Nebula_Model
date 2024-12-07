# 标准前馈神经网络层
import torch.nn as nn
from torch.nn.init import kaiming_normal_, xavier_normal_


class FeedforwardNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_layers,
                 output_size,
                 activation_fn='ReLU',
                 dropout_rate=0.5,
                 use_batchnorm=True,
                 weight_init='he'):
        """
        标准前馈神经网络
        :param input_size: 输入层特征数量
        :param hidden_layers: 隐藏层大小（List格式）
        :param output_size: 输出层的神经元数量
        :param activation_fn: 激活函数类型，'ReLU', 'LeakyReLU', 'Swish', 'GELU'
        :param dropout_rate: Dropout的比例
        :param use_batchnorm: 是否使用Batch Normalization
        :param weight_init: 权重初始化方法，'he' 或 'xavier'
        """
        super(FeedforwardNN, self).__init__()

        # 激活函数映射
        activation_map = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(negative_slope=0.01),
            'Swish': nn.SiLU(),
            'GELU': nn.GELU()
        }
        self.activation = activation_map.get(activation_fn, nn.ReLU())

        # 网络层
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))  # 全连接层
            layers.append(self.activation)  # 激活函数
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))  # Batch Normalization
            layers.append(nn.Dropout(p=dropout_rate))  # Dropout层
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, output_size))

        # 组合所有层
        self.network = nn.Sequential(*layers)

        # 权重初始化
        self._initialize_weights(weight_init)

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据
        :return: 网络的输出
        """
        return self.network(x)

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
