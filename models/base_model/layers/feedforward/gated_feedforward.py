# 门控前馈网络（增加非线性处理能力）
import torch.nn as nn


class GatedFeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.5, activation_fn='Swish', gate_fn='sigmoid', weight_init='he', use_layernorm=True):
        """
        高质量门控前馈神经网络，包含门控机制
        :param input_size: 输入层特征数量
        :param hidden_layers: 隐藏层大小（List格式）
        :param output_size: 输出层的神经元数量
        :param dropout_rate: Dropout的比例
        :param activation_fn: 激活函数的选择，'LeakyReLU', 'ELU', 'Swish', 'ReLU'
        :param gate_fn: 门控函数的选择，'sigmoid', 'tanh'
        :param weight_init: 权重初始化方法，'he' 或 'xavier'
        :param use_layernorm: 是否使用Layer Normalization
        """
        super(GatedFeedforwardNN, self).__init__()

        # 激活函数映射
        activation_map = {
            'LeakyReLU': nn.LeakyReLU(negative_slope=0.01),
            'ELU': nn.ELU(),
            'Swish': nn.SiLU(),  # Swish激活函数
            'ReLU': nn.ReLU(),
            'GELU': nn.GELU()  # GELU激活函数
        }
        self.activation = activation_map.get(activation_fn, nn.SiLU())

        # 门控函数映射
        gate_map = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        }
        self.gate_fn = gate_map.get(gate_fn, nn.Sigmoid())

        # 定义网络层
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))  # 全连接层
            layers.append(self.activation)  # 激活函数
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_size))  # Layer Normalization
            layers.append(nn.Dropout(p=dropout_rate))  # Dropout层
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, output_size))

        # 将层组合成网络
        self.network = nn.Sequential(*layers)

        # 门控机制：每个隐藏层的门控
        self.gates = nn.ModuleList([self.gate_fn for _ in range(len(hidden_layers))])

        # 权重初始化
        self._initialize_weights(weight_init)

    def forward(self, x):
        """
        定义前向传播，包括门控机制
        :param x: 输入数据
        :return: 网络的输出
        """
        for i, layer in enumerate(self.network):
            x = layer(x)

            # 应用门控机制：通过门控调整激活值
            if i % 4 == 1:  # 每层的激活函数是每4个模块中的第2个模块
                x = x * self.gates[i // 4](x)  # 使用门控调整

        return x

    def _initialize_weights(self, init_type='he'):
        """
        初始化网络的权重
        :param init_type: 权重初始化类型（'he' 或 'xavier'）
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

