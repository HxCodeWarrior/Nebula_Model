# 带位置编码的前馈层（Positionwise Feedforward Network）
import torch
import torch.nn as nn



class PEFeedForwardNN(nn.Module):
    """
    带位置编码的前馈神经网络层
    支持不同类型的位置编码（绝对、相对、旋转、学习型）
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 d_model,
                 dropout_rate=0.1,
                 activation_fn='ReLU',
                 positional_encoding_type='absolute',
                 max_len=5000,
                 use_layer_norm=True):
        """
        :param input_size: 输入的特征数量
        :param hidden_size: 隐藏层神经元数量
        :param output_size: 输出的特征数量
        :param d_model: 位置编码的维度，通常与hidden_size相同
        :param dropout_rate: Dropout的比例
        :param activation_fn: 激活函数的类型，默认为'ReLU'
        :param positional_encoding_type: 位置编码类型 ('absolute', 'relative', 'rotary', 'learned')
        :param max_len: 最大序列长度（用于位置编码）
        :param use_layer_norm: 是否使用Layer Normalization，默认为True
        """
        super(PEFeedForwardNN, self).__init__()

        # 激活函数映射
        activation_map = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(negative_slope=0.01),
            'Swish': nn.SiLU(),
            'GELU': nn.GELU(),
            'Tanh': nn.Tanh()
        }
        self.activation_fn = activation_map.get(activation_fn, nn.ReLU())

        # 选择位置编码类型
        if positional_encoding_type == 'absolute':
            self.positional_encoding = AbsolutePositionalEncoding(d_model=d_model, max_len=max_len)
        elif positional_encoding_type == 'relative':
            self.positional_encoding = RelativePositionalEncoding(d_model=d_model, max_len=max_len)
        elif positional_encoding_type == 'rotary':
            self.positional_encoding = RotaryPositionalEncoding(d_model=d_model, max_len=max_len)
        elif positional_encoding_type == 'learned':
            self.positional_encoding = LearnedPositionalEncoding(d_model=d_model, max_len=max_len)
        else:
            raise ValueError(
                "Unknown positional encoding type. Choose from 'absolute', 'relative', 'rotary', 'learned'.")

        # 前馈网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

        # LayerNorm 层（如果需要）
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_size)
            self.layer_norm2 = nn.LayerNorm(output_size)

    def forward(self, x):
        """
        前向传播：首先加上位置编码，然后通过前馈神经网络
        :param x: 输入张量，形状为 (batch_size, seq_len, input_size)
        :return: 输出张量，形状为 (batch_size, seq_len, output_size)
        """
        # 添加位置编码
        x = self.positional_encoding(x)

        # 前馈神经网络
        x = self.fc1(x)
        x = self.activation_fn(x)  # 激活函数
        x = self.dropout(x)  # Dropout

        if self.use_layer_norm:
            x = self.layer_norm1(x)

        x = self.fc2(x)

        if self.use_layer_norm:
            x = self.layer_norm2(x)

        return x


# 示例使用
batch_size = 32
seq_len = 50
input_size = 128
hidden_size = 256
output_size = 10
d_model = 128  # 位置编码维度，通常和输入大小相同
positional_encoding_type = 'absolute'  # 选择绝对位置编码

# 创建模型
model = PEFeedForwardNN(input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        d_model=d_model,
                        positional_encoding_type=positional_encoding_type)

# 输入数据：batch_size, seq_len, input_size
x = torch.randn(batch_size, seq_len, input_size)

# 前向传播
output = model(x)

print(f"Output shape: {output.shape}")
