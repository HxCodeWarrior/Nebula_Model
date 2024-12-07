# 旋转位置编码
import torch
import math


class RotaryPositionalEncoding(torch.nn.Module):
    def __init__(self, dim, max_len=512, base=10000.0):
        """
        初始化旋转位置编码（RoPE）模块。
        :param dim: 嵌入维度
        :param max_len: 支持的最大序列长度
        :param base: 控制旋转频率的基数
        """
        super(RotaryPositionalEncoding, self).__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # 计算旋转角度和正余弦函数
        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(base) / dim))  # [dim//2]

        # 计算正余弦的位置编码
        pe = torch.zeros(max_len, dim)  # [max_len, dim]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置：sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置：cos

        # 注册位置编码为缓冲区，不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        对输入的嵌入张量应用旋转位置编码。
        :param x: 输入的嵌入张量，形状为 (batch_size, seq_len, dim)
        :return: 经过旋转位置编码后的张量
        """
        seq_len = x.size(1)  # 获取输入序列的长度
        position_encoding = self.pe[:seq_len, :].unsqueeze(0).to(x.device)  # [1, seq_len, dim]

        # 在序列上添加位置编码
        x = x + position_encoding  # 将位置编码加到输入嵌入上

        # 执行旋转操作
        return self.apply_rotary_encoding(x)

    def apply_rotary_encoding(self, x):
        """
        对输入嵌入向量应用旋转位置编码。旋转操作通过实部和虚部的变化进行。
        :param x: 输入嵌入张量，形状为 (batch_size, seq_len, dim)
        :return: 旋转后的位置编码嵌入
        """
        seq_len = x.size(1)

        # 计算位置编码：这是已经计算好的位置编码值，我们可以将其直接应用于每个元素。
        position_encoding = self.pe[:seq_len, :].unsqueeze(0).to(x.device)  # [1, seq_len, dim]

        # 直接通过view操作合并维度，避免不必要的切片
        x_real, x_imag = x.chunk(2, dim=-1)  # 更加高效的实部和虚部提取，直接分开

        # 将旋转因子应用到实部和虚部
        pos_real, pos_imag = position_encoding.chunk(2, dim=-1)  # 旋转因子也直接分为实部和虚部

        # 实部和虚部的旋转操作
        rotated_real = x_real * pos_real - x_imag * pos_imag
        rotated_imag = x_real * pos_imag + x_imag * pos_real

        # 拼接旋转后的实部和虚部
        x_rot = torch.cat([rotated_real, rotated_imag], dim=-1)

        return x_rot


# 示例使用
if __name__ == '__main__':
    batch_size = 32
    seq_len = 50
    dim = 128

    # 输入张量，形状为 (batch_size, seq_len, dim)
    x = torch.randn(batch_size, seq_len, dim)

    # 初始化旋转位置编码模块
    rope = RotaryPositionalEncoding(dim=dim, max_len=512)

    # 获取旋转位置编码后的输出
    output = rope(x)

    print("Input shape:", x.shape)
    print("Output shape after rotation encoding:", output.shape)
