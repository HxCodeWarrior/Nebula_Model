# 绝对位置编码
import torch
import math


class AbsolutePositionalEncoding(torch.nn.Module):
    """
    绝对位置编码类，使用正弦和余弦函数对每个位置进行编码。
    该实现支持批处理，且能够动态计算位置编码以支持变长输入。
    """

    def __init__(self, max_len, d_model):
        """
        初始化位置编码模块
        :param max_len: 允许的最大序列长度
        :param d_model: 特征维度大小（即嵌入维度）
        """
        super(AbsolutePositionalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model

        # 预计算所有位置的编码 (在初始化时)
        # 这将节省重复计算的时间，但若输入序列长度不确定，可能需要动态生成
        self.positional_encodings = self.create_positional_encoding(max_len, d_model)

    def create_positional_encoding(self, max_len, d_model):
        """
        计算位置编码，使用正弦和余弦函数
        :param max_len: 最大序列长度
        :param d_model: 特征维度大小
        :return: 位置编码矩阵 (max_len, d_model)
        """
        # 位置向量，从0到max_len-1
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # 根据 d_model 计算每个维度的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 初始化一个位置编码矩阵
        pe = torch.zeros(max_len, d_model)

        # 为偶数维度计算正弦，为奇数维度计算余弦
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算正弦部分
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算余弦部分

        # 将位置编码的维度从 (max_len, d_model) 扩展为 (1, max_len, d_model)，以支持批处理
        pe = pe.unsqueeze(0)  # 形状变为 (1, max_len, d_model)
        return pe

    def forward(self, x):
        """
        前向传播方法，将位置编码加到输入张量中。
        :param x: 输入张量，形状为 (batch_size, seq_len, d_model)
        :return: 经过位置编码后的张量
        """
        seq_len = x.size(1)  # 获取输入序列的长度
        if seq_len > self.max_len:
            raise ValueError(f"输入序列长度 {seq_len} 超过了最大支持长度 {self.max_len}")

        # 将位置编码加到输入张量中
        return x + self.positional_encodings[:, :seq_len]

if __name__ == '__main__':
    max_len = 512   # 最大序列长度
    d_model = 256   # 特征维度

    # 初始化绝对位置编码
    absolute_pos_encoder = AbsolutePositionalEncoding(max_len, d_model)

    # 创建一个输入示例 (batch_size, seq_len, d_model)
    batch_size = 32
    seq_len = 128
    x = torch.randn(batch_size, seq_len, d_model)

    # 获取带有位置编码的输入
    output = absolute_pos_encoder(x)

    # 输出形状应该为 (batch_size, seq_len, d_model)
    print(output.size)
