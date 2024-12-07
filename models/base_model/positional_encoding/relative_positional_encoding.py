# 相对位置编码
import torch
import torch.nn as nn
import math

class RelativePositionalEncoding(nn.Module):
    """
    高级相对位置编码模块 (Relative Positional Encoding, RPE)。
    本模块通过结合傅里叶特征嵌入和学习型偏置矩阵，提升相对位置信息对自注意力机制的表达能力。
    """
    def __init__(self, d_model, num_heads, max_len=512):
        """
        初始化高级相对位置编码模块。

        参数:
        - d_model: 模型的嵌入维度 (Embedding Dimension)。
        - num_heads: 多头注意力的头数 (Number of Attention Heads)。
        - max_len: 最大序列长度 (Maximum Sequence Length)。
        """
        super(RelativePositionalEncoding, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须可以被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 每个注意力头的维度
        self.max_len = max_len

        # 初始化基于傅里叶特征的正余弦位置嵌入
        self.freqs = self._generate_sinusoidal_embeddings(max_len, self.head_dim)

        # 学习型相对位置偏置 (Learnable Relative Position Bias)
        self.relative_bias = nn.Parameter(
            torch.zeros(max_len, max_len, num_heads)
        )
        nn.init.xavier_uniform_(self.relative_bias)  # 使用 Xavier 初始化

    def forward(self, query, key):
        """
        计算带有相对位置编码的注意力分数。

        参数:
        - query: 查询张量 (Query Tensor)，形状为 (batch_size, seq_len, num_heads, head_dim)。
        - key: 键张量 (Key Tensor)，形状为 (batch_size, seq_len, num_heads, head_dim)。

        返回:
        - 带有相对位置偏置的注意力分数 (Attention Scores)，形状为 (batch_size, num_heads, seq_len, seq_len)。
        """
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert seq_len <= self.max_len, "序列长度超过了预设的最大长度"

        # 生成相对位置矩阵
        rel_positions = self._get_relative_positions(seq_len)

        # 计算查询与键的点积注意力分数
        attention_scores = torch.einsum('bqhd,bkhd->bhqk', query, key)

        # 添加学习型相对位置偏置
        relative_encoding = self.relative_bias[:seq_len, :seq_len, :].permute(2, 0, 1)  # (num_heads, seq_len, seq_len)
        attention_scores += relative_encoding.unsqueeze(0)  # 添加到注意力分数中

        # 将傅里叶特征嵌入扩展到多头维度
        freqs = self.freqs[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        freqs = freqs.repeat(batch_size, num_heads, 1, 1)  # (batch_size, num_heads, seq_len, head_dim)

        # 添加到查询和键
        query = query + freqs
        key = key + freqs

        return attention_scores

    def _generate_sinusoidal_embeddings(self, max_len, dim):
        """
        基于傅里叶特征生成正余弦位置嵌入。

        参数:
        - max_len: 最大序列长度。
        - dim: 嵌入维度。

        返回:
        - 一个形状为 (max_len, dim) 的张量，表示正余弦嵌入。
        """
        position = torch.arange(0, max_len).unsqueeze(1).float()  # 位置索引，形状 (max_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))  # 缩放因子

        embeddings = torch.zeros(max_len, dim)  # 初始化嵌入矩阵
        embeddings[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用正弦
        embeddings[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用余弦
        return embeddings

    def _get_relative_positions(self, seq_len):
        """
        计算相对位置矩阵。

        参数:
        - seq_len: 当前序列长度。

        返回:
        - 一个形状为 (seq_len, seq_len) 的张量，表示序列中各位置的相对位置。
        """
        indices = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)  # 相对位置索引
        return indices

# 示例代码
if __name__ == "__main__":
    batch_size = 2
    seq_len = 128
    d_model = 256
    num_heads = 8

    # 创建查询 (Query) 和键 (Key) 的输入张量
    query = torch.randn(batch_size, seq_len, num_heads, d_model // num_heads)
    key = torch.randn(batch_size, seq_len, num_heads, d_model // num_heads)

    # 实例化高级相对位置编码模块
    relative_pos_encoder = RelativePositionalEncoding(d_model, num_heads, max_len=512)

    # 计算带有相对位置编码的注意力分数
    attention_scores = relative_pos_encoder(query, key)
    print(attention_scores.shape)  # 输出形状应为 (batch_size, num_heads, seq_len, seq_len)


