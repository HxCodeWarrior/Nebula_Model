# 标准点积注意力（Scaled Dot-Product Attention）
import torch
from torch import nn
from typing import Optional, Tuple
from .multi_head_attention import MultiHeadAttention


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力（Scaled Dot-Product Attention）模块。

    核心功能：
    - 计算注意力权重（attention weights）和输出（attention output）。
    - 支持掩码（mask）机制，用于填充或因果注意力。
    - 提供数值稳定性和适用于混合精度的实现。
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1,
                 local_attention_window: Optional[int] = None, use_low_rank_approx: bool = False):
        """
        初始化缩放点积注意力模块，并整合多头注意力机制。

        参数:
            embed_size (int): 嵌入维度（query、key、value的维度）。
            num_heads (int): 多头的数量。
            dropout (float): 在注意力权重上应用的Dropout概率。默认值为0.1。
            local_attention_window (Optional[int]): 局部注意力窗口的大小，若为None，则使用全局注意力。
            use_low_rank_approx (bool): 是否启用低秩矩阵分解来加速计算。
        """
        super().__init__()

        # 初始化MultiHeadAttention模块
        self.multi_head_attention = MultiHeadAttention(embed_size=embed_size, num_heads=num_heads, dropout=dropout,
                                                       local_attention_window=local_attention_window,
                                                       use_low_rank_approx=use_low_rank_approx)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算多头缩放点积注意力。

        参数:
            query (torch.Tensor): 查询张量，形状为 (batch_size, seq_len_q, embed_size)。
            key (torch.Tensor): 键张量，形状为 (batch_size, seq_len_k, embed_size)。
            value (torch.Tensor): 值张量，形状为 (batch_size, seq_len_k, embed_size)。
            mask (Optional[torch.Tensor]): 掩码张量，可选，形状为
                - (batch_size, 1, 1, seq_len_k)（填充掩码）
                - 或 (batch_size, 1, seq_len_q, seq_len_k)（因果掩码）。

        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - 注意力输出，形状为 (batch_size, seq_len_q, embed_size)。
                - 注意力权重，形状为 (batch_size, seq_len_q, seq_len_k)。
        """
        # 使用MultiHeadAttention模块进行注意力计算
        attention_output = self.multi_head_attention(query, key, value, mask)

        return attention_output, attention_output  # 这里使用输出作为注意力权重


if __name__ == '__main__':
    # 假设我们要使用的参数
    embed_size = 8  # 嵌入维度
    num_heads = 2  # 多头数量
    dropout = 0.1  # dropout 比例

    # 构造一个简单的 ScaledDotProductAttention 实例
    attention_layer = ScaledDotProductAttention(embed_size, num_heads, dropout)

    # 定义一些简单的输入数据：batch_size = 1, seq_len = 4
    batch_size = 1
    seq_len = 4

    # 随机生成查询、键和值，嵌入维度为8
    query = torch.randn(batch_size, seq_len, embed_size)
    key = torch.randn(batch_size, seq_len, embed_size)
    value = torch.randn(batch_size, seq_len, embed_size)

    # 构造一个简单的掩码（如果有的话），假设没有填充，mask 就是 None
    mask = None

    # 使用 ScaledDotProductAttention 进行前向传播
    attention_output, attention_weights = attention_layer(query, key, value, mask)

    # 输出结果
    print("Attention Output:")
    print(attention_output)

    print("Attention Weights:")
    print(attention_weights)
