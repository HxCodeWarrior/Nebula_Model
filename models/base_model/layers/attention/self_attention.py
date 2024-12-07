# 自注意力机制（Self-Attention）
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .scaled_dot_product_attention import ScaledDotProductAttention  # 导入点积注意力模块


class SelfAttention(nn.Module):
    """
    自注意力机制（Self-Attention）模块，支持多头注意力、掩码机制、Dropout以及混合精度训练。
    """
    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1,
                 local_attention_window: Optional[int] = None, use_low_rank_approx: bool = False):
        """
        初始化自注意力模块。

        参数:
            embed_size (int): 嵌入维度，即查询（Query）、键（Key）和值（Value）的维度。
            num_heads (int): 多头注意力中的头数。
            dropout (float): 注意力权重上的Dropout概率，默认值为0.1。
            local_attention_window (Optional[int]): 局部注意力窗口的大小，若为None，则使用全局注意力。
            use_low_rank_approx (bool): 是否启用低秩矩阵分解来加速计算。
        """
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # 确保embed_size能够被num_heads整除
        assert self.head_dim * num_heads == embed_size, "embed_size必须能够被num_heads整除！"

        # 初始化点积注意力（Scaled Dot-Product Attention）模块
        self.scaled_dot_product_attention = ScaledDotProductAttention(
            embed_size=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            local_attention_window=local_attention_window,
            use_low_rank_approx=use_low_rank_approx
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.out = nn.Linear(embed_size, embed_size)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播，计算自注意力。

        参数:
            query (torch.Tensor): 查询张量，形状为 (batch_size, seq_len, embed_size)。
            key (torch.Tensor): 键张量，形状为 (batch_size, seq_len, embed_size)。
            value (torch.Tensor): 值张量，形状为 (batch_size, seq_len, embed_size)。
            mask (Optional[torch.Tensor]): 掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)。

        返回:
            torch.Tensor: 注意力输出，形状为 (batch_size, seq_len, embed_size)。
        """
        # 使用点积注意力进行计算
        attention_output, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)

        # 通过Dropout层
        attention_output = self.dropout(attention_output)

        # 输出层
        output = self.out(attention_output)

        return output


if __name__ == '__main__':
    def test_self_attention():
        batch_size = 2
        seq_len = 5
        embed_size = 8
        num_heads = 2
        mask = torch.randint(0, 2, (batch_size, seq_len, seq_len))  # 例如填充掩码

        # 随机生成输入数据
        query = torch.rand(batch_size, seq_len, embed_size)
        key = torch.rand(batch_size, seq_len, embed_size)
        value = torch.rand(batch_size, seq_len, embed_size)

        attention = SelfAttention(embed_size, num_heads)
        output = attention(query, key, value, mask)

        print("Output shape:", output.shape)  # 预期输出形状：(batch_size, seq_len, embed_size)
        print(f"Output:\n{output}")


    test_self_attention()

