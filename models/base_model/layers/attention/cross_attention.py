# 跨注意力机制（Encoder-Decoder Attention）
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .multi_head_attention import MultiHeadAttention

class CrossAttention(nn.Module):
    """
    跨层注意力机制，集成了多头注意力，支持局部与全局混合注意力，并优化了低秩矩阵分解。
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1,
                 local_attention_window: Optional[int] = None, use_low_rank_approx: bool = False):
        """
        初始化跨层注意力模块。

        参数:
            embed_size (int): 嵌入维度（query、key、value的维度）。
            num_heads (int): 多头的数量。
            dropout (float): Dropout的概率，用于增强泛化能力。
            local_attention_window (Optional[int]): 局部注意力窗口的大小，若为None，则使用全局注意力。
            use_low_rank_approx (bool): 是否启用低秩矩阵分解来加速计算。
        """
        super(CrossAttention, self).__init__()

        # 跨层注意力机制通过多头注意力进行实现
        self.multi_head_attention = MultiHeadAttention(embed_size=embed_size, num_heads=num_heads, dropout=dropout,
                                                       local_attention_window=local_attention_window,
                                                       use_low_rank_approx=use_low_rank_approx)

        # 引入层归一化（Layer Normalization）提升训练稳定性
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播计算跨层注意力。

        参数:
            query (Tensor): 查询张量，形状为 (batch_size, seq_len_q, embed_size)。
            key (Tensor): 键张量，形状为 (batch_size, seq_len_k, embed_size)。
            value (Tensor): 值张量，形状为 (batch_size, seq_len_k, embed_size)。
            mask (Optional[Tensor]): 可选的掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)。

        返回:
            Tensor: 经过跨层注意力计算后的输出张量，形状为 (batch_size, seq_len_q, embed_size)。
        """

        # 计算多头注意力
        attention_output = self.multi_head_attention(query, key, value, mask)

        # 输出结果通过Layer Normalization增强训练稳定性
        attention_output = self.layer_norm(attention_output + query)  # 残差连接

        # 最后的输出线性层
        output = self.out(attention_output)

        return output


if __name__ == '__main__':
    pass