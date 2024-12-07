# 自适应注意力机制（任务驱动调整）
"""
TODO: 整合前馈神经网络模块，优化AdaptiveTransformerLayer
"""
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .multi_head_attention import MultiHeadAttention


class AdaptiveAttention(nn.Module):
    """
    自适应自注意力机制模块，支持局部与全局注意力计算以及动态权重调整。
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1,
                 local_attention_window: Optional[int] = None, use_low_rank_approx: bool = False):
        """
        初始化自适应注意力模块。

        参数:
            embed_size (int): 嵌入维度。
            num_heads (int): 注意力头数。
            dropout (float): Dropout概率。
            local_attention_window (Optional[int]): 局部注意力窗口的大小。
            use_low_rank_approx (bool): 是否启用低秩矩阵分解来加速计算。
        """
        super(AdaptiveAttention, self).__init__()

        # 初始化MultiHeadAttention模块
        self.multi_head_attention = MultiHeadAttention(embed_size=embed_size, num_heads=num_heads, dropout=dropout,
                                                       local_attention_window=local_attention_window,
                                                       use_low_rank_approx=use_low_rank_approx)

        # 输出层
        self.out = nn.Linear(embed_size, embed_size)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播，计算自适应注意力。

        参数:
            query (Tensor): 查询张量，形状为 (batch_size, seq_len, embed_size)。
            key (Tensor): 键张量，形状为 (batch_size, seq_len, embed_size)。
            value (Tensor): 值张量，形状为 (batch_size, seq_len, embed_size)。
            mask (Optional[Tensor]): 掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)。

        返回:
            Tensor: 注意力输出，形状为 (batch_size, seq_len, embed_size)。
        """
        attention_output = self.multi_head_attention(query, key, value, mask)

        # 输出层
        output = self.out(attention_output)
        return output


class DynamicWeightAdjustment(nn.Module):
    """
    动态权重调整机制，用于动态学习如何调整注意力权重。
    通过多层感知机和自注意力机制，增强模型的动态调整能力。
    """

    def __init__(self, embed_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        """
        初始化动态权重调整模块。

        参数:
            embed_size (int): 嵌入维度。
            hidden_size (int): MLP中的隐藏层维度。
            num_layers (int): MLP的层数。
            dropout (float): Dropout的概率，用于提高泛化能力。
        """
        super(DynamicWeightAdjustment, self).__init__()

        # 多层感知机 (MLP) 网络，用于增强非线性表示能力
        layers = []
        input_size = embed_size
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())  # 使用ReLU激活函数增加非线性
            layers.append(nn.Dropout(dropout))  # 添加Dropout，防止过拟合
            input_size = hidden_size

        # 最后一层将输出调整为1维，作为权重调整因子
        self.mlp = nn.Sequential(*layers)
        self.final_layer = nn.Linear(hidden_size, 1)

    def forward(self, attention_weights: Tensor, query: Tensor, key: Optional[Tensor] = None,
                value: Optional[Tensor] = None) -> Tensor:
        """
        根据输入动态调整注意力权重。

        参数:
            attention_weights (Tensor): 当前的注意力权重。
            query (Tensor): 输入的查询张量。
            key (Optional[Tensor]): 键张量，用于增强上下文信息。
            value (Optional[Tensor]): 值张量，用于增强上下文信息。

        返回:
            Tensor: 调整后的注意力权重。
        """
        # 将query与key、value的上下文信息融合
        context = query
        if key is not None:
            context += key  # 通过加法融合查询和键的信息
        if value is not None:
            context += value  # 进一步加和值的信息

        # 通过MLP调整权重因子
        weight_factor = self.mlp(context)
        weight_factor = self.final_layer(weight_factor)  # 输出一个单一的权重因子

        # 调整注意力权重
        adjusted_weights = attention_weights * weight_factor  # 按照因子调整原始注意力权重

        # 返回调整后的注意力权重
        return adjusted_weights


class AdaptiveAttentionWithEfficiency(nn.Module):
    """
    结合自适应注意力和计算效率优化的注意力机制模块。
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1,
                 local_attention_window: Optional[int] = None):
        """
        初始化自适应注意力与效率优化模块。

        参数:
            embed_size (int): 嵌入维度。
            num_heads (int): 注意力头数。
            dropout (float): Dropout概率。
            local_attention_window (Optional[int]): 局部注意力窗口的大小。
        """
        super(AdaptiveAttentionWithEfficiency, self).__init__()

        self.attention = AdaptiveAttention(embed_size, num_heads, dropout, local_attention_window)
        self.dynamic_adjustment = DynamicWeightAdjustment(embed_size)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # 计算原始注意力输出
        attention_output = self.attention(query, key, value, mask)

        # 计算动态调整的注意力权重
        adjusted_weights = self.dynamic_adjustment(attention_output, query)

        # 返回调整后的输出
        return attention_output * adjusted_weights


class AdaptiveTransformerLayer(nn.Module):
    """
    包含自适应注意力机制和前馈网络的Transformer层。
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1,
                 local_attention_window: Optional[int] = None):
        """
        初始化自适应Transformer层。

        参数:
            embed_size (int): 嵌入维度。
            num_heads (int): 注意力头数。
            dropout (float): Dropout概率。
            local_attention_window (Optional[int]): 局部注意力窗口的大小。
        """
        super(AdaptiveTransformerLayer, self).__init__()

        self.attention = AdaptiveAttentionWithEfficiency(embed_size, num_heads, dropout, local_attention_window)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播，通过自适应注意力和前馈网络进行计算。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, seq_len, embed_size)。
            mask (Optional[Tensor]): 掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)。

        返回:
            Tensor: 输出张量，形状为 (batch_size, seq_len, embed_size)。
        """
        # 自适应注意力层
        attention_output = self.attention(x, x, x, mask)
        attention_output = self.layer_norm1(x + self.dropout(attention_output))

        # 前馈网络层
        ff_output = self.feed_forward(attention_output)
        output = self.layer_norm2(attention_output + self.dropout(ff_output))

        return output


class AdaptiveTransformer(nn.Module):
    """
    自适应Transformer模型，包含多个自适应Transformer层。
    """

    def __init__(self, num_layers: int, embed_size: int, num_heads: int, dropout: float = 0.1,
                 local_attention_window: Optional[int] = None):
        """
        初始化自适应Transformer模型。

        参数:
            num_layers (int): Transformer层的数量。
            embed_size (int): 嵌入维度。
            num_heads (int): 注意力头数。
            dropout (float): Dropout概率。
            local_attention_window (Optional[int]): 局部注意力窗口的大小。
        """
        super(AdaptiveTransformer, self).__init__()

        self.layers = nn.ModuleList([
            AdaptiveTransformerLayer(embed_size, num_heads, dropout, local_attention_window)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播，经过多个自适应Transformer层进行处理。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, seq_len, embed_size)。
            mask (Optional[Tensor]): 掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)。

        返回:
            Tensor: 输出张量，形状为 (batch_size, seq_len, embed_size)。
        """
        for layer in self.layers:
            x = layer(x, mask)

        return x
