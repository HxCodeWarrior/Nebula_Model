# 多尺度注意力机制（处理多种粒度的上下文）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from .multi_head_attention import MultiHeadAttention

class MultiScaleAttention(MultiHeadAttention):
    """
    基于多尺度注意力机制，继承自低秩矩阵分解的多头注意力机制。
    本模型支持多尺度特征提取、自适应特征融合，能够动态选择不同尺度的注意力特征。
    """

    def __init__(self,
                 embed_size: int,
                 num_heads: int,
                 num_scales: int,
                 rank: int = 0,
                 dropout: float = 0.1,
                 local_attention_window: Optional[int] = None,
                 use_low_rank_approx: bool = False,
                 dynamic_attention: bool = True):
        """
        初始化多尺度注意力模块。

        参数:
            embed_size (int): 嵌入维度（query、key、value的维度）。
            num_heads (int): 多头的数量。
            num_scales (int): 特征的尺度数。
            rank (int): 低秩矩阵的秩。
            dropout (float): Dropout的概率，用于增强泛化能力。
            local_attention_window (Optional[int]): 局部注意力窗口的大小，若为None，则使用全局注意力。
            dynamic_attention (bool): 是否启用动态注意力机制（可选择性使用局部与全局注意力）。
        """
        super(MultiScaleAttention, self).__init__(embed_size, num_heads, rank, dropout)

        self.num_scales = num_scales
        self.dynamic_attention = dynamic_attention
        self.local_attention_window = local_attention_window
        self.use_low_rank_approx = use_low_rank_approx

        # 定义卷积层用于不同尺度的特征提取
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(embed_size, embed_size, kernel_size=2 * (i + 1) + 1, padding=i + 1)
            for i in range(num_scales)
        ])

        # 用于学习不同尺度的加权系数
        self.scale_attention = nn.Linear(num_scales, 1)

        # 引入非线性加权策略
        self.non_linear_attention = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 1)
        )

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播计算多尺度注意力。首先在每个尺度上提取特征，然后进行自适应特征融合，最后应用多头注意力机制。

        参数:
            query (Tensor): 查询张量，形状为 (batch_size, seq_len, embed_size)。
            key (Tensor): 键张量，形状为 (batch_size, seq_len, embed_size)。
            value (Tensor): 值张量，形状为 (batch_size, seq_len, embed_size)。
            mask (Optional[Tensor]): 可选的掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)。

        返回:
            Tensor: 经过多尺度注意力计算后的输出张量，形状为 (batch_size, seq_len, embed_size)。
        """
        # 提取多尺度特征
        scale_features = []
        for i in range(self.num_scales):
            scale_feature = self.scale_convs[i](query.transpose(1, 2))  # 转置为 (batch_size, embed_size, seq_len)
            scale_features.append(scale_feature.transpose(1, 2))  # 转回为 (batch_size, seq_len, embed_size)

        # 计算每个尺度的加权系数
        scale_weights = torch.cat(
            [self.attention(q, k, v, mask)[1] for q, k, v in zip(scale_features, scale_features, scale_features)],
            dim=-1)

        # 计算动态加权（可以是基于上下文的加权机制）
        if self.dynamic_attention:
            scale_weights = self.dynamic_attention_weights(scale_weights, query, key, value)
        else:
            scale_weights = self.scale_attention(scale_weights).squeeze(-1)

        scale_weights = F.softmax(scale_weights, dim=-1)

        # 调整 scale_weights 的形状，使其能够与 fused_features 对齐
        scale_weights = scale_weights.unsqueeze(1).unsqueeze(-1)  # 变为 (batch_size, 1, num_scales, 1)
        scale_weights = scale_weights.expand(-1, self.num_scales, -1, -1)   # 变为 (batch_size, num_scales, seq_len, 1)

        # 根据加权系数融合不同尺度特征
        fused_features = torch.stack(scale_features, dim=1)  # (batch_size, num_scales, seq_len, embed_size)
        fused_features = torch.sum(fused_features * scale_weights, dim=1)  # 加权求和

        # 重新塑形fused_features以匹配多头注意力的输入要求
        # fused_features = fused_features.transpose(0, 1).contiguous()
        print(f'fused_features shape:{fused_features.shape}')

        # 然后应用低秩优化的多头注意力机制
        return super(MultiScaleAttention, self).forward(fused_features, fused_features, fused_features, mask)

    def dynamic_attention_weights(self, scale_weights: Tensor, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        动态计算加权系数，基于输入的上下文信息（如query, key, value）。
        在此，我们引入非线性加权策略来增强尺度之间的自适应性。
        """
        # 使用非线性函数生成权重
        context_attention = self.non_linear_attention(query)
        context_attention = context_attention.squeeze(-1)  # 将最后一维压缩

        # 使用softmax来确保所有的权重和为1
        return context_attention

    def attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        重写父类的attention方法，允许对不同尺度的特征应用标准的多头注意力机制。

        参数:
            query (Tensor): 查询张量，形状为 (batch_size, num_heads, seq_len_q, head_dim)。
            key (Tensor): 键张量，形状为 (batch_size, num_heads, seq_len_k, head_dim)。
            value (Tensor): 值张量，形状为 (batch_size, num_heads, seq_len_k, head_dim)。
            mask (Optional[Tensor]): 可选的掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)。

        返回:
            Tuple[Tensor, Tensor]: 注意力输出和注意力权重。
        """
        batch_size, seq_len, embed_size = query.size()
        head_dim = embed_size // self.num_heads  # 确保 embed_size 是 num_heads 的整数倍

        # 调整为 (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1,2)
        key = key.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        # 执行标准的多头注意力计算
        return super(MultiScaleAttention, self).attention(query, key, value, mask)


if __name__ == '__main__':
    def test_multi_scale_attention():
        # 输入参数
        batch_size = 4
        seq_len = 10
        embed_size = 32
        num_heads = 4
        num_scales = 4  # 这里设置为3种尺度
        rank = 1
        dropout = 0.1

        # 创建一个 MultiScaleAttention 模型实例
        model = MultiScaleAttention(
            embed_size=embed_size,
            num_heads=num_heads,
            num_scales=num_scales,
            rank=rank,
            dropout=dropout,
            use_low_rank_approx=True
        )

        # 模拟输入的查询、键、值张量（通常是经过嵌入层处理的）
        query = torch.randn(batch_size, seq_len, embed_size)  # (batch_size, seq_len, embed_size)
        key = torch.randn(batch_size, seq_len, embed_size)  # (batch_size, seq_len, embed_size)
        value = torch.randn(batch_size, seq_len, embed_size)  # (batch_size, seq_len, embed_size)

        # 可选的mask，设为None表示不使用mask
        mask = None

        # 执行前向传播
        output = model(query, key, value, mask)

        # 输出结果
        print("Output shape:", output.shape)
        # 期望输出形状为 (batch_size, seq_len, embed_size)
        assert output.shape == (batch_size, seq_len, embed_size), f"Expected shape (4, 10, 32), but got {output.shape}"


    # 执行测试
    test_multi_scale_attention()

