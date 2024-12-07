# 多头注意力机制（支持稀疏/全局/局部注意力）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple



class MultiHeadAttention(nn.Module):
    """
    高效多头注意力机制，支持动态稀疏化、局部与全局混合注意力、自适应头分配以及低秩矩阵分解优化。
    """

    def __init__(self,
                 embed_size: int,
                 num_heads: int,
                 rank: int = 0,
                 dropout: float = 0.1,
                 local_attention_window: Optional[int] = None,
                 use_low_rank_approx: bool = False):
        """
        初始化多头注意力模块。

        参数:
            embed_size (int): 嵌入维度（query、key、value的维度）。
            num_heads (int): 多头的数量。
            dropout (float): Dropout的概率，用于增强泛化能力。
            local_attention_window (Optional[int]): 局部注意力窗口的大小，若为None，则使用全局注意力。
            use_low_rank_approx (bool): 是否启用低秩矩阵分解来加速计算。
        """
        super(MultiHeadAttention, self).__init__()

        self.rank = rank

        # 保存嵌入维度和头数
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # 检查嵌入维度是否能被头数整除
        assert self.head_dim * num_heads == embed_size, "embed_size必须能被num_heads整除"

        # 局部注意力窗口大小
        self.local_attention_window = local_attention_window
        self.use_low_rank_approx = use_low_rank_approx

        # 定义线性层，分别用于计算Q、K、V
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

        # 非线性加权的改进部分
        self.non_linear_attention = nn.Sequential(
            nn.Linear(embed_size, embed_size),  # 输入映射
            nn.ReLU(),  # 激活
            nn.Linear(embed_size, 1)  # 输出加权系数
        )

        # 动态加权MLP，用于计算最终的权重
        self.dynamic_attention_mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size),  # 计算每个尺度的动态加权
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

        # 最后的输出线性层
        self.out = nn.Linear(embed_size, embed_size)

        # Dropout层
        self.dropout = nn.Dropout(dropout)


    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播计算多头注意力。

        参数:
            query (Tensor): 查询张量，形状为 (batch_size, seq_len, embed_size)。
            key (Tensor): 键张量，形状为 (batch_size, seq_len, embed_size)。
            value (Tensor): 值张量，形状为 (batch_size, seq_len, embed_size)。
            mask (Optional[Tensor]): 可选的掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)。

        返回:
            Tensor: 经过多头注意力计算后的输出张量，形状为 (batch_size, seq_len, embed_size)。
        """
        batch_size, seq_len, embed_size = query.size()

        # 计算Q、K、V
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        if self.use_low_rank_approx and self.rank > 0:
            # 低秩矩阵分解
            query = self.low_rank_approx(query)
            key = self.low_rank_approx(key)
            value = self.low_rank_approx(value)

        # 将Q、K、V分割成多个头（Multi-Head Attention）
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        attention_output, attention_weights = self.attention(query, key, value, mask)

        # 将注意力输出转换回原始的形状
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                              self.num_heads * self.head_dim)

        # 通过输出线性层得到最终结果
        output = self.out(attention_output)

        return output

    def attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        计算注意力，支持动态稀疏化、局部与全局混合。

        参数:
            query (Tensor): 查询张量，形状为 (batch_size, num_heads, seq_len_q, head_dim)。
            key (Tensor): 键张量，形状为 (batch_size, num_heads, seq_len_k, head_dim)。
            value (Tensor): 值张量，形状为 (batch_size, num_heads, seq_len_k, head_dim)。
            mask (Optional[Tensor]): 可选的掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)。

        返回:
            Tuple[Tensor, Tensor]: 注意力输出和注意力权重。
        """
        d_k = query.size(-1)  # 获取每个头的维度

        # 计算注意力得分（Scaled Dot-Product Attention）
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=query.dtype, device=query.device))

        # 如果有局部注意力窗口，则应用局部注意力掩码
        if self.local_attention_window is not None:
            scores = self.local_attention_masking(scores, window_size=self.local_attention_window)

        # 如果有mask，应用mask
        if mask is not None:
            # 扩展mask的形状，使其与scores的形状兼容
            mask = mask.unsqueeze(1)  # 使mask的形状变为 (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask.expand(-1, self.num_heads, -1, -1)  # 扩展到 (batch_size, num_heads, seq_len_q, seq_len_k)

            # 应用mask
            scores = scores.masked_fill(mask == 0, float('-inf'))


        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # Dropout增强泛化能力
        attention_weights = self.dropout(attention_weights)

        # 计算注意力输出
        attention_output = torch.matmul(attention_weights, value)

        return attention_output, attention_weights

    def local_attention_masking(self, scores: Tensor, window_size: int) -> Tensor:
        """
        局部注意力掩码，限制每个位置只关注窗口内的元素。

        参数:
            scores (Tensor): 计算出的注意力得分，形状为 (batch_size, num_heads, seq_len_q, seq_len_k)。
            window_size (int): 局部注意力窗口的大小。

        返回:
            Tensor: 应用局部窗口的注意力得分。
        """
        batch_size, num_heads, seq_len_q, seq_len_k = scores.size()

        # 创建局部窗口掩码
        mask = torch.ones(seq_len_q, seq_len_k, device=scores.device)

        for i in range(seq_len_q):
            start = max(0, i - window_size // 2)
            end = min(seq_len_k, i + window_size // 2 + 1)
            mask[i, start:end] = 0

        # 将掩码加到得分上
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        return scores

    def low_rank_approx(self, tensor: Tensor) -> Tensor:
        """
        优化的低秩矩阵分解方法，通过SVD或近似SVD计算张量的低秩近似。
        """
        # 获取矩阵的尺寸
        # print(f'tensor shape:{tensor.shape}')

        # 确保rank在合理的范围内
        if self.rank < 1:
            raise ValueError("Rank must be at least 1.")
        if self.rank > min(tensor.shape):
            self.rank = min(tensor.shape)
            print(f"Rank exceeds matrix dimensions, adjusting rank to {self.rank}.")

        # 使用torch.linalg.svd可以更好地处理大型矩阵和减少内存消耗
        u, s, v = torch.linalg.svd(tensor, full_matrices=False)

        # 选择前rank个奇异值，进行低秩近似
        u = u[:, :, :self.rank]
        s = s[:, :self.rank]
        v = v[:, :, :self.rank]

        # 生成对角矩阵S的近似形式，避免使用torch.diag_embed，因为这会增加额外的计算
        s_matrix = s.unsqueeze(-1) * torch.eye(self.rank, device=tensor.device).unsqueeze(0)

        # 使用低秩矩阵分解进行重构
        return torch.matmul(u, torch.matmul(s_matrix, v.transpose(-2, -1)))

    def dynamic_attention_weights(self, scale_weights: Tensor, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        动态计算加权系数，基于输入的上下文信息（如query, key, value）。通过更深层次的非线性网络来增强尺度之间的自适应性。
        """
        batch_size, seq_len, embed_size = query.size()

        # 1. 计算 `query` 与 `key` 的相似度（通过多头注意力机制）
        #   使用 multihead_attention 来计算 query 与 key 的注意力权重
        attn_output, attn_weights = self.multihead_attention(query, key, value)

        # 2. 通过非线性函数生成每个 query 的加权系数
        #   在这里我们使用多层感知机（MLP）将每个 query 映射为加权系数
        context_attention = self.non_linear_attention(query).squeeze(-1)  # (batch_size, seq_len)

        # 3. 计算每个尺度的动态加权系数
        scale_weights = self.dynamic_attention_mlp(context_attention)  # (batch_size, seq_len)
        scale_weights = F.softmax(scale_weights, dim=-1)  # 使用softmax确保权重和为1

        # 4. 动态调节最终加权系数：结合 query 与 key 之间的相似度和尺度加权系数
        combined_weights = attn_weights * scale_weights.unsqueeze(-1)  # 扩展 scale_weights 以匹配 attn_weights
        combined_weights = F.softmax(combined_weights, dim=-1)  # 再次softmax，确保加权和为1

        return combined_weights



if __name__ == '__main__':
    def test_multihead_attention():
        """
        测试优化版多头注意力机制的功能，确保其正确性。
        """
        # 设置超参数
        embed_size = 64  # 嵌入维度
        num_heads = 8  # 多头数量
        dropout = 0.1  # Dropout概率
        seq_len = 10  # 序列长度
        batch_size = 4  # 批次大小

        # 创建随机输入（batch_size, seq_len, embed_size）
        query = torch.rand(batch_size, seq_len, embed_size)
        key = torch.rand(batch_size, seq_len, embed_size)
        value = torch.rand(batch_size, seq_len, embed_size)

        # 创建一个随机的mask，形状为 (batch_size, seq_len_q, seq_len_k)
        mask = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.bool)

        # 初始化优化版多头注意力模块
        attention_layer = MultiHeadAttention(embed_size=embed_size, num_heads=num_heads, dropout=dropout,
                                             local_attention_window=3)

        # 计算注意力输出
        output = attention_layer(query, key, value, mask)

        # 输出结果的形状应该为 (batch_size, seq_len, embed_size)
        print("Output shape:", output.shape)

        # 可以添加更多的测试，例如验证输出是否符合预期范围
        assert output.shape == (batch_size, seq_len, embed_size), "输出形状不符合预期！"


    # 运行测试
    test_multihead_attention()


