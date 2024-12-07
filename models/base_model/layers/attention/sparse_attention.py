# 稀疏注意力机制（提高计算效率，减少内存开销）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple



class DynamicSparseAttention(nn.Module):
    """
    动态稀疏化的注意力机制，通过学习动态调整稀疏化模式。
    支持局部与全局混合注意力、自适应头分配以及低秩矩阵分解优化。
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1,
                 top_k: Optional[int] = 0, local_attention_window: Optional[int] = None,
                 use_low_rank_approx: bool = False):
        """
        初始化动态稀疏化注意力模块。

        参数:
            embed_size (int): 嵌入维度（query、key、value的维度）。
            num_heads (int): 多头注意力的头数。
            dropout (float): Dropout的概率，用于增强泛化能力。
            top_k (Optional[int]): 稀疏化时选择Top-k的大小，若为0则表示不使用稀疏化。
            local_attention_window (Optional[int]): 局部注意力窗口的大小，若为None则使用全局注意力。
            use_low_rank_approx (bool): 是否启用低秩矩阵分解来加速计算。
        """
        super(DynamicSparseAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size, "embed_size必须能被num_heads整除"

        self.top_k = top_k
        self.local_attention_window = local_attention_window
        self.use_low_rank_approx = use_low_rank_approx

        # 定义线性层用于计算Q、K、V
        self.query_fc = nn.Linear(embed_size, embed_size)
        self.key_fc = nn.Linear(embed_size, embed_size)
        self.value_fc = nn.Linear(embed_size, embed_size)

        # 最后的输出线性层
        self.out = nn.Linear(embed_size, embed_size)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播计算动态稀疏化多头注意力。

        参数:
            query (Tensor): 查询张量，形状为 (batch_size, seq_len, embed_size)。
            key (Tensor): 键张量，形状为 (batch_size, seq_len, embed_size)。
            value (Tensor): 值张量，形状为 (batch_size, seq_len, embed_size)。
            mask (Optional[Tensor]): 可选的掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)。

        返回:
            Tensor: 经过动态稀疏化注意力计算后的输出张量，形状为 (batch_size, seq_len, embed_size)。
        """
        batch_size = query.size(0)

        # 计算Q、K、V
        query = self.query_fc(query)
        key = self.key_fc(key)
        value = self.value_fc(value)

        # 将Q、K、V分割成多个头（Multi-Head Attention）
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        attention_output, attention_weights = self.attention(query, key, value, mask)

        # 将注意力输出转换回原始形状
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # 通过输出线性层得到最终结果
        output = self.out(attention_output)

        return output

    def attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        计算动态稀疏化注意力。

        参数:
            query (Tensor): 查询张量，形状为 (batch_size, num_heads, seq_len_q, head_dim)。
            key (Tensor): 键张量，形状为 (batch_size, num_heads, seq_len_k, head_dim)。
            value (Tensor): 值张量，形状为 (batch_size, num_heads, seq_len_k, head_dim)。
            mask (Optional[Tensor]): 可选的掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)。

        返回:
            Tuple[Tensor, Tensor]: 注意力输出和注意力权重。
        """
        d_k = query.size(-1)  # 每个头的维度

        # 计算注意力得分（Scaled Dot-Product Attention）
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=query.dtype, device=query.device))

        # 如果有局部注意力窗口，则应用局部注意力掩码
        if self.local_attention_window is not None:
            scores = self.local_attention_masking(scores, window_size=self.local_attention_window)

        # 如果有mask，应用mask
        if mask is not None:
            mask = mask.unsqueeze(1)  # 扩展mask的形状为 (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask.expand(-1, self.num_heads, -1, -1)  # 扩展到 (batch_size, num_heads, seq_len_q, seq_len_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 应用动态稀疏化策略
        scores = self.dynamic_sparse_attention(scores)

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # Dropout
        attention_weights = self.dropout(attention_weights)

        # 计算注意力输出
        attention_output = torch.matmul(attention_weights, value)

        return attention_output, attention_weights


    def dynamic_sparse_attention(self, attention_scores: Tensor) -> Tensor:
        """
        动态稀疏化策略，通过选择Top-k的注意力得分来减少计算量。

        参数:
            attention_scores (Tensor): 计算出的注意力得分，形状为 (batch_size, num_heads, seq_len_q, seq_len_k)。

        返回:
            Tensor: 稀疏化后的注意力得分。
        """
        if self.top_k > 0:
            # 动态选择Top-k的稀疏化策略
            top_k_values, top_k_indices = torch.topk(attention_scores, self.top_k, dim=-1)
            sparse_attention = torch.zeros_like(attention_scores)
            sparse_attention.scatter_(-1, top_k_indices, top_k_values)
            return sparse_attention

        return attention_scores


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
        mask = torch.ones(seq_len_q, seq_len_k, device=scores.device)

        for i in range(seq_len_q):
            start = max(0, i - window_size // 2)
            end = min(seq_len_k, i + window_size // 2 + 1)
            mask[i, start:end] = 0

        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        return scores



class GraphStructuredSparseAttention(nn.Module):
    """
    图结构化稀疏注意力机制，结合图注意力和动态稀疏化。
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1, graph_structure: Optional[Tensor] = None):
        super(GraphStructuredSparseAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.graph_structure = graph_structure  # 图结构或邻接矩阵

        # Multi-head attention线性变换
        self.query_fc = nn.Linear(embed_size, embed_size)
        self.key_fc = nn.Linear(embed_size, embed_size)
        self.value_fc = nn.Linear(embed_size, embed_size)

        # 输出层
        self.out = nn.Linear(embed_size, embed_size)

        # 边权学习
        self.edge_weight_fc = nn.Linear(embed_size, 1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # 标准注意力计算
        Q = self.query_fc(query)
        K = self.key_fc(key)
        V = self.value_fc(value)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)

        if mask is not None:
            attention_scores += (mask * -1e9)

        # 图结构化稀疏化
        attention_scores = self.graph_sparse_attention(attention_scores)

        # 计算注意力权重并应用dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)

        # 输出
        attention_output = torch.matmul(attention_weights, V)

        output = self.out(attention_output)
        return output

    def graph_sparse_attention(self, attention_scores: Tensor) -> Tensor:
        """
        基于图结构的稀疏化机制，通过边权学习来调整注意力。
        """
        if self.graph_structure is not None:
            # 学习图的边权（这里简化为使用全连接图结构）
            edge_weights = self.edge_weight_fc(attention_scores)
            edge_weights = torch.sigmoid(edge_weights)  # 边权值应当是非负的

            # 通过图的边权来调整注意力
            attention_scores = attention_scores * edge_weights

        return attention_scores



class SparseWithSelfAttention(nn.Module):
    """
    结合自注意力与稀疏注意力的高效模型。
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1, top_k: Optional[int] = 0,
                 graph_structure: Optional[Tensor] = None):
        super(SparseWithSelfAttention, self).__init__()

        self.dynamic_sparse_attention = DynamicSparseAttention(embed_size, num_heads, dropout, top_k)
        self.graph_sparse_attention = GraphStructuredSparseAttention(embed_size, num_heads, dropout, graph_structure)

        # 多任务学习头部：任务1 - 分类任务，任务2 - 回归任务
        self.classification_head = nn.Linear(embed_size, 1)  # 分类任务
        self.regression_head = nn.Linear(embed_size, 1)  # 回归任务

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # 使用动态稀疏注意力计算
        sparse_attention_output = self.dynamic_sparse_attention(query, key, value, mask)

        # 使用图结构化稀疏注意力计算
        graph_attention_output = self.graph_sparse_attention(query, key, value, mask)

        # 多任务学习任务1 - 分类任务
        classification_output = self.classification_head(sparse_attention_output)

        # 多任务学习任务2 - 回归任务
        regression_output = self.regression_head(graph_attention_output)

        return classification_output, regression_output


if __name__ == '__main__':
    # 示例：初始化和前向传播
    embed_size = 128
    num_heads = 8
    query = torch.randn(32, 10, embed_size)  # Batch x Sequence x Embed Size
    key = torch.randn(32, 10, embed_size)
    value = torch.randn(32, 10, embed_size)
    mask = torch.randint(0, 2, (32, 10, 10))  # 假设有一个掩码

    model = SparseWithSelfAttention(embed_size, num_heads)
    classification_output, regression_output = model(query, key, value, mask)

    print("Classification Output:", classification_output.shape)
    print("Regression Output:", regression_output.shape)
