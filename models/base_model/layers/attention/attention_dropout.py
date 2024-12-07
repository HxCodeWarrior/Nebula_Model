# 注意力模块的Dropout机制（增强泛化能力）
import torch


class AttentionWithDropout(torch.nn.Module):
    def __init__(self, embed_size, heads, dropout_rate=0.1, adaptive_dropout=False, dynamic_schedule=False):
        super(AttentionWithDropout, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.adaptive_dropout = adaptive_dropout  # 是否使用动态Dropout
        self.dynamic_schedule = dynamic_schedule  # 是否使用动态调整策略

        # 定义线性变换层
        self.query_linear = torch.nn.Linear(embed_size, embed_size)
        self.key_linear = torch.nn.Linear(embed_size, embed_size)
        self.value_linear = torch.nn.Linear(embed_size, embed_size)

        # 最终的输出层
        self.fc_out = torch.nn.Linear(embed_size, embed_size)

        # Dropout层
        self.attention_dropout = torch.nn.Dropout(dropout_rate)  # 应用于注意力权重的Dropout
        self.output_dropout = torch.nn.Dropout(dropout_rate)  # 应用于最终输出的Dropout

    def forward(self, query, key, value, mask=None, current_loss=None):
        N = query.shape[0]  # Batch size
        value_len = value.shape[1]  # Value序列长度
        query_len = query.shape[1]  # Query序列长度

        # 分割嵌入层，得到多个头
        query = self.query_linear(query).view(N, query_len, self.heads, self.embed_size // self.heads)
        key = self.key_linear(key).view(N, value_len, self.heads, self.embed_size // self.heads)
        value = self.value_linear(value).view(N, value_len, self.heads, self.embed_size // self.heads)

        # 转置以便得到维度 [N, heads, seq_len, depth]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # 计算Scaled Dot-Product Attention
        energy = torch.matmul(query, key.permute(0, 1, 3, 2))  # Shape: [N, heads, query_len, key_len]
        energy = energy / (self.embed_size ** (1 / 2))  # 缩放

        # 如果有mask则应用
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        # 使用softmax计算注意力权重
        attention = torch.nn.functional.softmax(energy, dim=-1)  # Shape: [N, heads, query_len, key_len]

        # 动态调整Dropout率
        if self.adaptive_dropout:
            attention = self.dynamic_dropout(attention, current_loss)
        else:
            # 普通的Dropout应用
            attention = self.attention_dropout(attention)

        # 用注意力权重加权求和Value
        out = torch.matmul(attention, value)  # Shape: [N, heads, query_len, depth]

        # 合并多个头并通过线性层输出
        out = out.permute(0, 2, 1, 3).contiguous().view(N, query_len, self.heads * (self.embed_size // self.heads))
        out = self.fc_out(out)

        # 最终输出的Dropout
        out = self.output_dropout(out)

        return out

    def dynamic_dropout(self, attention, current_loss):
        """
        根据当前损失动态调整Dropout率，使用动态Dropout率来优化训练过程。
        """
        if current_loss is not None:
            # 动态调整Dropout率：如果损失下降得较慢，则降低Dropout率，反之则增加Dropout率
            new_dropout_rate = self.compute_dynamic_dropout(current_loss)
            self.attention_dropout.p = new_dropout_rate  # 更新Dropout率
        return self.attention_dropout(attention)

    def compute_dynamic_dropout(self, current_loss):
        """
        计算并返回新的Dropout率
        """
        # 假设基于损失的变化动态调整Dropout
        if self.dynamic_schedule:
            if current_loss > 0.1:
                return 0.3  # 损失较大时，增加Dropout率
            else:
                return 0.1  # 损失较小时，减少Dropout率
        return self.dropout_rate  # 默认返回原始Dropout率

