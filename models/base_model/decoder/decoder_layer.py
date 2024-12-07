# 解码器层定义
import torch.nn as nn
from torch import Tensor
from typing import Optional
from models.base_model.layers.attention.multi_head_attention import MultiHeadAttention
from models.base_model.layers.feedforward.feedforward import FeedforwardNN
from models.base_model.layers.utils.dropout import Dropout
from models.base_model.layers.normalization.layer_norm import LayerNorm
from models.base_model.layers.utils.clipping import GradientClipping


class DecoderLayer(nn.Module):
    """
    优化后的解码层 (Decoder Layer)，集成了多头注意力、前馈神经网络、层归一化、Dropout技术以及梯度裁剪。
    使用自定义的 LayerNorm 进行归一化处理。

    参数：
    - embed_size (int): 输入和输出的嵌入向量维度。
    - num_heads (int): 多头注意力机制中的注意力头数。
    - ff_hidden_layers (list): 前馈神经网络的隐藏层尺寸列表。
    - ff_output_size (int): 前馈神经网络的输出尺寸。
    - dropout (float): Dropout概率，控制神经元的随机丢弃。
    - dropout_type (str): Dropout类型（如 "standard"、"spatial"等）。
    - min_p (float): 自适应Dropout的最小丢弃概率。
    - decay_factor (float): 自适应Dropout的衰减因子。
    - smooth_factor (float): 用于平滑Dropout的因子。
    - local_attention_window (Optional[int]): 本地注意力窗口大小（用于局部注意力机制）。
    - use_low_rank_approx (bool): 是否使用低秩近似。
    - gradient_clipping_config (dict): 梯度裁剪配置字典，包含模式和阈值等参数。
    """

    def __init__(
            self,
            embed_size: int,
            num_heads: int,
            ff_hidden_layers: list,
            ff_output_size: int,
            dropout: float = 0.1,
            dropout_type: str = "standard",
            min_p: float = 0.1,
            decay_factor: float = 0.99,
            smooth_factor: float = 0.05,
            local_attention_window: Optional[int] = None,
            use_low_rank_approx: bool = False,
            gradient_clipping_config: dict = None,
    ):
        super(DecoderLayer, self).__init__()

        # 初始化多头注意力机制
        self.attention1 = MultiHeadAttention(
            embed_size,
            num_heads,
            dropout=dropout,
            local_attention_window=local_attention_window,
            use_low_rank_approx=use_low_rank_approx
        )

        # 编码器-解码器的注意力机制
        self.attention2 = MultiHeadAttention(
            embed_size,
            num_heads,
            dropout=dropout,
            local_attention_window=local_attention_window,
            use_low_rank_approx=use_low_rank_approx
        )

        # 初始化前馈神经网络（Feedforward NN）
        self.feed_forward = FeedforwardNN(
            input_size=embed_size,
            hidden_layers=ff_hidden_layers,
            output_size=ff_output_size,
            activation_fn='ReLU',  # 可配置激活函数
            dropout_rate=dropout,
            use_batchnorm=True,  # 是否使用BatchNorm
            weight_init='he'  # 初始化方式，可选择 'he' 或 'xavier'
        )

        # 层归一化 (LayerNorm)，通过自定义的 LayerNorm 模块
        self.norm1 = LayerNorm(embed_size)  # 第一次归一化
        self.norm2 = LayerNorm(embed_size)  # 第二次归一化

        # Dropout模块，使用自适应Dropout
        self.dropout = Dropout(
            p=dropout,
            dropout_type=dropout_type,
            min_p=min_p,
            decay_factor=decay_factor,
            smooth_factor=smooth_factor
        )

        # 梯度裁剪模块
        self.gradient_clipping = GradientClipping(**gradient_clipping_config) if gradient_clipping_config else None

    def forward(self,
                x: Tensor,
                memory: Tensor,
                mask: Optional[Tensor] = None,
                epoch: Optional[int] = None) -> Tensor:
        """
        解码层的前向传播函数。包含自注意力机制、前馈神经网络、Dropout以及梯度裁剪。

        参数：
        - x (Tensor): 输入张量，形状为 (batch_size, seq_len, embed_size)，表示解码器的输入。
        - memory (Tensor): 编码器输出，形状为 (batch_size, seq_len, embed_size)，通常称为记忆。
        - mask (Optional[Tensor]): 可选的掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)，用于防止不必要的信息传播。
        - epoch (Optional[int]): 当前训练轮次，用于调整梯度裁剪。

        返回：
        - Tensor: 解码层的输出，形状为 (batch_size, seq_len, embed_size)。
        """
        # 第一步：多头自注意力机制 (Self-Attention)
        attn_output1 = self.attention1(x, x, x, mask)  # 使用 x 本身作为 Query, Key, Value
        x = self.norm1(x + attn_output1)  # 残差连接与归一化
        x = self.dropout(x)  # Dropout

        # 第二步：编码器-解码器注意力机制 (Encoder-Decoder Attention)
        attn_output2 = self.attention2(x, memory, memory, mask)  # 使用 memory 作为 Key, Value
        x = self.norm2(x + attn_output2)  # 残差连接与归一化
        x = self.dropout(x)  # Dropout

        # 第三步：前馈神经网络 (Feedforward Network)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)  # 残差连接与归一化
        x = self.dropout(x)  # Dropout

        # 梯度裁剪
        if self.gradient_clipping:
            self.gradient_clipping.clip_gradients(self, epoch)

        return x

    def _get_activation(self) -> nn.Module:
        """
        根据指定名称返回激活函数。

        参数：
        - activation (str): 激活函数名称（如 'relu', 'gelu', 'silu'）。

        返回：
        - 激活函数对象。
        """
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'gelu':
            return nn.GELU()
        elif self.activation == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数：{self.activation}")


if __name__ == '__main__':
    pass
