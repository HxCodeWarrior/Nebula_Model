import torch


class AttentionMaskGenerator:
    def __init__(self, seq_len, pad_token=0):
        """
        初始化AttentionMaskGenerator类，生成自注意力掩码和填充掩码。

        :param seq_len: 序列的长度
        :param pad_token: 填充符号的标记，默认值为0
        """
        self.seq_len = seq_len
        self.pad_token = pad_token

    def generate_decoder_mask(self):
        """
        生成解码器的自注意力掩码（self-attention mask），
        用于防止模型在解码时查看未来的单词。

        :return: 形状为 (seq_len, seq_len) 的自注意力掩码矩阵
        """
        # 使用 torch.triu() 创建上三角矩阵，表示禁止关注未来的单词
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1)

        # 将未来位置的掩码值设置为负无穷，表示这些位置不能被关注
        mask = mask * float('-inf')  # 未来的单词置为负无穷
        mask = mask + torch.ones(self.seq_len, self.seq_len)  # 当前单词位置和之前的单词置为0

        return mask

    def generate_padding_mask(self, input_seq):
        """
        生成填充掩码，用于指示哪些位置是填充，哪些位置是有效的。

        :param input_seq: 输入序列，形状为 (batch_size, seq_len)
        :return: 形状为 (batch_size, 1, 1, seq_len) 的填充掩码
        """
        # 根据输入序列生成布尔掩码，True 表示有效的词，False 表示填充
        mask = (input_seq != self.pad_token).unsqueeze(1).unsqueeze(2)  # 扩展维度至 (batch_size, 1, 1, seq_len)
        return mask

    def combine_masks(self, self_attention_mask, padding_mask):
        """
        结合自注意力掩码和填充掩码，生成最终的掩码。

        :param self_attention_mask: 解码器自注意力掩码，形状为 (seq_len, seq_len)
        :param padding_mask: 填充掩码，形状为 (batch_size, 1, 1, seq_len)
        :return: 结合后的掩码，形状为 (batch_size, seq_len, seq_len)
        """
        batch_size = padding_mask.size(0)
        seq_len = self_attention_mask.size(0)

        # 扩展 padding_mask 以匹配 self_attention_mask 的形状
        padding_mask = padding_mask.expand(batch_size, 1, seq_len, seq_len)  # 扩展到(batch_size, 1, seq_len, seq_len)

        # 将 self_attention_mask 扩展到(batch_size, seq_len, seq_len)
        combined_mask = self_attention_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)

        # 组合自注意力掩码和填充掩码
        combined_mask = combined_mask + padding_mask.squeeze(1).squeeze(1)  # 将填充掩码加到自注意力掩码
        return combined_mask


# 测试 AttentionMaskGenerator 类
if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 2
    seq_len = 5
    input_seq = torch.tensor([[1, 2, 0, 4, 5], [6, 0, 0, 7, 8]])  # batch_size = 2, seq_len = 5

    # 创建 AttentionMaskGenerator 实例
    mask_generator = AttentionMaskGenerator(seq_len)

    # 生成自注意力掩码
    self_attention_mask = mask_generator.generate_decoder_mask()
    print("Self-attention Mask:\n", self_attention_mask)

    # 生成填充掩码
    padding_mask = mask_generator.generate_padding_mask(input_seq)
    print("\nPadding Mask:\n", padding_mask)

    # 结合掩码
    combined_mask = mask_generator.combine_masks(self_attention_mask, padding_mask)
    print("\nCombined Mask:\n", combined_mask)
