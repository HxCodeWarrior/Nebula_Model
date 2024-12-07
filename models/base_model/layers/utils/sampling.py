import torch
import torch.nn.functional as F
from .temperature import TemperatureAdjuster


class TopPSampling:
    def __init__(
            self,
            top_p: float = 0.9,
            temperature: float = 1.0,
            dynamic_temp: bool = False,
            repetition_penalty: float = 1.0,
            use_dynamic_temperature: bool = False,
            temperature_adjuster: TemperatureAdjuster = None
    ):
        """
        初始化高质量的Top-p采样策略。

        :param top_p: Top-p采样的p值。通常设定为0.9，表示保留累积概率大于等于0.9的候选词。
        :param temperature: 温度缩放参数，调节概率分布的平滑度。温度越高，生成的结果越随机。
        :param dynamic_temp: 是否启用动态温度调整。通常在生成过程中可以逐渐增加温度。
        :param repetition_penalty: 惩罚机制，避免生成重复的token。默认为1.0，不使用惩罚。
        :param use_dynamic_temperature: 是否使用动态温度调整器。
        :param temperature_adjuster: 传入温度调整器实例以进行动态温度调整。
        """
        self.top_p = top_p
        self.temperature = temperature
        self.dynamic_temp = dynamic_temp
        self.repetition_penalty = repetition_penalty

        # 使用传入的TemperatureAdjuster进行动态温度调整
        self.use_dynamic_temperature = use_dynamic_temperature
        self.temperature_adjuster = temperature_adjuster

    def apply_repetition_penalty(self,
                                 logits: torch.Tensor,
                                 past_tokens):
        """
        应用重复惩罚，降低已经生成的token的概率。

        :param logits: 未经过softmax的模型输出。
        :param past_tokens: 已经生成的tokens的索引。
        :return: 应用重复惩罚后的logits。
        """
        if self.repetition_penalty != 1.0:
            for token in past_tokens:
                logits[:, token] /= self.repetition_penalty  # 对已经生成的tokens应用惩罚
        return logits

    def adjust_temperature(self,
                           logits: torch.Tensor,
                           step: int = 0,
                           content_quality: float = 1.0,
                           perplexity: float = 1.0):
        """
        根据动态温度策略调整logits。

        :param logits: 模型的未缩放的logits。
        :param step: 当前生成的步数，用于调整温度。
        :param content_quality: 内容质量，用于温度平滑调整。
        :param perplexity: 当前生成内容的困惑度评分.
        :return: 调整后的logits。
        """
        if self.use_dynamic_temperature and self.temperature_adjuster:
            # 使用动态温度调整器
            logits = self.temperature_adjuster.adjust_logits_with_temperature(logits, step, content_quality, perplexity)
        else:
            # 固定温度
            logits = logits / self.temperature
        return logits

    def sample(self,
            logits: torch.Tensor,
            past_tokens: torch.Tensor = None,
            step: int = 0,
            content_quality: float = 1.0,
            perplexity: float = 1.0
    ) -> torch.Tensor:
        """
        从给定的logits中采样一个token，使用高质量的Top-p采样策略。

        :param logits: 形状为[batch_size, vocab_size]的logits张量。
        :param past_tokens: 已生成的tokens索引，用于避免重复生成。
        :param step: 当前生成的步数，用于动态温度调整。
        :param content_quality: 内容质量，用于温度平滑调整。
        :param perplexity: 当前生成内容的困惑度评分.
        :return: 采样得到的token索引。
        """
        # 应用温度调整
        logits = self.adjust_temperature(logits, step, content_quality, perplexity)

        # 应用重复惩罚机制
        if past_tokens is not None and past_tokens.size(1) > 0:
            logits = self.apply_repetition_penalty(logits, past_tokens)

        # 计算概率分布
        probs = F.softmax(logits, dim=-1)

        # 对每个样本进行Top-p采样
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # 计算累计概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 找到累计概率超过top_p的第一个位置
        cutoff_index = (cumulative_probs > self.top_p).max(dim=-1)[1]  # 直接获取超过top_p的第一位置)

        # 扩展cutoff_index以进行gather操作
        cutoff_index_expanded = cutoff_index.unsqueeze(1).expand(-1, sorted_probs.size(1))

        # 为每个样本截断到超过p值的词汇
        sorted_probs = torch.gather(sorted_probs, 1, cutoff_index_expanded)
        sorted_indices = torch.gather(sorted_indices, 1, cutoff_index_expanded)

        # 归一化概率
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        # 根据概率进行采样
        sampled_token = torch.multinomial(sorted_probs, 1).squeeze(-1)

        # 返回对应的token索引
        return sorted_indices.gather(1, sampled_token.unsqueeze(1)).squeeze(-1)



class TopKSampling:
    def __init__(
            self,
            top_k: int = 50,
            temperature: float = 1.0,
            dynamic_temp: bool = False,
            repetition_penalty: float = 1.0,
            use_dynamic_temperature: bool = False,
            temperature_adjuster: TemperatureAdjuster = None
    ):
        """
        初始化高质量的Top-k采样策略。

        :param top_k: Top-k采样的k值。k=0表示不使用Top-k采样。
        :param temperature: 温度缩放参数，调节概率分布的平滑度。
        :param dynamic_temp: 是否启用动态温度调整，根据生成步数调整温度。
        :param repetition_penalty: 惩罚机制，降低已生成token的概率，避免重复生成。
        """
        self.top_k = top_k
        self.temperature = temperature
        self.dynamic_temp = dynamic_temp
        self.repetition_penalty = repetition_penalty
        self.use_dynamic_temperature = use_dynamic_temperature
        self.temperature_adjuster = temperature_adjuster

    def apply_repetition_penalty(self,
                                 logits,
                                 past_tokens):
        """
        应用重复惩罚，降低已经生成的token的概率。

        :param logits: 未经过softmax的模型输出。
        :param past_tokens: 已经生成的tokens的索引。
        :return: 应用重复惩罚后的logits。
        """
        if self.repetition_penalty != 1.0:
            for token in past_tokens:
                logits[:, token] /= self.repetition_penalty  # 对已生成的tokens应用惩罚
        return logits

    def adjust_temperature(self,
                           logits: torch.Tensor,
                           step: int = 0,
                           content_quality: float = 1.0,
                           perplexity: float = 1.0):
        """
        根据动态温度策略调整logits。

        :param logits: 模型的未缩放的logits。
        :param step: 当前生成的步数，用于调整温度。
        :param content_quality: 内容质量，用于温度平滑调整。
        :param perplexity: 当前生成内容的困惑度评分.
        :return: 调整后的logits。
        """
        if self.use_dynamic_temperature and self.temperature_adjuster:
            logits = self.temperature_adjuster.adjust_logits_with_temperature(logits, step, content_quality, perplexity)
        else:
            logits = logits / self.temperature

        return logits

    def sample(self,
               logits: torch.Tensor,
               past_tokens: torch.Tensor = None,
               step: int = 0,
               content_quality: float = 1.0,
               perplexity: float = 1.0):
        """
        从给定的logits中采样一个token，使用高质量的Top-k采样策略。

        :param logits: 形状为[batch_size, vocab_size]的logits张量。
        :param past_tokens: 已生成的tokens索引，用于避免重复生成。
        :param step: 当前生成的步数，用于动态温度调整。
        :param content_quality: 内容质量，用于温度平滑调整。
        :param perplexity: 当前生成内容的困惑度评分.
        :return: 采样得到的token索引。
        """
        # 应用温度调整
        logits = self.adjust_temperature(logits, step, content_quality, perplexity)

        # 应用重复惩罚机制
        if past_tokens is not None and past_tokens.size(1) > 0:
            logits = self.apply_repetition_penalty(logits, past_tokens)

        # 计算概率分布
        probs = F.softmax(logits, dim=-1)

        # 对每个样本进行Top-k采样
        if self.top_k > 0:
            # 获取Top-k最大概率的索引和值
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

            # 对Top-k的概率进行归一化
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            # 从Top-k中采样
            sampled_token = torch.multinomial(top_k_probs, 1).squeeze(-1)

            # 获取采样的词汇的索引
            sampled_token_index = torch.gather(top_k_indices, -1, sampled_token.unsqueeze(-1))

            return sampled_token_index.squeeze(-1)
        else:
            # 如果Top-k为0，直接从整个词汇表中采样
            return torch.multinomial(probs, 1).squeeze(-1)



class MixSampling(TopPSampling, TopKSampling):
    def __init__(self,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 temperature: float = 1.0,
                 dynamic_temp: bool = False,
                 repetition_penalty: float = 1.0,
                 use_dynamic_temperature: bool = False,
                 temperature_adjuster: TemperatureAdjuster = None,
                 sampling_order: str = "top_p_first"
                 ):
        """
        初始化混合采样策略，结合Top-p和Top-k采样策略。

        :param top_p: Top-p采样的p值。
        :param top_k: Top-k采样的k值。
        :param temperature: 温度缩放参数，调节概率分布的平滑度。
        :param dynamic_temp: 是否启用动态温度调整。
        :param repetition_penalty: 惩罚机制，避免重复生成。
        :param use_dynamic_temperature: 是否使用动态温度调整器。
        :param temperature_adjuster: 温度调整器实例。
        :param sampling_order: 采样顺序，"top_p_first" 表示先Top-p再Top-k，"top_k_first" 表示先Top-k再Top-p。
        """
        super().__init__(top_p, temperature, dynamic_temp, repetition_penalty, use_dynamic_temperature,
                         temperature_adjuster)
        self.top_p_sampler = TopPSampling(
            top_p=top_p,
            temperature=temperature,
            dynamic_temp=dynamic_temp,
            repetition_penalty=repetition_penalty,
            use_dynamic_temperature=use_dynamic_temperature,
            temperature_adjuster=temperature_adjuster
        )
        self.top_k_sampler = TopKSampling(
            top_k=top_k,
            temperature=temperature,
            dynamic_temp=dynamic_temp,
            repetition_penalty=repetition_penalty,
            use_dynamic_temperature=use_dynamic_temperature,
            temperature_adjuster=temperature_adjuster
        )

        self.sampling_order = sampling_order

    def sample(self, logits: torch.Tensor, past_tokens: torch.Tensor = None, step: int = 0, content_quality: float = 1.0, perplexity: float = 1.0):
        """
        混合采样方法，按照指定顺序执行Top-p和Top-k采样。

        :param logits: 模型的logits。
        :param past_tokens: 已生成的tokens索引，用于避免重复生成。
        :param step: 当前生成的步数，用于动态温度调整。
        :param content_quality: 内容质量，用于温度平滑调整。
        :param perplexity: 当前生成内容的困惑度评分。
        :return: 采样得到的token索引。
        """
        # logits = self.adjust_temperature(logits, step, content_quality, perplexity)
        # logits = self.apply_repetition_penalty(logits, past_tokens)

        if self.sampling_order == "top_p_first":
            # Step 1: 使用Top-p采样限制候选词汇范围
            top_p_token = self.top_p_sampler.sample(
                logits,
                past_tokens,
                step,
                content_quality,
                perplexity)

            # Step 2: 使用Top-k采样进一步从Top-p采样的候选词汇中进行选择
            # 获取Top-p后的logits，重新计算后续采样
            logits = logits.gather(1, top_p_token.unsqueeze(-1))  # 使用Top-p采样结果作为输入

            sampled_token = self.top_k_sampler.sample(
                logits,
                past_tokens,
                step,
                content_quality,
                perplexity)

            return sampled_token
        elif self.sampling_order == "top_k_first":
            # Step 1: 使用Top-k采样限制候选词汇范围
            top_k_token = self.top_k_sampler.sample(
                logits,
                past_tokens,
                step,
                content_quality,
                perplexity)

            # Step 2: 使用Top-p采样进一步从Top-k采样的候选词汇中进行选择
            # 获取Top-k后的logits，重新计算后续采样
            logits = logits.gather(1, top_k_token.unsqueeze(-1))  # 使用Top-k采样结果作为输入

            sampled_token = self.top_p_sampler.sample(
                logits,
                past_tokens,
                step,
                content_quality,
                perplexity)

            return sampled_token



if __name__ == '__main__':
    # 假设我们有一个批次大小为2，词汇表大小为10的模型输出logits
    batch_size = 2
    vocab_size = 10
    logits = torch.randn(batch_size, vocab_size)  # 随机生成的logits

    # 假设没有过去的tokens
    past_tokens = torch.empty((batch_size, 0))  # 空的past_tokens张量

    # 当前步数（生成的第几步），内容质量评分和困惑度评分
    step = 5
    content_quality = 0.8
    perplexity = 1.2


    # 创建一个简单的温度调整器（如果需要动态调整温度）
    class TemperatureAdjuster:
        def adjust_logits_with_temperature(self, logits, step, content_quality, perplexity):
            # 简单的温度调整（此处为示例，实际情况可以更复杂）
            return logits / (content_quality * step + 1)


    temperature_adjuster = TemperatureAdjuster()

    # 创建Top-p采样策略实例
    top_p_sampler = TopPSampling(top_p=0.9, temperature=1.0, temperature_adjuster=temperature_adjuster)

    # 使用Top-p采样从logits中采样一个token
    sampled_token_top_p = top_p_sampler.sample(logits=logits, past_tokens=past_tokens, step=step,
                                               content_quality=content_quality, perplexity=perplexity)
    print(f"Top-p sampled token: {sampled_token_top_p}")

    # 创建Top-k采样策略实例
    top_k_sampler = TopKSampling(top_k=3, temperature=1.0, temperature_adjuster=temperature_adjuster)

    # 使用Top-k采样从logits中采样一个token
    sampled_token_top_k = top_k_sampler.sample(logits=logits, past_tokens=past_tokens, step=step,
                                               content_quality=content_quality, perplexity=perplexity)
    print(f"Top-k sampled token: {sampled_token_top_k}")

    # 创建Mix采样策略实例
    mix_sampler = MixSampling(top_p=0.9, top_k=5, temperature=1.0, temperature_adjuster=temperature_adjuster)

    # 使用Mix采样从logits中采样一个token
    sampled_token_mix = mix_sampler.sample(logits=logits, past_tokens=past_tokens, step=step,
                                             content_quality=content_quality, perplexity=perplexity)
    print(f"Mix sampled token (Top-p first, then Top-k): {sampled_token_mix}")

    # 使用逆向策略（Top-k first, then Top-p）
    sampled_token_reverse = mix_sampler.sample(logits=logits, past_tokens=past_tokens, step=step,
                                                 content_quality=content_quality, perplexity=perplexity)
    print(f"Reverse sampled token (Top-k first, then Top-p): {sampled_token_reverse}")
