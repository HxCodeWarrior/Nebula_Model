import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureAdjuster(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 hidden_size: int = 256,
                 output_size: int = 1,
                 initial_temp: float = 1.0,  # 初始化温度
                 min_temp: float = 0.7,
                 max_temp: float = 1.5,
                 dynamic_factor: float = 0.01,
                 decay_mode: str = 'exponential'):
        """
        高效且灵活的动态温度调整器，支持初始化温度以及混合调整策略。

        :param input_size: 输入特征的维度，例如生成步数、困惑度等。
        :param hidden_size: 隐藏层大小，用于学习温度调整的表示。
        :param output_size: 输出的维度（温度调整因子）。
        :param initial_temp: 初始化温度，控制生成内容的初始多样性。
        :param min_temp: 最小温度。
        :param max_temp: 最大温度。
        :param dynamic_factor: 温度调整因子，控制温度变化的速度。
        :param decay_mode: 温度衰减模式，支持 'exponential' 和 'linear'。
        """
        super(TemperatureAdjuster, self).__init__()

        # 网络结构，用于学习温度调整因子
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.initial_temp = initial_temp  # 初始化温度
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.dynamic_factor = dynamic_factor
        self.decay_mode = decay_mode

    def forward(self, x):
        """
        前向传播，计算温度调整因子。

        :param x: 输入特征，包含生成步数、困惑度等。
        :return: 温度调整因子（标量）。
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # 温度因子的激活函数，确保它在[min_temp, max_temp]范围内
        temperature = torch.sigmoid(x) * (self.max_temp - self.min_temp) + self.min_temp
        return temperature


    def adjust_logits_with_temperature(self, logits: torch.Tensor, step: int, content_quality: float = 1.0,
                                       perplexity: float = 1.0) -> torch.Tensor:
        """
        根据当前步数和温度调整logits。

        :param logits: 形状为[batch_size, vocab_size]的logits张量。
        :param step: 当前生成的步数。
        :param content_quality: 内容质量评分，用于平滑调整温度。
        :param perplexity: 当前生成内容的困惑度评分。
        :return: 根据调整后的温度缩放的logits。
        """
        # 输入特征，包含步数、内容质量、困惑度等（例如使用step和perplexity作为输入）
        input_features = torch.tensor([step, content_quality, perplexity], dtype=torch.float32).unsqueeze(
            0)  # Batch size为1
        temperature = self(input_features)  # 获取温度调整因子

        # 调整logits
        adjusted_logits = logits / temperature
        return adjusted_logits


    def adjust_temperature(self, step: int, content_quality: float = 1.0, perplexity: float = 1.0) -> float:
        """
        根据当前步数、内容质量和困惑度调整温度。

        :param step: 当前生成的步数。
        :param content_quality: 内容质量评分。
        :param perplexity: 当前生成内容的困惑度评分。
        :return: 调整后的温度值。
        """
        # 使用初始化温度作为基准
        input_features = torch.tensor([step, content_quality, perplexity], dtype=torch.float32).unsqueeze(0)
        temperature = self(input_features).item()

        # 将step转换为Tensor以便和temperature进行操作
        step_tensor = torch.tensor(step, dtype=torch.float32)

        # 选择温度调整模式
        if self.decay_mode == 'exponential':
            temperature = self.initial_temp * torch.exp(-self.dynamic_factor * step_tensor)  # 使用指数衰减
        elif self.decay_mode == 'linear':
            temperature = self.initial_temp - self.dynamic_factor * step_tensor  # 使用线性衰减

        # 可以加入内容质量因子的平滑调整（例如，通过文本重复性来调整温度）
        temperature *= content_quality
        # 返回最终的温度值，保证在范围内
        temperature =  max(self.min_temp, min(self.max_temp, temperature))

        return temperature


if __name__ == '__main__':
    # 示例：使用优化后的动态温度调整器
    logits = torch.randn(1, 10000)  # 模拟一个batch大小为1，词汇量为10000的logits
    step = 10  # 假设生成的步数是10
    content_quality = 0.8  # 假设内容质量为0.8，较低表示生成质量较差
    perplexity = 30.0  # 假设当前生成内容的困惑度是30

    # 初始化优化后的动态温度调整器
    temperature_adjuster = TemperatureAdjuster(
        input_size=3,
        hidden_size=128,
        output_size=1,
        initial_temp=1.2,
        min_temp=0.7,
        max_temp=1.5,
        decay_mode='exponential'
    )

    # 使用该温度调整器调整logits
    adjusted_logits = temperature_adjuster.adjust_logits_with_temperature(
        logits, step=step,
        content_quality=content_quality,
        perplexity=perplexity
    )

    print("Adjusted Logits with Optimized Dynamic Temperature:", adjusted_logits)

    # 获取调整后的温度
    adjusted_temperature = temperature_adjuster.adjust_temperature(
        step=step,
        content_quality=content_quality,
        perplexity=perplexity
    )
    print("Adjusted Temperature:", adjusted_temperature)
