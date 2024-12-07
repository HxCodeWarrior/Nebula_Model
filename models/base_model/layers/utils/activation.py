# 激活函数模块（如ReLU、GELU等）
import numpy as np


# 基础激活函数类
class ActivationFunction:
    """
    激活函数的基础类，所有激活函数都应继承此类。
    """

    def __init__(self, regularization=None, reg_lambda=0.01):
        """
        初始化激活函数。

        参数：
        regularization -- 正则化方式，默认是None，可选 'l2' 或 'dropout'
        reg_lambda -- 正则化强度，默认为0.01
        """
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.cache = None

    def forward(self, X):
        """
        向前传播：对输入数据X应用激活函数。

        参数：
        X -- 输入数据，形状为 (batch_size, num_features)

        返回：
        激活函数的输出
        """
        raise NotImplementedError("forward method must be implemented in subclass.")

    def backward(self, dA, cache):
        """
        反向传播：计算损失函数相对于输入的梯度。

        参数：
        dA -- 从上一层传递来的梯度
        cache -- 前向传播时缓存的值，用于计算反向传播

        返回：
        对应的梯度
        """
        raise NotImplementedError("backward method must be implemented in subclass.")

    def apply_regularization(self, X):
        """
        应用正则化。这里可以根据 `regularization` 类型来处理。

        参数：
        X -- 激活函数的输入

        返回：
        正则化损失项
        """
        if self.regularization == 'l2':
            return 0.5 * self.reg_lambda * np.sum(np.square(X))  # L2 正则化
        elif self.regularization == 'dropout':
            # 简单实现Dropout（具体实现可以进一步优化）
            dropout_mask = np.random.binomial(1, 0.5, size=X.shape)  # 50%的丢弃
            return np.mean(dropout_mask)  # 返回丢弃的比例
        else:
            return 0


# 1. ReLU 激活函数
class ReLU(ActivationFunction):
    def forward(self, X):
        self.cache = X  # 缓存输入，用于反向传播
        output = np.maximum(0, X)
        reg_loss = self.apply_regularization(X)
        return output, reg_loss

    def backward(self, dA, cache):
        X = cache
        dX = dA * (X > 0)  # 对于ReLU，负值部分梯度为0
        return dX


# 2. Sigmoid 激活函数
class Sigmoid(ActivationFunction):
    def forward(self, X):
        self.cache = 1 / (1 + np.exp(-np.clip(X, -500, 500)))  # 使用np.clip避免溢出
        reg_loss = self.apply_regularization(X)
        return self.cache, reg_loss

    def backward(self, dA, cache):
        sigmoid = cache
        dX = dA * sigmoid * (1 - sigmoid)
        return dX


# 3. Tanh 激活函数
class Tanh(ActivationFunction):
    def forward(self, X):
        self.cache = np.tanh(X)
        reg_loss = self.apply_regularization(X)
        return self.cache, reg_loss

    def backward(self, dA, cache):
        tanh = cache
        dX = dA * (1 - tanh ** 2)
        return dX


# 4. LeakyReLU 激活函数
class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01, regularization=None, reg_lambda=0.01):
        super().__init__(regularization, reg_lambda)
        self.alpha = alpha

    def forward(self, X):
        self.cache = X
        output = np.where(X > 0, X, self.alpha * X)
        reg_loss = self.apply_regularization(X)
        return output, reg_loss

    def backward(self, dA, cache):
        X = cache
        dX = dA * np.where(X > 0, 1, self.alpha)
        return dX


# 5. ELU 激活函数
class ELU(ActivationFunction):
    def __init__(self, alpha=1.0, regularization=None, reg_lambda=0.01):
        super().__init__(regularization, reg_lambda)
        self.alpha = alpha

    def forward(self, X):
        self.cache = np.where(X > 0, X, self.alpha * (np.exp(X) - 1))
        reg_loss = self.apply_regularization(X)
        return self.cache, reg_loss

    def backward(self, dA, cache):
        X = cache
        dX = dA * np.where(X > 0, 1, X + self.alpha)
        return dX


# 6. Softmax 激活函数
class Softmax(ActivationFunction):
    def forward(self, X):
        # 防止溢出：通过减去最大值来进行数值稳定化
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        self.cache = exp_X / np.sum(exp_X, axis=-1, keepdims=True)
        reg_loss = self.apply_regularization(X)
        return self.cache, reg_loss

    def backward(self, dA, cache):
        softmax = cache
        dX = dA * softmax * (1 - softmax)
        return dX


# 7. Swish 激活函数
class Swish(ActivationFunction):
    def forward(self, X):
        self.cache = 1 / (1 + np.exp(-X))  # Sigmoid(x)计算
        output = X * self.cache
        reg_loss = self.apply_regularization(X)
        return output, reg_loss

    def backward(self, dA, cache):
        sigmoid = cache
        dX = dA * (sigmoid * (1 + X * (1 - sigmoid)))  # Swish的反向传播
        return dX


# 8. Mish 激活函数
class Mish(ActivationFunction):
    def forward(self, X):
        self.cache = np.tanh(np.log(1 + np.exp(X)))  # Softplus计算
        output = X * self.cache
        reg_loss = self.apply_regularization(X)
        return output, reg_loss

    def backward(self, dA, cache):
        softplus = cache
        dX = dA * (softplus + X * (1 - softplus ** 2))
        return dX


# 激活函数管理器类
class ActivationManager:
    def __init__(self):
        self.activations = {}

    def add_activation(self, name, activation_function):
        """
        将激活函数加入管理器。

        参数：
        name -- 激活函数的名字
        activation_function -- 激活函数的实例
        """
        self.activations[name] = activation_function

    def add_custom_activation(self, name, func):
        """
        支持用户添加自定义激活函数

        参数：
        name -- 激活函数的名字
        func -- 用户自定义的激活函数（Lambda 函数或可调用对象）
        """
        self.activations[name] = func

    def forward(self, name, X):
        """
        执行前向传播。

        参数：
        name -- 激活函数的名字
        X -- 输入数据

        返回：
        激活函数的输出
        """
        activation = self.activations.get(name)
        if activation is None:
            raise ValueError(f"Activation function '{name}' not found.")
        output, reg_loss = activation.forward(X)
        return output, reg_loss

    def backward(self, name, dA, cache):
        """
        执行反向传播。

        参数：
        name -- 激活函数的名字
        dA -- 从上一层传来的梯度
        cache -- 前向传播时缓存的值

        返回：
        对应的梯度
        """
        activation = self.activations.get(name)
        if activation is None:
            raise ValueError(f"Activation function '{name}' not found.")
        dX = activation.backward(dA, cache)
        return dX



# 测试模块
if __name__ == "__main__":
    # 创建激活函数管理器
    manager = ActivationManager()

    # 添加激活函数
    manager.add_activation('relu', ReLU())
    manager.add_activation('sigmoid', Sigmoid())
    manager.add_activation('tanh', Tanh())
    manager.add_activation('leakyrelu', LeakyReLU())
    manager.add_activation('elu', ELU())
    manager.add_activation('softmax', Softmax())
    manager.add_activation('swish', Swish())
    manager.add_activation('mish', Mish())

    # 输入数据
    X = np.array([[1.0, -2.0, 3.0], [-1.0, 4.0, -0.5]])

    # 测试前向传播
    print("ReLU Forward:")
    print(manager.forward('relu', X))

    print("Sigmoid Forward:")
    print(manager.forward('sigmoid', X))

    print("Tanh Forward:")
    print(manager.forward('tanh', X))

    print("LeakyReLU Forward:")
    print(manager.forward('leakyrelu', X))

    print("ELU Forward:")
    print(manager.forward('elu', X))

    print("Softmax Forward:")
    print(manager.forward('softmax', X))

    print("Swish Forward:")
    print(manager.forward('swish', X))

    print("Mish Forward:")
    print(manager.forward('mish', X))

    # 测试反向传播
    dA = np.array([[0.5, -0.5, 0.3], [0.1, -0.2, 0.4]])

    print("ReLU Backward:")
    print(manager.backward('relu', dA, manager.activations['relu'].cache))

    print("Sigmoid Backward:")
    print(manager.backward('sigmoid', dA, manager.activations['sigmoid'].cache))

    print("Tanh Backward:")
    print(manager.backward('tanh', dA, manager.activations['tanh'].cache))

    print("LeakyReLU Backward:")
    print(manager.backward('leakyrelu', dA, manager.activations['leakyrelu'].cache))

    print("ELU Backward:")
    print(manager.backward('elu', dA, manager.activations['elu'].cache))

    print("Softmax Backward:")
    print(manager.backward('softmax', dA, manager.activations['softmax'].cache))

    print("Swish Backward:")
    print(manager.backward('swish', dA, manager.activations['swish'].cache))

    print("Mish Backward:")
    print(manager.backward('mish', dA, manager.activations['mish'].cache))
