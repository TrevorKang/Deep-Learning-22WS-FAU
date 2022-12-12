import numpy as np


class Sgd:

    def __init__(self, learning_rate):
        """
            learning-rate: float
        """
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        :param weight_tensor:
        :param gradient_tensor:
        :return: updated weights according to the basic gradient descent update scheme.
        """
        updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight


class SgdWithMomentum:
    # TODO
    def __init__(self, learning_rate, momentum_rate=0.9):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.vk = 0  # momentum

    def calculate_update(self, weight_tensor, gradient_tensor):
        """

        :param weight_tensor:
        :param gradient_tensor:
        :return: updated weights based on current and past gradients
        """
        self.vk = self.vk * self.momentum_rate - gradient_tensor * self.learning_rate
        updated_tensor = weight_tensor + self.vk
        return updated_tensor


class Adam:
    # TODO
    def __init__(self, learning_rate=0.001, mu=0.9, rho=0.999):
        self.learning_rate = learning_rate
        self.mu = mu  # beta1, trainable parameter
        self.rho = rho  # beta2, trainable parameter
        self.vk = 0
        self.rk = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        """

        :param weight_tensor:
        :param gradient_tensor:
        :return: updated weights based on current and past gradients, but with a bias correction
        """
        gk = gradient_tensor
        self.vk = self.mu * self.vk + (1 - self.mu) * gk
        self.rk = self.rho * self.rk + (1 - self.rho) * gk * gk
        # Bias correction
        vk_hat = self.vk / (1 - self.mu ** self.k)
        rk_hat = self.rk / (1 - self.rho ** self.k)
        self.k += 1  # counter
        updated_tensor = weight_tensor - self.learning_rate * vk_hat / (np.sqrt(rk_hat) + 1e-8)
        # epsilon: scaling factor
        return updated_tensor

