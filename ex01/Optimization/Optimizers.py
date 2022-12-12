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
        updated_weight = np.subtract(weight_tensor, self.learning_rate * gradient_tensor)
        return updated_weight
