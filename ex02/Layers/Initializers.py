import numpy as np
from Layers.Base import BaseLayer


class Constant(BaseLayer):
    """
    determines the constant value used for weight initialization, typically for bias.
    """
    # TODO
    def __init__(self, value=0.1):
        super().__init__()
        self.value = value

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        initialized_tensor = self.value * np.ones(weights_shape)
        return initialized_tensor


class UniformRandom(BaseLayer):
    # TODO
    def __init__(self):
        super().__init__()

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        initialized_tensor = np.random.uniform(0, 1, weights_shape)
        return initialized_tensor


class Xavier(BaseLayer):
    # TODO
    def __init__(self):
        super().__init__()

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2.0 / (fan_in + fan_out))
        initialized_tensor = np.random.normal(0, sigma, weights_shape)
        return initialized_tensor


class He(BaseLayer):
    # TODO
    def __init__(self):
        super().__init__()

    def initialize(self, weights_shape, fan_in, fan_out=None):
        sigma = np.sqrt(2.0 / fan_in)
        initialized_tensor = np.random.normal(0, sigma, weights_shape)
        return initialized_tensor
