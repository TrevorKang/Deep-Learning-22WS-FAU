import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    # TODO
    """
        reshapes the multi-dimensional input to a one dimensional feature vector
    """
    def __init__(self):
        super().__init__()
        self.batch_size = None
        self.height = None
        self.width = None
        self.num_channel = None

    def forward(self, input_tensor):
        """

        :param input_tensor: (batch_size, height, width, #channel)
        :return: (batch_size, height * width * #channel)
        """
        self.batch_size, self.height, self.width, self.num_channel = input_tensor.shape
        return np.reshape(input_tensor.reshape(-1), (self.batch_size, self.height * self.width * self.num_channel))

    def backward(self, error_tensor):
        """

        :param error_tensor: (batch_size, height * width * #channel)
        :return: (batch_size, height, width, #channel)
        """
        return error_tensor.reshape((self.batch_size, self.height, self.width, self.num_channel))

