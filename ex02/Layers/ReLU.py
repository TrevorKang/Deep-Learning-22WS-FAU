import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None

    def forward(self, input_tensor):
        """

        :param input_tensor: a matrix, (batch_size, input-size) --> (b,n)
        :return: a tensor, the 'input tensor' for the next layer --> (b,n)
        """
        self.input_tensor = input_tensor
        self.output_tensor = np.maximum(0, input_tensor)
        # print(self.output_tensor.shape)
        return self.output_tensor

    def backward(self, error_tensor):
        """

        :param error_tensor: (b,n)
        :return: En-1, the 'output tensor' for the previous layer --> (b, n)
        """
        self.output_tensor = error_tensor.copy()
        self.output_tensor[self.input_tensor <= 0] = 0
        # print(self.output_tensor.shape)
        return self.output_tensor
