import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None

    def forward(self, input_tensor):
        """

        :param input_tensor:
        :return:the estimated class probabilities for each row
        """
        # note:use xk = xk - max (x) to increase numerical stability

        x_shift = input_tensor - np.max(input_tensor)
        xk = np.exp(x_shift)
        esum = np.sum(np.exp(x_shift), axis=1, keepdims=True)
        # keepdims : the result will broadcast correctly against the input array.
        self.output_tensor = xk / esum
        # print("shape of forward", self.output_tensor.shape)
        return self.output_tensor

    def backward(self, error_tensor):
        """

        :param error_tensor:
        :return: a tensor that serves as the error tensor for the previous layer.
        """
        # TODO

        e_n_ = error_tensor-np.sum(np.multiply(error_tensor,self.output_tensor),axis=1,keepdims=True)
        self.output_tensor = self.output_tensor*e_n_
        # print("shape of backward", self.output_tensor.shape)
        return self.output_tensor
