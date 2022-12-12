import numpy as np
from Layers.Base import BaseLayer
from Optimization import *


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        """

        :param input_size:
        :param output_size:
        """
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights_shape = (input_size, output_size)
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))  # initial weights, +1 cuz bias
        self.input_tensor = None
        self.output_tensor = None
        self._optimizer = None
        self.gradient_weight = None

    def forward(self, input_tensor):
        """

        :param input_tensor: a matrix, (batch_size, input-size) --> (b,n)
        :return: a tensor that serves as the input tensor for the next layer --> (b, m)
        """
        batch_size, input_size = np.shape(input_tensor)
        # Storage all the prime variables
        self.input_tensor = np.concatenate((input_tensor, np.ones((batch_size, 1))), axis=1)  # (b, n+1)
        self.output_tensor = np.dot(self.input_tensor, self.weights)  # (b, n+1) (n+1, m) ---> (b, m)
        self.output_size = np.shape(self.output_tensor)[1]  # m: the output size
        return self.output_tensor

    @property
    def optimizer(self):
        """
        setter & getter method
        :return: self._optimizer
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        """

        :param error_tensor: En with the same shape as out_tensor Y of forward()--> (b, m)
        :return: the error tensor for the previous layer, En-1 in shape of X -->(b, n)
        """
        upper_error = np.dot(error_tensor, self.weights.T)  # (b, m)(m, n+1) --> (b, n+1)
        upper_error = np.delete(upper_error, -1, axis=1)  # bias in the last column shall be deleted
        self.gradient_weight = np.dot(self.input_tensor.T, error_tensor)  # Dot product of X_prime.T & En_prime
        # print(self.gradient_weight.shape)

        # e.g.
        # layer.optimizer = Optimizers.Sgd(learning_rate)
        # _optimizer then can be set to perform SGD
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weight)
        return upper_error

    @property
    def gradient_weights(self):
        return self.gradient_weight

    # TODO
    def initialize(self, weights_initializer, bias_initializer):
        # Append initialized weights to our current weights
        self.weights[:-1, :] = weights_initializer.initialize(self.weights_shape, self.input_size, self.output_size)
        # Append the initialized bias to the weights matrix
        self.weights[-1, :] = bias_initializer.initialize(self.output_size, self.input_size, self.output_size)


