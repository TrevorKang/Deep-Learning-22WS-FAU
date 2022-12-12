import numpy as np
import copy


class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer

        self.loss = []  # contain the loss value for each iteration after calling train.
        self.layers = []
        self.data_layer = None  # provide input data and labels
        self.loss_layer = None  # referring to the special layer providing loss and prediction.

    def forward(self):
        # find in the file(Helpers-->.next)
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            # recursively calls forward on its layer passing the input-data
        return self.loss_layer.forward(input_tensor,
                                       self.label_tensor)

    def backward(self):

        error_tensor = self.loss_layer.backward(
            self.label_tensor)
        # recursively calls backward on its layers passing the error
        for layer in self.layers[::-1]:  # from back to front
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        """
        TODO: Splice all the layer(append trainable and non-trainable layers to the list layers.)
        :param layer:
        :return: self.layers = [....]
        """

        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)  # deep-copy & shallow copy
        self.layers.append(layer)

    def train(self, iterations):
        """

        :param iterations: the number of iteration
        :return:  store the loss for each iteration.
        """
        for number_of_iteration in range(iterations):
            self.loss.append(self.forward())  # stores the loss for each iteration.
            self.backward()

    def test(self, input_tensor):
        """

        :param input_tensor:
        :return:  returns the prediction of the last layer.
        """
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
