import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.CrossEntropyLoss = None

    def forward(self, prediction_tensor, label_tensor):
        # TODO
        # when label is 1 --->  correct classification
        self.prediction_tensor = prediction_tensor
        loss = label_tensor * np.log(prediction_tensor + np.finfo(float).eps)
        self.CrossEntropyLoss = -np.sum(loss)
        return self.CrossEntropyLoss

    def backward(self, label_tensor):
        # TODO
        # Q:The gradient prohibits predictions of 0 as well ,so we need
        result = -label_tensor / (self.prediction_tensor + np.finfo(float).eps)
        return result
