import numpy as np


class BaseLayer:
    """
        This class will be inherited by
        every layer in our framework.
    """
    def __init__(self):
        self.trainable = False
        self.weights = []

