import unittest
import numpy as np
import Optimizers


class TestOptimizers1(unittest.TestCase):

    def test_sgd(self):
        optimizer = Optimizers.Sgd(1.)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.]))
