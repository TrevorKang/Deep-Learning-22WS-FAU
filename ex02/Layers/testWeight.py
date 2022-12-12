import numpy as np


input_size = 4
catagories = 10
weight = np.random.uniform(0, 1, (input_size + 1, catagories))

print('weights:', weight, '\n')

print('weights[-1, :]', weight[-1, :], '\n')

print('weights[:-1,:]', weight[:-1, :], '\n')

