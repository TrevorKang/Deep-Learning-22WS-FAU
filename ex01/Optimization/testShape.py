import numpy as np

# batch_size = 32
# input_size = 64
#
# input_tensor = np.ones((input_size, batch_size))
#
# print(input_tensor.shape)
#
# print(np.ones((4, 3)))

a = np.random.randn(3, 4)
print(a)

a[a <= 0] = 0
print(a)
