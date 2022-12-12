import numpy as np

a = (2, 3)
print('a shape: ', np.shape(a))
# c = (a + (1,))
# print(np.shape(c))
# print(b)
b = np.expand_dims(np.array(a), axis=1)
print('b: ', b)
print('b shape: ', b.shape)

c = a + (1,)
print('c: ', c)
print('c shape: ', np.shape(c))

d = (a, 1)
print('c: ', d)

kernel_size = (3, 5, 8)
# 3 channels
# 5*5, 8*8

stride_shape = [2]
print(np.shape(stride_shape))
print((stride_shape + stride_shape))

num_kernels = 5
print((num_kernels,) + kernel_size)


