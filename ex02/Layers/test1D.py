import numpy as np
batch_size = 2
a = np.array(range(3 * 15 * batch_size), dtype=float)
a = a.reshape((batch_size, 3, 15))
print(a)
print(a[1,1].shape)

b = a[1,1]
b = np.pad(b, (1,1), constant_values=0)
print(b.shape)