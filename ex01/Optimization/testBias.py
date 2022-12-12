import numpy as np

a = np.zeros((5, 4))
print(a.shape)

b = np.ones((1, 4))

print(np.concatenate((a, b), axis=0))

print(b.shape[1])

print(np.shape(b)[1])