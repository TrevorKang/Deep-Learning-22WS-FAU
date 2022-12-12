import numpy as np
from scipy.ndimage import correlate as ncorrelate

a = np.arange(36, dtype=np.int32).reshape((6, 6))
print(a)

def conv_helper(a_slice, kernel):
    s = np.multiply(a_slice, kernel)
    z = np.sum(s)
    return z


# suppose stride as (2,2)
kernel = np.array(
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]])
print(kernel)
c1 = ncorrelate(a, kernel, None, 'constant')
c2 = np.zeros((6, 6))
a_padded = np.pad(a, ((1, 1), (1, 1)), constant_values=0)
print(a_padded)
for i in range(6):
    for j in range(6):
        single_clip = a_padded[i:i+3, j:j+3]
        c2[i, j] = conv_helper(single_clip, kernel)

print(c2)
print(c1)
k = (3,3) + (1,)
print(k)
