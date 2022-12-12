import numpy as np
from scipy.ndimage import correlate

input_tensor = np.array(range(27)).reshape((3,3,3))
print(input_tensor)
filter = np.array([[[0,1,0],
                   [1,1,1],
                   [0,1,0]]])

res = correlate(input_tensor, filter, None, 'constant')
print(res)