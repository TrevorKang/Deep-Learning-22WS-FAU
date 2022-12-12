import numpy as np

np.random.seed(1337)

input_tensor = np.random.random((4, 5, 5))
print(input_tensor)
# print(a)
# print()

s = np.zeros((5, 5))
for i in range(4):
    s += input_tensor[i]
print(s)
print()

stride = 2
num_channel = 3

t = input_tensor[num_channel//2][::stride, ::stride]
k = s[::stride, ::stride]
print(t)
print(k)

temp = np.array([1,2,3,4])
print(*temp[2:])