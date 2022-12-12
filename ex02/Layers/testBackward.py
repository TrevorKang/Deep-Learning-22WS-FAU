import numpy as np
from Layers.Base import BaseLayer
from Layers.Initializers import UniformRandom
from Optimization import *
from Layers.Flatten import Flatten
from copy import deepcopy
import Helpers
import scipy.signal

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True  # trainable layers
        self.num_kernels = num_kernels
        self.input_tensor = []
        self.output_tensor = []

        # stride_shape
        if np.shape(stride_shape) == (1,):
            self.stride_shape = (stride_shape + stride_shape)
        else:
            self.stride_shape = stride_shape

        self.kernel_shape = convolution_shape  # 2D, kernel shape (c,m,n) _tuple

        # initialize the parameters of this layer uniformly random in the range [0; 1)
        self.weights_shape = (self.num_kernels,) + self.kernel_shape
        self.weights = UniformRandom().initialize(weights_shape=self.weights_shape)
        self.bias = UniformRandom().initialize(weights_shape=self.num_kernels)

        self._gradient_weights = None
        self._gradient_bias = None

        self._optimizer = None
        self.optimizer_weights = None
        self.optimizer_bias = None

    def conv_helper(self, a_slice, kernel):
        s = np.multiply(a_slice, kernel)
        z = np.sum(s)
        return z

    def forward(self, input_tensor):
        """
        returns a tensor that serves as the input tensor for the next layer
        :param input_tensor: （b:样本数，c:通道数, y:样本纵向高度, x:样本横向宽度）
        :return: output_tensor: (b, c:卷积核数量, y, x)
        """

        self.input_tensor = input_tensor
        # 2D convolution
        if len(self.input_tensor.shape) == 4:
            (num_samples, num_channels, y, x) = self.input_tensor.shape
        # 卷积核的尺寸 m x n
            m = self.kernel_shape[1]
            n = self.kernel_shape[2]
        # 计算padding与stride后输出尺寸
            pad_y1 = m // 2
            if m % 2 == 0:
                pad_y2 = pad_y1 - 1
            else:
                pad_y2 = pad_y1
            pad_x1 = n // 2
            if n % 2 == 0:
                pad_x2 = pad_x1 - 1
            else:
                pad_x2 = pad_x1
            n_H = int((y - m + pad_y1 + pad_y2) / self.stride_shape[0]) + 1
            n_W = int((x - n + pad_x1 + pad_x2) / self.stride_shape[1]) + 1
            # 用0初始化输出张量
            self.output_tensor = np.zeros((num_samples, self.num_kernels, n_H, n_W))
            # 根据padding进行填充
            input_tensor_padded = np.pad(input_tensor, ((0, 0), (0, 0),
                                        (pad_y1, pad_y2),
                                        (pad_x1, pad_x2)), constant_values=0)
            for i in range(num_samples):
                single_sample = input_tensor_padded[i]
                for c in range(self.num_kernels):
                    for h in range(n_H):
                        for w in range(n_W):
                            # 定位当前的切片位置
                            vertical_start = h * self.stride_shape[0]  # y向 subsampling
                            vertical_end = vertical_start + m
                            horizont_start = w * self.stride_shape[1]  # x向 subsampling
                            horizont_end = horizont_start + n
                            single_clip = single_sample[:, vertical_start:vertical_end, horizont_start:horizont_end]
                            self.output_tensor[i, c, h, w] = self.conv_helper(single_clip,
                                                                              self.weights[c, :, :, :]
                                                                              ) + self.bias[c]
            assert (self.output_tensor.shape == (num_samples, self.num_kernels, n_H, n_W))
        else:
            # 1D Convolution
            # 思路同上 只是不进行x方向的互相关操作
            (num_samples, num_channels, y) = self.input_tensor.shape
            pad = self.kernel_shape[1] // 2
            output_length = int((y - self.kernel_shape[1] + pad * 2) / self.stride_shape[0]) + 1
            self.output_tensor = np.zeros((num_samples, self.num_kernels,  output_length))
            input_tensor_padded = np.pad(input_tensor, ((0, 0), (0, 0), (pad, pad)), constant_values=0)
            for i in range(num_samples):
                single_sample = input_tensor_padded[i]
                for c in range(self.num_kernels):
                    for h in range(output_length):
                        start = h * self.stride_shape[0]
                        end = start + self.kernel_shape[1]
                        single_clip = single_sample[:, start:end]
                        self.output_tensor[i, c, h] = self.conv_helper(single_clip, self.weights[c,:,:]) + self.bias[c]

            assert (self.output_tensor.shape == (num_samples, self.num_kernels, output_length))
        return self.output_tensor

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        """
        setter & getter method
        storing the optimizer for this layer
        :return: self._optimizer
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.optimizer_bias = optimizer.copy()
        self.optimizer_weights = optimizer.copy()

    def backward(self, error_tensor):
        """
        # 1. upper tensor
        # 2. gradient
        # 3. bias
        # 4. optimize gradient and bias
        :param error_tensor: 上一层输出， (b, c:卷积核数量, y, x)
        :return: (b, c: 频道数, y, x)
        """
        # TODO
        # 获取原向量基本信息
        A_prev = self.input_tensor
        (num_samples, num_channels, n_H_prev, n_W_prev) = A_prev.shape  # (2, 3, 10, 14)

        # 获取error tensor(En)的基本信息
        (num_samples, num_kernels, n_H, n_W) = error_tensor.shape  # (2, 4, 10 ,14)

        # 获取weights(kernel)的基本信息
        (num_kernels, num_channels, m, n) = self.weights_shape  # (4, 3, 5, 8)

        # 求超参数 pad1 pad2 以及stride
        pad_y1 = m // 2
        if m % 2 == 0:
            pad_y2 = pad_y1 - 1
        else:
            pad_y2 = pad_y1
        pad_x1 = n // 2
        if n % 2 == 0:
            pad_x2 = pad_x1 - 1
        else:
            pad_x2 = pad_x1

        stride_y = self.stride_shape[0]
        stride_x = self.stride_shape[1]

        # 初始化各个梯度的结构
        dA_prev = np.zeros((num_samples, num_channels, n_H_prev, n_W_prev))
        self._gradient_weights = np.zeros((num_kernels, num_channels, m, n))
        self._gradient_bias = np.zeros(num_kernels)

        # zero-padding
        A_prev_padded = np.pad(A_prev, ((0, 0), (0, 0),
                                        (pad_y1, pad_y2),
                                        (pad_x1, pad_x2)), constant_values=0)
        dA_prev_padded = np.pad(dA_prev, ((0, 0), (0, 0),
                                        (pad_y1, pad_y2),
                                        (pad_x1, pad_x2)), constant_values=0)

        # 卷积
        for i in range(num_samples):
            # 选择第i个样本，降维
            a_prev_padded = A_prev_padded[i]
            da_prev_padded = dA_prev_padded[i]
            # TODO
            for c in range(num_kernels):
                for h in range(n_H):
                    for w in range(n_W):
                        vertical_start = h * stride_y
                        vertical_end = vertical_start + m
                        horizont_start = w * stride_x
                        horizont_end = horizont_start + n

                        a_slice = a_prev_padded[:, vertical_start:vertical_end, horizont_start:horizont_end]
                        # 计算梯度
                        da_prev_padded[:, vertical_start:vertical_end, horizont_start:horizont_end] += \
                            self.weights[c, :, :, :] * error_tensor[i, c, h, w]
                        self._gradient_weights[c, :, :, :] += a_slice * error_tensor[i, c, h, w]
                        self._gradient_bias[c] += error_tensor[i, c, h, w]

            dA_prev[i, :, :, :] = da_prev_padded[:, pad_y1:-pad_y2, pad_x1:-pad_x2]

        if self.optimizer is not None:
            self.weights = self.optimizer_weights.calculate_update(self.weights, self._gradient_weights)
            self.bias = self.optimizer_bias.calculate_update(self.bias, self._gradient_weights)
        return dA_prev

    def backward_new(self, error_tensor):
        # rearrange: transpose the weights and flip the kernel
        weights = np.fliplr(np.swapaxes(self.weights, 0, 1))

        # Basic Info from the input, could be used to build upper tensor
        (num_samples, num_channels, n_H_prev, n_W_prev) = self.input_tensor.shape

        # Basic Info from the error tensor
        (num_samples, num_kernels, n_H, n_W) = error_tensor.shape

        # Initialize upper tensor with zeros
        upper_tensor = np.zeros((num_samples, num_channels, n_H_prev, n_W_prev))

        # unlike the forward, in backprop we need up-sampling
        upsampled_error_tensor = np.zeros((num_samples, num_kernels, n_H_prev, n_W_prev))
        for i in range(num_samples):
            for c in range(num_channels):
                upsampled_error_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[i]
                # same padding in spatial domain
                single_output = scipy.signal.convolve(upsampled_error_tensor[i], weights[c], 'same')
                # valid padding in channel axis
                single_output = single_output[num_channels//2]
                upper_tensor[i, c, :, :] = single_output

        # calculate in gradient of weights and bias
        self._gradient_weights, self._gradient_bias = self.calculate_gradients(error_tensor)

        if self.optimizer is not None:
            self.weights = self.optimizer_weights.calculate_update(self.weights, self._gradient_weights)
            self.bias = self.optimizer_bias.calculate_update(self.bias, self._gradient_bias)
        return upper_tensor

    def calculate_gradients(self, error_tensor):

        (num_samples, num_channels, n_H_prev, n_W_prev) = self.input_tensor.shape

        gradient_weight = np.zeros((self.num_kernels, *self.kernel_shape))
        gradient_bias = np.zeros(self.num_kernels)

        upsampled_error_tensor = np.zeros((num_samples, self.num_kernels, *self.input_tensor.shape[2:]))
        for i in range(num_samples):
            # upsampling and followed by convolution
            if np.shape(self.stride_shape) == (2,):
                upsampled_error_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[i]
                # zero padding for conv
                pad_y1 = self.kernel_shape[1] // 2
                pad_y2 = (self.kernel_shape[1] - 1) // 2

                pad_x1 = (self.kernel_shape[1] - 1) // 2
                pad_x2 = (self.kernel_shape[2] - 1) // 2
                padding = ((0, 0), (pad_y1, pad_y2), (pad_x1, pad_x2))
            else:
                upsampled_error_tensor[:, :, ::self.stride_shape[0]] = error_tensor[i]
                padding = ((0, 0), (self.kernel_shape[1] // 2, (self.kernel_shape[1] - 1) // 2))

            input_tensor_padded = np.pad(self.input_tensor[i], padding, mode='constant', constant_values=0)
            single_kernel_output = np.zeros((self.num_kernels, *self.kernel_shape))
            for ker in range(self.num_kernels):
                for ch in range(num_channels):
                    single_kernel_output[ker, ch] = scipy.signal.correlate(input_tensor_padded[ch], upsampled_error_tensor[i, ker], 'valid')
            gradient_weight += single_kernel_output
            # gradient of bias is simply the sums over error tensor
            if np.shape(self.stride_shape) == (2,):
                gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
            else:
                gradient_bias = np.sum(error_tensor, axis=(0, 2))

        return gradient_weight, gradient_bias



class L2Loss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)


if __name__ == '__main__':
    print('------Test for Backprop - 2D Case-----')
    print('Basic Info:')
    kernel_shape = (3, 5, 8)
    num_kernels = 4
    input_size = (3, 10, 14)
    batch_size = 2
    categories = 105

    conv = Conv((3, 2), kernel_shape, num_kernels)
    input_tensor = np.array(range(np.prod(input_size) * batch_size))
    input_tensor = input_tensor.reshape(batch_size, *input_size)
    print('Shape of input - ', input_tensor.shape)
    # print('Hyperparameter - Stride ', conv.stride_shape)
    #
    output_tensor = conv.forward(input_tensor)
    print('Shape of output - ', output_tensor.shape)
    error_tensor = conv.backward_new(output_tensor)
    print('Shape of error tensor - ', error_tensor.shape)

    # print()
    # print('-----Test for Gradient and Bias-----')
    # print('Test gradient:')

    # label_tensor = np.zeros([batch_size, categories])
    # for i in range(batch_size):
    #     label_tensor[i, np.random.randint(0, categories)] = 1
    # np.random.seed(1337)
    # input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
    #
    # layers = list()
    # layers.append(Conv((1, 1), (3, 3, 3), 3))
    # layers.append(Flatten())
    # layers.append(L2Loss())
    # diff = Helpers.gradient_check(layers, input_tensor, label_tensor)
    # print(np.sum(diff))






