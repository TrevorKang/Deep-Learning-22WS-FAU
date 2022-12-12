import numpy as np
from copy import deepcopy
from Layers.Base import BaseLayer
from Layers.Initializers import UniformRandom
import scipy.ndimage
import scipy.signal


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True  # trainable layers
        self.num_kernels = num_kernels
        self.input_tensor = []
        self.output_tensor = []

        # stride_shape
        self.stride_shape = stride_shape
        # 3D, kernel shape (c,m,n) _tuple
        # 2D, kernel shape (c, m)
        self.kernel_shape = convolution_shape

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

    def gradients_helper(self, error_tensor):

        (num_samples, num_channels, n_H_prev, n_W_prev) = self.input_tensor.shape

        gradient_weight = np.zeros((self.num_kernels, *self.kernel_shape))
        gradient_bias = np.zeros(self.num_kernels)

        # gradient of weights
        upsampled_error_tensor = np.zeros((num_samples, self.num_kernels, *self.input_tensor.shape[2:]))
        for i in range(num_samples):
            # down sampling and followed by convolution
            if np.shape(self.stride_shape) == (2,):
                upsampled_error_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[i]
                # zero padding for conv
                pad_y1 = self.kernel_shape[1] // 2
                pad_y2 = (self.kernel_shape[1] - 1) // 2

                pad_x1 = (self.kernel_shape[2]) // 2
                pad_x2 = (self.kernel_shape[2] - 1) // 2
                padding = ((0, 0), (pad_y1, pad_y2), (pad_x1, pad_x2))
            else:
                upsampled_error_tensor[:, :, ::self.stride_shape[0]] = error_tensor[i]
                padding = ((0, 0), (self.kernel_shape[1] // 2, (self.kernel_shape[1] - 1) // 2))

            input_tensor_padded = np.pad(self.input_tensor[i], padding, mode='constant', constant_values=0)
            single_kernel_output = np.zeros((self.num_kernels, *self.kernel_shape))
            for k in range(self.num_kernels):
                for ch in range(num_channels):
                    single_kernel_output[k, ch] = scipy.signal.correlate(
                                                    input_tensor_padded[ch], upsampled_error_tensor[i, k], 'valid')
            gradient_weight += single_kernel_output

            # gradient of bias is simply the sums over error tensor
            if np.shape(self.stride_shape) == (2,):
                gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
            else:
                gradient_bias = np.sum(error_tensor, axis=(0, 2))

        return gradient_weight, gradient_bias

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def forward(self, input_tensor):
        """
        returns a tensor that serves as the input tensor for the next layer
        :param input_tensor: （b:样本数，c:通道数, y:样本纵向高度, x:样本横向宽度）
        :return: output_tensor: (b, c:卷积核数量, y, x)
        """

        self.input_tensor = input_tensor
        # 2D convolution
        if len(self.input_tensor.shape) == 4:
            self.input_tensor = input_tensor
            (num_samples, num_channels, y, x) = self.input_tensor.shape
            if y % self.stride_shape[0] != 0:
                n_H = y // self.stride_shape[0] + 1
            else:
                n_H = y // self.stride_shape[0]
            if x % self.stride_shape[1] != 0:
                n_W = x // self.stride_shape[1] + 1
            else:
                n_W = x // self.stride_shape[1]
            # 用0初始化输出张量
            self.output_tensor = np.zeros((num_samples, self.num_kernels, n_H, n_W))
            # 卷积（cross correlation）
            for i in range(num_samples):
                single_sample = input_tensor[i]  # 3,10,14
                for k in range(self.num_kernels):
                    # TODO:
                    correlated_tensor = scipy.ndimage.correlate(single_sample, self.weights[k, :, :, :], None,
                                                                'constant')
                    # valid padding in channel axis
                    single_kernel_output = correlated_tensor[num_channels // 2]
                    # down sampling
                    strided_output = single_kernel_output[::self.stride_shape[0], ::self.stride_shape[1]]
                    self.output_tensor[i, k, :, :] = strided_output + self.bias[k]
        else:
            # 2D Convolution
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
        self.optimizer_bias = deepcopy(optimizer)
        self.optimizer_weights = deepcopy(optimizer)

    def backward(self, error_tensor):
        """
                # 1. upper tensor
                # 2. gradient
                # 3. bias
                # 4. optimization of gradient and bias
                :param error_tensor: 上一层输出， (b, c:卷积核数量, y, x)
                :return: (b, c: 频道数, y, x)
                """
        # TODO
        # 2D Convolution:
        if len(error_tensor.shape) == 4:
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
                    single_output = single_output[self.num_kernels // 2]
                    upper_tensor[i, c, :, :] = single_output

            # calculate in gradient of weights and bias
            self._gradient_weights, self._gradient_bias = self.gradients_helper(error_tensor)

            if self.optimizer is not None:
                self.weights = self.optimizer_weights.calculate_update(self.weights, self._gradient_weights)
                self.bias = self.optimizer_bias.calculate_update(self.bias, self._gradient_bias)
        # 1D Convolution
        else:
            A_prev = self.input_tensor
            (num_samples, num_channels, n_H_prev) = A_prev.shape
            (num_samples, num_kernels, n_H) = error_tensor.shape
            (num_kernels, num_channels, m) = self.weights_shape
            pad = m // 2
            stride = self.stride_shape[0]
            upper_tensor = np.zeros((num_samples, num_channels, n_H_prev))
            self._gradient_weights = np.zeros((num_kernels, num_channels, m))
            self._gradient_bias = np.zeros(num_kernels)
            A_prev_padded = np.pad(A_prev, ((0, 0), (0, 0),
                                            (pad, pad)), constant_values=0)
            dA_prev_padded = np.pad(upper_tensor, ((0, 0), (0, 0),
                                            (pad, pad)), constant_values=0)
            for i in range(num_samples):
                # 选择第i个样本，降维
                a_prev_padded = A_prev_padded[i]
                da_prev_padded = dA_prev_padded[i]
                # TODO
                for c in range(num_kernels):
                    for h in range(n_H):
                        start = h * stride
                        end = start + m
                        a_slice = a_prev_padded[:, start:end]
                        da_prev_padded[:, start:end] += \
                            self.weights[c, :, :] * error_tensor[i, c, h]
                        self._gradient_weights[c, :, :] += a_slice * error_tensor[i, c, h]
                        self._gradient_bias[c] += error_tensor[i, c, h]
                upper_tensor[i, :, :] = da_prev_padded[:, pad:-pad]

        # 对权重和偏移进行优化
        if self.optimizer is not None:
            self.weights = self.optimizer_weights.calculate_update(self.weights, self._gradient_weights)
            self.bias = self.optimizer_bias.calculate_update(self.bias, self._gradient_bias)
        return upper_tensor

    def initialize(self, weights_initializer, bias_initializer):
        # reinitialize weights and biases
        input_size = np.prod(self.kernel_shape)  # b * m * n
        output_size = (self.num_kernels * self.kernel_shape[1] * self.kernel_shape[2])  # k * m * n
        self.weights = weights_initializer.initialize(self.weights_shape, input_size, output_size)
        self.bias = bias_initializer.initialize(self.num_kernels, input_size, output_size)
