import numpy as np
import scipy.ndimage

from Layers.Base import BaseLayer
from Layers.Initializers import UniformRandom
# from scipy.ndimage import correlate as ncorrelate

from scipy.ndimage.filters import gaussian_filter


class Conv(BaseLayer):
    # TODO
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True  # trainable layers
        self.num_kernels = num_kernels
        self.input_tensor = []
        self.output_tensor = []
        # TODO:根据stride_shape进行讨论
        self.stride_shape = stride_shape
        # TODO: determines whether this object provides a 1D or a 2D convolution layer
        self.kernel_shape = convolution_shape  # 2D, kernel shape (c,m,n) _tuple
        # TODO: initialize the parameters of this layer uniformly random in the range [0; 1)
        self.weights_shape = (self.num_kernels,) + self.kernel_shape
        self.weights = UniformRandom().initialize(weights_shape=self.weights_shape)
        self.bias = UniformRandom().initialize(weights_shape=self.num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        # TODO:
        self._optimizer = None
        self.optimizer_weights = None
        self.optimizer_bias = None

    def same_padding_helper(self, X):
        m = self.kernel_shape[1]
        n = self.kernel_shape[2]

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

        return np.pad(X, ((0, 0), (0, 0),
                          (pad_y1, pad_y2),
                          (pad_x1, pad_x2)), constant_values=0)

    def conv_helper(self, a_slice, kernel):
        s = np.multiply(a_slice, kernel)
        z = np.sum(s)
        return z

    # TODO:returns a tensor that serves as the input tensor for the next layer
    def forward(self, input_tensor):
        """

        :param input_tensor: （b:样本数，c:通道数, y:样本纵向高度, x:样本横向宽度）
        :return: output_tensor: (b, c:卷积核数量, y, x)
        """

        self.input_tensor = input_tensor
        (num_samples, num_channels, y, x) = self.input_tensor.shape
        if y % self.stride_shape[0] != 0:
            n_H = y//self.stride_shape[0] + 1
        else:
            n_H = y//self.stride_shape[0]
        if x % self.stride_shape[1] != 0:
            n_W = x // self.stride_shape[1] + 1
        else:
            n_W = x // self.stride_shape[1]
        # 用0初始化输出张量
        self.output_tensor = np.zeros((num_samples, self.num_kernels, n_H, n_W))
        # 卷积（cross correlation）
        for i in range(num_samples):
            single_sample = input_tensor[i]   # 3,10,14
            for k in range(self.num_kernels):
                # TODO:
                correlated_tensor = scipy.ndimage.correlate(single_sample, self.weights[k, :, :, :], None, 'constant')
                single_kernel_output = correlated_tensor[num_channels//2]
                strided_output = single_kernel_output[::self.stride_shape[0], ::self.stride_shape[1]]
                self.output_tensor[i, k, :, :] = strided_output + self.bias[k]
        assert(self.output_tensor.shape == (num_samples, self.num_kernels, n_H, n_W))
        return self.output_tensor


if __name__ == '__main__':
    np.random.seed(1337)
    maps_in = 2
    bias = 1
    # kernel_shape = (3, 5, 8)
    # num_kernels = 4
    # input_shape = (3, 10, 14)
    # batch_size = 2
    conv = Conv((1, 1), (maps_in, 3, 3), 1)
    filter = (1. / 15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
    conv.weights = np.repeat(filter[None, ...], maps_in, axis=1)
    conv.bias = np.array([bias])
    input_tensor = np.random.random((1, maps_in, 10, 14))
    expected_output = bias
    for map_i in range(maps_in):
        expected_output = expected_output + gaussian_filter(input_tensor[0, map_i, :, :], 0.85, mode='constant',
                                                            cval=0.0, truncate=1.0)
    output_tensor = conv.forward(input_tensor).reshape((10, 14))
    difference = np.max(np.abs(expected_output - output_tensor) / maps_in)
    print(difference)


    # print('------Basic Info------')
    # # print(conv.bias)
    # print('shape of bias: ', conv.bias.shape)
    # print('shape of weights: ', conv.weights.shape)
    # print('shape of stride:', conv.stride_shape)
    # input_tensor = np.array(range(int(np.prod(input_shape) * batch_size)), dtype=float)
    # input_tensor = input_tensor.reshape(batch_size, *input_shape)
    # print('input size:', input_tensor.shape)
    # #
    # print('expected forward size:', (batch_size, num_kernels, *input_shape[1:]))
    # print('-----Forward-----')
    # output_tensor = conv.forward(input_tensor)
    # print('output size: ', output_tensor.shape)

    # print('------Forward Multi Channel:------')
    # for map_i in range(maps_in):
    #     expected_output = expected_output + gaussian_filter(input_tensor[0, map_i, :, :], 0.85, mode='constant',
    #                                                         cval=0.0, truncate=1.0)
    # print('expected output size: ', expected_output.shape)
    # print('expected output: ', expected_output)
    #
    # output_tensor = conv.forward(input_tensor).reshape((10, 14))
    # print('output size: ', output_tensor.shape)
    # print('output: ', output_tensor)
    #
    # # difference = np.max(np.abs(expected_output - output_tensor) / maps_in)
    # print(np.max(np.abs(expected_output - output_tensor) / maps_in))
    # # print(np.max(expected_output))