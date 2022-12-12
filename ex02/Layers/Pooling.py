import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    # TODO
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.Collection_of_masks = {}  # Need this later for storing the max values positions

    # TODO: returns a tensor that serves as the input tensor for the next layer
    # NOTE:  implemented only for the 2D case.
    def forward(self, input_tensor):
        self.input_tensor=input_tensor
        self.batch_size, self.channel, self.height_input, self.width_input = input_tensor.shape
        height_pooling, weight_pooling = self.pooling_shape
        # TODO: Move the kernel in a vertical direction
        y_times = 1 + (self.height_input - height_pooling) // self.stride_shape[0]
        # TODO:Move the kernel in a horizontal direction
        x_times = 1 + (self.width_input - weight_pooling) // self.stride_shape[1]
        output_tensor = np.zeros((self.batch_size, self.channel, y_times, x_times))
        for b in range(self.batch_size):
            for c in range(self.channel):
                for i in range(y_times):
                    for j in range(x_times):
                        # Select the range of the pooling kernel action in input_tensor
                        y_start = i * self.stride_shape[0]
                        y_end = y_start + height_pooling
                        x_start = j * self.stride_shape[1]
                        x_end = x_start + weight_pooling
                        # range_operations = self.input_tensor[:, :, y_start:y_end, x_start:x_end]
                        output_tensor[b,c,i,j]=np.max(self.input_tensor[b, c, y_start:y_end, x_start:x_end])
        return output_tensor

    def create_mask_from_window(self, x):
        """
        TODO: This function creates a mask matrix to hold the location of the maximum :1-->max;
        """
        mask = x == np.max(x)
        return mask

    def backward(self, error_tensor):
        prev_error_tensor = np.zeros(self.input_tensor.shape)
        batch, channel, y_out, x_out = error_tensor.shape
        y_pool, x_pool = self.pooling_shape
        for b in range(batch):
            single_sample=self.input_tensor[b]
            for c in range(channel):
                for i in range(y_out):
                    for j in range(x_out):
                        # get the index in the region i,j where the value is the maximum
                        y_start = i * self.stride_shape[0]
                        y_end = y_start + y_pool
                        x_start = j * self.stride_shape[1]
                        x_end = x_start + x_pool
                        # start slicing
                        sample_slice=single_sample[c, y_start:y_end, x_start:x_end]
                        # Get the maximum position of 1 in the mask matrix
                        mask = self.create_mask_from_window(sample_slice)
                        # Multiply the local gradient by the incoming gradient following the chain rule.
                        prev_error_tensor[b, c, y_start:y_end, x_start:x_end] += np.multiply(mask, error_tensor[b,c,i,j])

        return prev_error_tensor