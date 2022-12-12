import numpy as np
from matplotlib import pyplot as plt

# Daiqi LIU, om19arag
# Xingjian KANG, ev00ykob


class Checker:

    def __init__(self, resolution, tile_size):
        """
        :param resolution: integer, defines the number of pixels in each dimension
        :param tile_size: integer, defines the number of pixel an individual tile has in each dimension
                         output:store the pattern.
        """
        # TODO
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.ndarray([resolution, resolution])

    def draw(self):
        """
        create a checkerboard as ndarray
        :return: a copy of output, ndarray
        """
        # TODO
        # assume the resolution is 250, means we have 250x250 pixels, tile size is 25 pixels
        # 5 repetitions of the array below
        # |black white|
        # |white black|
        if self.resolution%(2*self.tile_size)!=0:
            print("error: values for resolution that are evenly dividable by 2· tile size.")
            quit()
        else:
            square_size = self.tile_size
            black = np.zeros([square_size, square_size])
            white = np.ones([square_size, square_size])
            '''
                first 2*2 unit:
            '''
            up = np.concatenate((black, white), axis=1)
            down = np.concatenate((white, black), axis=1)
            unit = np.concatenate((up, down), axis=0)
            repetition = self.resolution // (2 * self.tile_size)
            self.output = np.tile(unit, (repetition, repetition))
        return self.output.copy()

    def show(self):
        """
        :return: a grayscale image
        """
        # TODO
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Circle:

    def __init__(self, resolution, radius, position):
        """

        :param resolution: int, number of pixel in each dimension
        :param radius: int, radius of the circle
        :param position: tuple, contains x- and y-coordinate of the circle center
        """

        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.ndarray((resolution, resolution))

    def draw(self):
        # create the circle pattern
        background = np.ones((self.resolution, self.resolution))
        x_axis = np.linspace(0, self.resolution, self.resolution)
        y_axis = np.linspace(0, self.resolution, self.resolution)
        XX, YY = np.meshgrid(x_axis, y_axis)
        # set evenly spaced axis x/y
        # set pixels within the circle as 1
        background[(XX - self.position[0]) ** 2 + (YY - self.position[1]) ** 2 > self.radius ** 2] = 0
        self.output = background
        return self.output.copy()

    def show(self):
        # show the image of binary circle
        plt.axis("off")
        plt.imshow(self.output, cmap="gray")
        plt.show()


class Spectrum:

    def __init__(self, resolution):
        """

        :param resolution: integer, number of pixel in each dimension
        """

        self.resolution = resolution
        self.output = np.empty((resolution, resolution, 3))

    def draw(self):
        #
        spectrum = np.zeros([self.resolution, self.resolution, 3], dtype=np.uint8)
        # Green
        spectrum[:, :, 1] = np.linspace(0, 255, self.resolution)
        # In order to get the desired color in the lower left corner, perform dimension change processing
        spectrum = spectrum.swapaxes(0, 1)
        # BLUE
        spectrum[:, :, 2] = np.linspace(255, 0, self.resolution)
        # Red
        spectrum[:, :, 0] = np.linspace(0, 255, self.resolution)

        self.output = spectrum / 255

        # self_Spectrum=Spectrum(50)
        # self_Spectrum.draw()
        # self_Spectrum.show()

        return self.output.copy()

    def show(self):
        plt.figure()
        plt.title("Figure 3: RGB spectrum example.", y=-0.1)
        plt.axis("off")
        plt.imshow(self.output, "brg")
        plt.show()
