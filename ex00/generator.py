import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform

# Xingjian KANG, ev00ykob
# Daiqi LIU, om19arag

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.


class ImageGenerator:
    # TODO: implement constructor
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        """
        Define all members of your generator class object as global members here.These need to include:
            the batch size
            the image size
            flags for different augmentations and whether the data should be shuffled for each epoch
        Also depending on the size of your data-set you can consider loading all images into memory here already.
        The labels are stored in json format and can be directly loaded as dictionary.
        Note that the file names correspond to the dicts of the label dictionary.
        """
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.epoch = 0
        # Honestly, I still puzzled a lot about this part QAQ
        # incl. why the batch num starts from 1 rather than 0 and why minus 1 in current_epoch(),
        # I fixed this weird Epoch-Issue with an even more weird way.
        # I humbly ask my team-mate E for his understanding, cuz I really can not figure out why :)
        self.batch_num = 1
        self.data_num = np.arange(100)
        if shuffle:
            np.random.shuffle(self.data_num)
        if 100 % batch_size != 0:  # this part is used to deal with the weird batch size
            add = self.data_num[:(batch_size - (100 % batch_size))]  # re-use th image in the first batch
            self.data_num = np.concatenate((self.data_num, add), axis=0)
        self.data_set_size = self.data_num.shape[0]
        self.image_index = np.array_split(self.data_num, self.data_num.shape[0]/self.batch_size)

    def next(self):
        """
        This function creates a batch of images and corresponding labels and returns them.
        A "batch" of images just means a bunch, say 10 images that are forwarded at once.
        Note that your amount of total data might not be divisible without remainder with the batch_size.

        :return: [images] - ndarray of images, [labels] -
        """
        # print("Current Batch: ", gen.batch_num)
        data_path = r'C:\Users\Kosij\DeepLearningNg\22ws\ex0_1\data\\'
        with open('Labels.json', 'r') as fcc_file:
            fcc_data = json.load(fcc_file)
        images = []  # a batch of images
        labels = []  # an array with the corresponding labels
        img_numbers = self.image_index[self.batch_num - 1]
        for i in range(self.batch_size):
            new_image = np.load(data_path + str(img_numbers[i]) + '.npy')
            # Data augmentation
            if self.mirroring is True or self.rotation is True:
                new_image = self.augment(new_image)
            # Resizing the image
            new_image = skimage.transform.resize(new_image, self.image_size)
            images.append(new_image)
            labels.append(fcc_data[str(img_numbers[i])])
        # case when the index reaches dataset's end
        if (self.batch_num + 1) * self.batch_size >= self.data_set_size:
            self.batch_num = 0
            self.epoch += 1  # I hate this increment although
            if self.shuffle:
                np.random.shuffle(self.data_num)
        else:
            self.batch_num += 1
        return np.asarray(images), labels

    def augment(self, img):
        """
        use a flag to control the probability
        :param img: single image sample
        :return: the randomly transformed image, mirrored or rotated
        """
        # Mirroring
        if self.mirroring:
            if np.random.randint(2) == 1:
                img = np.flipud(img)
        # Rotation
        if self.rotation:
            if np.random.randint(2) == 1:
                img = np.rot90(img, np.random.randint(1, 4))
        return img

    def current_epoch(self):
        """
        :return: current epoch number
        """
        return self.epoch - 1

    def class_name(self, x):
        """
        take care of the python dict
        :param x: integer label
        :return: class name for a specific input, string
        """
        return self.class_dict[x]

    def show(self):
        """
        In order to verify that the generator creates batches as required, this functions calls next to get a
        batch of images and labels and visualizes it.
        """
        images, labels = self.next()
        col = 8
        row = self.batch_size // col
        for i in range(self.batch_size):
            plt.subplot(col, row, i+1)
            plt.title(self.class_name(labels[i]))
            plt.imshow(images[i], "gray")
            plt.axis("off")
        plt.show()


if __name__ == '__main__':
    file_path = "./data/"
    label_path = "./Labels.json"
    batch_size = 64
    image_size = (64, 64, 3)
    gen = ImageGenerator(file_path, label_path, batch_size, image_size, shuffle=True, rotation=True, mirroring=True)
    gen.show()
    print("Current Epoch: ", gen.current_epoch())
    print("--------------"
          "")
    gen.show()
    print("Current Epoch: ", gen.current_epoch())
    print("--------------"
          "")
    gen.show()
    print("Current Epoch: ", gen.current_epoch())
    print("--------------"
          "")
    gen.show()
    print("Current Epoch: ", gen.current_epoch())
    print("--------------"
          "")
    gen.show()
    print("Current Epoch: ", gen.current_epoch())
    print("--------------"
          "")
    gen.show()
    print("Current Epoch: ", gen.current_epoch())
    print("--------------"
          "")
