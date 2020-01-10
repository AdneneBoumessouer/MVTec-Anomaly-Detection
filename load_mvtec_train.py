"""
This script builds a 4D Tensor of shape (batch_size x length x width x channels) from the good and defect MVTec train images contained 
in the train subdirectory of each object class
"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import os


def build_tensor_from_img_dir(DIR, resize=True):
    """Transforms images contained in a directory in a tensor of format (batch_size x length x width x channels)"""

    # number of images in directory
    batch_size = len(
        [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
    )

    # get image resolution
    img = mpimg.imread(DIR + "000.png")
    length, width = img.shape[0], img.shape[1]

    # get image float type
    float_type = img.dtype

    # triplicate if images are gray_scaled
    triplicate = False
    if len(img.shape) == 2 or img.shape[2] == 1:
        triplicate = True

    # initialize
    imgs = np.zeros(shape=(batch_size, length, width, 3), dtype=float_type)

    # read images
    list_img = list(os.walk(DIR))[0][2]
    list_img.sort()
    for i, img_i in enumerate(list_img):
        img = mpimg.imread(DIR + "/" + img_i)
        if triplicate:
            img = np.repeat(img[:, :, np.newaxis], 3, -1)
        imgs[i] = img

    # resize to fit CAE input shape of 256 x 256 x 3
    if resize:
        images_resized = tf.image.resize(imgs, [256, 256]).numpy()
        return images_resized

    return imgs


mvtec_path = "datasets/mvtec"
# mvtec_path = "/home/adnene33/Documents/WS1920/Automatisierungstechnisches_Projekt/prototype/prototype_1/dataset/mvtec"
class_names = next(os.walk(mvtec_path))[1]

class_DIR_dict = {}
class_TENSOR_dict = {}

# from os import listdir
# from os.path import isfile, join


for class_name in class_names:
    class_DIR_dict[class_name] = list(
        os.walk(mvtec_path + "/" + class_name + "/train/good/")  # added '/'
    )[0][0]
    # class_DIR_dict[class_name] = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    class_TENSOR_dict[class_name] = []

# create a training set for the autoencoder
for class_name in class_names:
    DIR = class_DIR_dict[class_name]
    class_TENSOR_dict[class_name] = build_tensor_from_img_dir(DIR, resize=True)

X_train = np.concatenate(list(class_TENSOR_dict.values()), axis=0)

np.save("X_train", X_train)
# X_train.shape
