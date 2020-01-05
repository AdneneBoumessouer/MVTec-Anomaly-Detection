"""
This script builds a 4D Tensor of shape (batch_size x length x width x channels) from the good and defect MVTec test images contained 
in the test subdirectory of each object class
"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import os


def build_tensor_from_img_dir(DIR, resize=True):
    """Returns images contained in a directory as a tensor of format (batch_size x length x width x channels)"""

    # number of images in directory
    batch_size = len(
        [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
    )

    # get image resolution
    img = mpimg.imread(DIR + "/000.png")
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
        imgs = tf.image.resize(imgs, [256, 256]).numpy()

    # labels: good -> 0, defect -> 1
    if "good" in DIR.split("/"):
        labels = np.zeros(shape=batch_size)
    else:
        labels = np.ones(shape=batch_size)

    return imgs, labels


mvtec_path = "datasets/mvtec/"
class_names = next(os.walk(mvtec_path))[1]

class_DIR_dict = {}
class_TENSOR_dict = {}
class_LABEL_dict = {}

for class_name in class_names:
    class_DIR_dict[class_name] = [
        x[0] for x in os.walk(mvtec_path + class_name + "/test/")
    ][1:]
    class_TENSOR_dict[class_name] = []
    class_LABEL_dict[class_name] = []


# create a test set (tensor-images / labels)
for class_name in class_names:
    for DIR in class_DIR_dict[class_name]:
        tensor, labels = build_tensor_from_img_dir(DIR, resize=True)
        class_TENSOR_dict[class_name].append(tensor)
        class_LABEL_dict[class_name].append(labels)

for class_name in class_names:
    tensor = np.concatenate(class_TENSOR_dict[class_name], axis=0)
    class_TENSOR_dict[class_name] = tensor

    label = np.concatenate(class_LABEL_dict[class_name], axis=0)
    class_LABEL_dict[class_name] = label

X_test = np.concatenate(list(class_TENSOR_dict.values()), axis=0)
y_test = np.concatenate(list(class_LABEL_dict.values()), axis=0)


np.save("X_test", X_test)
np.save("y_test", y_test)
