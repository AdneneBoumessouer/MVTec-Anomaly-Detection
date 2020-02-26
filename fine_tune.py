import os
import sys

# import tensorflow as tf
# from tensorflow import keras
# import keras.backend as K
import custom_loss_functions
import utils

# from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.stats
from numpy import expand_dims


# import requests
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# plt.style.use("seaborn-darkgrid")

# import datetime
import csv
import pandas as pd
import json

import argparse
from pathlib import Path

# import computer vision functions
import cv2 as cv
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


def plot_img_at_index(X, index):
    _, _, _, channels = X.shape
    plt.figure()
    if channels == 1:
        plt.imshow(X[index, :, :, 0], cmap=plt.cm.gray)
    elif channels == 3:
        plt.imshow(X[index, :, :, 0])
    plt.show()


def plot_img(img):
    ndims = len(img.shape)
    plt.figure()
    if ndims == 2:
        plt.imshow(img, cmap=plt.cm.gray)
    elif ndims == 3:
        _, _, channels = img.shape
        if channels == 3:
            plt.imshow(img)
        else:
            plt.imshow(img[:, :, 0], cmap=plt.cm.gray)
    plt.show()


def hist_image(img):
    img_1d = img.flatten()
    plt.figure()
    plt.hist(img_1d, bins=200, density=True, stacked=True, label="image histogram")
    # plot pdf
    mu = img_1d.mean()
    sigma = img_1d.std()
    minimum = np.amin(img_1d)
    maximum = np.amax(img_1d)
    X = np.linspace(start=minimum, stop=maximum, num=400, endpoint=True)
    pdf_x = [scipy.stats.norm(mu, sigma).pdf(x) for x in X]
    plt.plot(X, pdf_x, label="pixel distribution")
    plt.legend()
    plt.show()


def threshold_images(images, threshold):
    """
    All pixel values < threshold  ==> 0, else ==> 255
    """
    images_th = np.zeros(shape=images.shape[:-1])
    for i, image in enumerate(images):
        image_th = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)[1]
        images_th[i] = image_th
    return images_th


def label_images(images):
    """
    Segments images into images of connected components (anomalous regions).
    Returns segmented images and a list containing their areas sorted 
    in descending order. 
    """
    images_labeled = np.zeros(shape=images.shape)
    areas_all = []
    for i, image in enumerate(images):
        # segment current image in connected components
        image_labeled = label(image)
        images_labeled[i] = image_labeled
        # compute areas of anomalous regions in the current image
        regions = regionprops(image_labeled)
        areas = [region.area for region in regions]
        areas_all.extend(areas)
    areas_all.sort(reverse=True)
    return images_labeled, areas_all


def main(args):
    # set paths
    model_path = args.path
    min_area = args.area
    parent_dir = str(Path(model_path).parent)
    val_dir = os.path.join(parent_dir, "val_results")

    # create a directory to save fine-tuning results (threshold)
    save_dir = os.path.join(parent_dir, "fine_tune")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # load arrays
    resmaps_val = np.load(os.path.join(val_dir, "imgs_val_diff.npy"))

    # Convert to 8-bit unsigned int
    # (unnecessary if working exclusively with scikit image, see .img_as_float())
    resmaps_val = img_as_ubyte(resmaps_val)

    threshold = 0

    while True:
        # threshold residual maps
        resmaps_th = threshold_images(resmaps_val, threshold)

        # compute connected components
        resmaps_labeled, areas_all = label_images(resmaps_th)

        # check if area of largest anomalous region is below the minimum area
        if min_area > areas_all[0]:
            # save threshold value
            fine_tune = {
                "threshold": str(threshold),
            }
            with open(os.path.join(save_dir, "fine_tune.json"), "w") as json_file:
                json.dump(fine_tune, json_file)
            # stop
            break

        threshold = threshold + 1
        print(threshold)


# create parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
)
parser.add_argument(
    "-a",
    "--area",
    type=int,
    required=True,
    metavar="",
    help="minimum area for a connected component",
)
args = parser.parse_args()

if __name__ == "__main__":
    main(args)

model_path = "saved_models/MSE/25-02-2020_08:54:06/CAE_mvtec2_b12.h5"
min_area = 200
