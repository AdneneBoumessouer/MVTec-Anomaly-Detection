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
    images_th = np.zeros(shape=images.shape)
    for i, image in enumerate(images):
        image_th = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
        images_th[i] = image_th
    return images_th


def label_images(images):
    """
    Segments images into connected components (regions).
    Returns segmented images and a list containing information about the regions.
    """
    images_labeled = np.zeros(shape=images.shape)
    regions_all = []
    for i, image in enumerate(images):
        image_labeled = label(image)
        images_labeled[i] = image_labeled
        regions_all.append(regionprops(image_labeled))
    return images_labeled, regions_all


def check_regions(regions_all, min_area):
    """Checks if there is at least one connected component (region) that is 
    larger than the user defined min_area"""
    for regions in regions_all:
        for region in regions:
            if region.area > min_area:
                return True
    return False


def main(args):
    # set paths
    model_path = args.path
    parent_dir = str(Path(model_path).parent)
    val_dir = os.path.join(parent_dir, "val_results")

    # create a directory to save fine-tuning results (threshold)
    save_dir = os.path.join(parent_dir, "fine_tune")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # load arrays
    resmap_val = np.load(os.path.join(val_dir, "imgs_val_diff.npy"))
    # resmap_test = np.load(os.path.join(test_dir, "imgs_test_diff.npy"))

    min_area = 20
    threshold = 0

    while True:
        # threshold residual maps
        resmaps_th = threshold_images(resmap_val, threshold)

        # compute connected components
        resmaps_labeled, regions_all = label_images(resmaps_th)

        # check if connected componennts exceed minimum defect area
        if check_regions(regions_all, min_area) == False:
            # save threshold value
            fine_tune = {
                "threshold": str(threshold),
            }
            with open(os.path.join(save_dir, "fine_tune.json"), "w") as json_file:
                json.dump(fine_tune, json_file)
            # stop
            break

        threshold = threshold + 1


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

