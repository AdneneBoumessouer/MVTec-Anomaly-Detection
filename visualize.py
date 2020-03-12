import os
import sys

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import custom_loss_functions
import utils
import datetime

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


# import validation functions
from validate import threshold_images as threshold_images
from validate import label_images as label_images

# =========================================================================
# visualization functions


def plot_img_at_index(X, index):
    _, _, _, channels = X.shape
    fig = plt.figure()
    if channels == 1:
        plt.imshow(X[index, :, :, 0], cmap=plt.cm.gray)
    elif channels == 3:
        plt.imshow(X[index, :, :, 0])
    plt.show()


def plot_img(img):
    ndims = len(img.shape)
    fig = plt.figure()
    if ndims == 2:
        plt.imshow(img, cmap=plt.cm.gray)
    elif ndims == 3:
        _, _, channels = img.shape
        if channels == 3:
            plt.imshow(img)
        else:
            plt.imshow(img[:, :, 0], cmap=plt.cm.gray)
    plt.show()
    return fig


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


# =========================================================================

imgs_test_input = np.load(
    "results/mvtec/capsule/mvtec2/MSE/25-02-2020_08:54:06/test/th_28_a_50/imgs_test_input.npy",
    allow_pickle=True,
)

imgs_test_pred = np.load(
    "results/mvtec/capsule/mvtec2/MSE/25-02-2020_08:54:06/test/th_28_a_50/imgs_test_pred.npy",
    allow_pickle=True,
)

resmaps_test = np.load(
    "results/mvtec/capsule/mvtec2/MSE/25-02-2020_08:54:06/test/th_28_a_50/imgs_test_diff.npy",
    allow_pickle=True,
)

index = 68
plot_img(imgs_test_input[index])
plot_img(imgs_test_pred[index])
plot_img(resmaps_test[index])


# ---------------------------------------------

# Convert to 8-bit unsigned int for further processing
resmaps_test = img_as_ubyte(resmaps_test)

# ---------------------------------------------

threshold = 0
# threshold residual maps
resmaps_th = threshold_images(resmaps_test, threshold)
index = 68
img = resmaps_th[index]  # resmaps_th
plot_img(img)

# ---------------------------------------------

# compute anomalous regions
resmaps_labeled, areas_all = label_images(resmaps_th)
index = 68
img = resmaps_labeled[index]  # resmaps_th
plot_img(img)

# ---------------------------------------------


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


resmaps_labeled = label(resmaps_th)

index = 68

label_image = resmaps_labeled[index]
plt.imshow(label_image, cmap="nipy_spectral")

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(label_image)


for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 10:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)


# https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label

