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
from skimage.filters import median
import cv2


# import validation functions
from validate import threshold_images as threshold_images
from validate import label_images as label_images

# =========================================================================
# visualization functions


def plot_img_at_index(X, index):
    plt.style.use("default")
    _, _, _, channels = X.shape
    fig = plt.figure()
    if channels == 1:
        plt.imshow(X[index, :, :, 0], cmap=plt.cm.gray)
    elif channels == 3:
        plt.imshow(X[index, :, :, 0])
    plt.show()


def plot_img(img, title=None):
    plt.style.use("default")
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
    plt.title(title)
    plt.show()
    # return fig


def hist_image(img, bins=400, range_x=None, title=None):
    # plot histogram
    plt.style.use("seaborn-darkgrid")
    img_1d = img.flatten()
    plt.figure()
    plt.hist(img_1d, bins=bins, density=True, stacked=True, label="image histogram")
    
    # print descriptive values
    mu = img_1d.mean()
    sigma = img_1d.std()
    minimum = np.amin(img_1d)
    maximum = np.amax(img_1d)
    print("{} \n mu = {} \n sigma = {} \n min = {}\n max = {}".format(title, mu, sigma, minimum, maximum))
    
    # compute and plot probability distribution function    
    X = np.linspace(start=minimum, stop=maximum, num=200, endpoint=True)
    pdf_x = [scipy.stats.norm(mu, sigma).pdf(x) for x in X]
    plt.plot(X, pdf_x, label="pixel distribution")
    plt.title(title)
    if range_x is not None:
        plt.xlim(range_x[0], range_x[1])
    plt.legend()
    plt.show()
    
def hist_image_uint8(img, range_x=None, title=None):
    # plot histogram    
    plt.style.use("seaborn-darkgrid")
    plt.figure()
    # compute bin edges
    img_1d = img.flatten()
    d = np.diff(np.unique(img_1d)).min()
    left_of_first_bin = img_1d.min() - float(d)/2
    right_of_last_bin = img_1d.max() + float(d)/2  
    bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)
    plt.hist(img_1d, bins=bins, density=True, stacked=True, label="pixel value histogram")
    
    # print descriptive values
    mu = img_1d.mean()
    sigma = img_1d.std()
    minimum = np.amin(img_1d)
    maximum = np.amax(img_1d)
    print("{} \n mu = {} \n sigma = {} \n min = {}\n max = {}".format(title, mu, sigma, minimum, maximum))
    
    # compute and plot probability distribution function    
    X = np.linspace(start=minimum, stop=maximum, num=200, endpoint=True)
    pdf_x = [scipy.stats.norm(mu, sigma).pdf(x) for x in X]
    plt.plot(X, pdf_x, label="pixel distribution")
    plt.title(title)
    if range_x is not None:
        plt.xlim(range_x[0], range_x[1])
    plt.legend()
    plt.show()



# =========================================================================

# VALIDATION
# imgs_val_input = np.load("results/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/finetune/imgs_val_input.npy", allow_pickle=True)
# imgs_val_pred = np.load("results/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/finetune/imgs_val_pred.npy", allow_pickle=True)
resmaps_val = np.load("results/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/finetune/resmaps_val.npy", allow_pickle=True)

index_val = 0
# plot_img(imgs_val_input[index_val])
# plot_img(imgs_val_pred[index_val])
plot_img(resmaps_val[index_val])



# TEST
# imgs_test_input = np.load("results/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/test/th_32_a_70/imgs_test_input.npy", allow_pickle=True)
# imgs_test_pred = np.load("results/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/test/th_32_a_70/imgs_test_pred.npy", allow_pickle=True)
resmaps_test = np.load("results/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/test/th_32_a_70/resmaps_test.npy", allow_pickle=True)

index_test = 68
# plot_img(imgs_test_input[index_test])
# plot_img(imgs_test_pred[index_test])
plot_img(resmaps_test[index_test])

# ============================= PLAYGROUND ====================================

# inspect pixel value distribution of val and test -----------------------
hist_image(resmaps_val, title="resmaps_val")
hist_image(resmaps_test, title="resmaps_test")
# hist_image(resmaps_val, range_x=(-0.6, 0.6), title="resmaps_val")
# hist_image(resmaps_test, range_x=(-0.6, 0.6), title="resmaps_test")


# scale pixel values in [0,1] and plot --------------------------------------
resmaps_val_scaled = (resmaps_val + 1)/2
index_val = 0
plot_img(resmaps_val[index_val])
plot_img(resmaps_val_scaled[index_val])
hist_image(resmaps_val_scaled, title="resmaps_val_scaled")

resmaps_test_scaled = (resmaps_test + 1)/2
index_test = 68
plot_img(resmaps_test[index_test])
plot_img(resmaps_test_scaled[index_test])
hist_image(resmaps_test_scaled, title="resmaps_test_scaled")


# Convert to 8-bit unsigned int for further processing -----------------------
resmaps_val_uint8 = img_as_ubyte(resmaps_val_scaled)
index_val = 0
plot_img(resmaps_val_uint8[index_val])
hist_image_uint8(resmaps_val_uint8, title="resmaps_val_uint8")
hist_image_uint8(resmaps_val_uint8, range_x=(0,255), title="resmaps_val_uint8")

resmaps_test_uint8 = img_as_ubyte(resmaps_test_scaled)
index_test = 68
plot_img(resmaps_test_uint8[index_test])
hist_image_uint8(resmaps_test_uint8, title="resmaps_test_uint8")
hist_image_uint8(resmaps_test_uint8, range_x=(0,255), title="resmaps_test_uint8")

bins_soll = hist_image_uint8(resmaps_test_uint8, range_x=(0,255), title="resmaps_test_uint8") # REMOVE

# threshold residual maps and plot results ----------------------------------
threshold = 140
resmaps_val_th_uint8 = threshold_images(resmaps_val_uint8, threshold)
index_val = 5
plot_img(resmaps_val_th_uint8[index_val], title="resmaps_test_th_uint8[{}]\n threshold = {}".format(index_val, threshold))


threshold = 141
resmaps_test_th_uint8 = threshold_images(resmaps_test_uint8, threshold)
index_test = 68
plot_img(resmaps_test_th_uint8[index_test], title="resmaps_test_th_uint8[{}]\n threshold = {}".format(index_test, threshold))


# investigate shrinking down the threshold interval from [pixel_min, pixel_max] to a smaller interval that contains relevant thresholds ------------------

# flatten resmaps
resmaps_val_uint8_1d = resmaps_val_uint8.flatten()

# compute descriptive values
mu = resmaps_val_uint8_1d.mean()
sigma = resmaps_val_uint8_1d.std()
minimum = np.amin(resmaps_val_uint8_1d) # pixel_min
maximum = np.amax(resmaps_val_uint8_1d) # pixel_max

# compute pdf and cdf
X = np.linspace(start=minimum, stop=maximum, num=200, endpoint=True)
pdf_x = [scipy.stats.norm(mu, sigma).pdf(x) for x in X]
cdf_x = [scipy.stats.norm(mu, sigma).cdf(x) for x in X]

# plot histogram, pdf and cdf
hist_image_uint8(resmaps_val_uint8, title="resmaps_val_uint8")
plt.plot(X, cdf_x, label="cdf")

# define relevant threshold ranges
cdf_relevant_range = np.arange(start=0.9, stop=1, step=0.0001) # start=0.9, stop=1, step=0.0001
# start=0.95, stop=1, step=0.00001 => th_min = 125, th_max = 149
th_relevant_range = [scipy.stats.norm(mu, sigma).ppf(cdf_value) for cdf_value in cdf_relevant_range]
th_min = int(round(np.amin(np.array(th_relevant_range)), 1)) - 1 # 124
th_max = int(round(np.amax(np.array(th_relevant_range)), 1)) + 1 # 135

# investigate resulting thresholded images with different values of selected relevant thresholds
threshold = 124
resmaps_val_th_uint8 = threshold_images(resmaps_val_uint8, threshold)
index_val = 5
plot_img(resmaps_val_th_uint8[index_val], title="resmaps_val_th_uint8[{}]\n threshold = {}".format(index_val, threshold))

# threshold = 124
resmaps_test_th_uint8 = threshold_images(resmaps_test_uint8, threshold)
index_test = 68
plot_img(resmaps_test_th_uint8[index_test], title="resmaps_val_th_uint8[{}]\n threshold = {}".format(index_test, threshold))



# investigate smoothing image with median filter ------------------------------------------------------------

def median_filter(images, kernel_size=5):
    """
    Filter images according to Median Filtering.
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html

    Parameters
    ----------
    images : array of uint8
        Thresholded residual maps.
    kernel_size : int, optional
        Size of the kernel window. The default is 5.

    Returns
    -------
    images_filtered : array of uint8
        Filtered images.

    """
    images = images.astype("uint8")
    images_filtered = np.zeros(shape=images.shape, dtype="uint8")
    for i, image in enumerate(images):
        image_filtered = cv2.medianBlur(image, kernel_size)
        images_filtered[i] = image_filtered
    return images_filtered


# VALIDATION
threshold = 127
index_val = 5
resmaps_val_th_uint8 = threshold_images(resmaps_val_uint8, threshold)

# plot before median filter
plot_img(resmaps_val_th_uint8[index_val], title="resmaps_test_th_uint8[{}]\n threshold = {}".format(index_val, threshold))
# plot after median filter
resmaps_val_th_uint8_fil = median_filter(resmaps_val_th_uint8)
plot_img(resmaps_val_th_uint8_fil[index_val], title="resmaps_val_th_uint8_fil[{}]\n threshold = {}".format(index_val, threshold))


# TEST
threshold = 124
index_test = 68
resmaps_test_th_uint8 = threshold_images(resmaps_test_uint8, threshold)

# plot before median filter
plot_img(resmaps_test_th_uint8[index_test], title="resmaps_test_th_uint8[{}]\n threshold = {}".format(index_test, threshold))
# plot after median filter
resmaps_test_th_uint8_fil = median_filter(resmaps_test_th_uint8)
plot_img(resmaps_test_th_uint8_fil[index_test], title="resmaps_test_th_uint8_fil[{}]\n threshold = {}".format(index_test, threshold)) 




    

# =============================================================================
# =============================================================================



# =============================================================================
# # investigate applying gaussian filter ----------------------------------
# threshold = 120
# resmaps_val_th_uint8 = threshold_images(resmaps_val_uint8, threshold)
# index_val = 5
# plot_img(resmaps_val_th_uint8[index_val], title="resmaps_test_th_uint8[{}]\n threshold = {}".format(index_val, threshold))
# =============================================================================



# =============================================================================
# # determine offset to previous method (no rescaling)

# resmaps_test_old_uint8 = img_as_ubyte(resmaps_test)
# hist_image_uint8(resmaps_test_old_uint8, range_x=(0,255), title="resmaps_test_old_uint8")
# threshold = 28
# # resmaps_test_th = threshold_images(resmaps_test, threshold)
# resmaps_test_old_th_uint8 = threshold_images(resmaps_test_old_uint8, threshold)
# index_test = 68
# plot_img(resmaps_test_old_th_uint8[index_test], title="resmaps_test_old_th_uint8[{}]\n threshold = {}".format(index_test, threshold))
# # OFFSET = 113
# =============================================================================


# =============================================================================
# =============================================================================

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




# # OLD PATHS 
# imgs_test_input = np.load("results/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/test/th_28_a_50/imgs_test_input.npy", allow_pickle=True)
# imgs_test_pred = np.load("results/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/test/th_28_a_50/imgs_test_pred.npy", allow_pickle=True)
# resmaps_test = np.load("results/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/test/th_28_a_50/imgs_test_diff.npy", allow_pickle=True)