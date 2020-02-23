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

plt.style.use("seaborn-darkgrid")

# import datetime
import csv
import pandas as pd
import json

# import argparse
from pathlib import Path

# set paths
model_path = "saved_models/MSE/21-02-2020_17:47:13/CAE_mvtec_b12.h5"
parent_dir = str(Path(model_path).parent)
val_dir = os.path.join(parent_dir, "val_results")
test_dir = os.path.join(parent_dir, "test_results")

# load arrays
imgs_val_input = np.load(os.path.join(val_dir, "imgs_val_input.npy"))
imgs_val_pred = np.load(os.path.join(val_dir, "imgs_val_pred.npy"))
imgs_val_diff = np.load(os.path.join(val_dir, "imgs_val_diff.npy"))

imgs_test_input = np.load(os.path.join(test_dir, "imgs_test_input.npy"))
imgs_test_pred = np.load(os.path.join(test_dir, "imgs_test_pred.npy"))
imgs_test_diff = np.load(os.path.join(test_dir, "imgs_test_diff.npy"))

# load DataFrames
df_val = pd.read_pickle(os.path.join(val_dir, "df_val.pkl"))
df_test = pd.read_pickle(os.path.join(test_dir, "df_test.pkl"))


# flatten
imgs_val_diff_1d = imgs_val_diff.flatten()
imgs_test_diff_1d = imgs_test_diff.flatten()

# ===================== WITHOUT POST PROCESSING ============================
# plot histogram val
plt.hist(imgs_val_diff_1d, bins=500, density=True, stacked=True, alpha=0.5, label="val")

# plot histogram test
plt.hist(
    imgs_test_diff_1d, bins=500, density=True, stacked=True, alpha=0.5, label="test"
)

# plot pdf val
mu_val = imgs_val_diff_1d.mean()
sigma_val = imgs_val_diff_1d.std()
min_val = np.amin(imgs_val_diff_1d)
max_val = np.amax(imgs_val_diff_1d)

X_val = np.linspace(start=min_val, stop=max_val, num=1000, endpoint=True)
pdf_x_val = [scipy.stats.norm(mu_val, sigma_val).pdf(x) for x in X_val]
plt.plot(X_val, pdf_x_val, label="pdf_x_val")

# plot pdf test
mu_test = imgs_test_diff_1d.mean()
sigma_test = imgs_test_diff_1d.std()
min_test = np.amin(imgs_test_diff_1d)
max_test = np.amax(imgs_test_diff_1d)

X_test = np.linspace(start=min_test, stop=max_test, num=1000, endpoint=True)
pdf_x_test = [scipy.stats.norm(mu_test, sigma_test).pdf(x) for x in X_test]
plt.plot(X_test, pdf_x_test, label="pdf_x_test")

plt.title("pixel value distribution of val and test Resmaps")
plt.xlabel("pixel intensity")
plt.ylabel("probability")
plt.legend()
plt.show()

# ====================================================================

mean_mean_val = df_val["mean"].mean()
mean_std_val = df_val["std"].mean()

mean_mean_test = df_test["mean"].mean()
mean_std_test = df_test["std"].mean()

plt.imshow(imgs_val_diff[0])
np.amin(imgs_val_diff)
np.amax(imgs_val_diff)

img_test_diff = imgs_test_diff[68]
plt.imshow(img_test_diff)
plt.show()
# np.amin(imgs_test_diff)
# np.amax(imgs_test_diff)


# ===================== WITH THRESHOLDING ============================
X_val = imgs_val_diff.copy()
X_test = imgs_test_diff.copy()

threshold = 0.3

# 68


def threshold_array(X, threshold):
    X_th = X.copy()
    X_th[X_th < threshold] = 0
    return X_th


def create_images_from_array(X, dst_dict, filenames):
    for i in range(len(X)):
        image = X[i]
        plt.imshow(image)
        filename = "_".join(filenames[i].split("/"))
        filepath = os.path.join(dst_dict, filename)
        plt.savefig(filepath)


# threshold val images with 0
dst_dict_val = "/home/adnene33/Desktop/images_thresholded/val"
filenames_val = df_val["filenames"]
X_val_th0 = threshold_array(X_val, 0)
create_images_from_array(X_val_th0, dst_dict_val, filenames_val)

# threshold test images with 0
dst_dict_test = "/home/adnene33/Desktop/images_thresholded/test"
filenames_test = df_test["filenames"]
X_test_th0 = threshold_array(X_test, 0)
create_images_from_array(X_test_th0, dst_dict_test, filenames_test)

X_val_th0_1d = X_val_th0.flatten()
X_test_th0_1d = X_test_th0.flatten()

# plot
plt.hist(X_val_th0_1d, bins=100, density=True, stacked=True, alpha=0.5, label="val")
plt.hist(X_test_th0_1d, bins=100, density=True, stacked=True, alpha=0.5, label="test")
plt.legend()
plt.title("pixel value distribution of val and test Resmaps")
plt.xlabel("pixel intensity")
plt.ylabel("probability")
plt.show()

df_test[df_test["filenames"] == "crack/000.png"]

