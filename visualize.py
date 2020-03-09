import os
import sys

# import tensorflow as tf
# from tensorflow import keras
# import keras.backend as K
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


def main(args):
    # set paths
    model_path = args.path
    min_area = args.area
    parent_dir = str(Path(model_path).parent)
    val_dir = os.path.join(parent_dir, "val_results")

    # create a directory to save fine-tuning results (threshold)
    model_dir_name = os.path.basename(str(Path(model_path).parent))
    now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    save_dir = os.path.join(os.getcwd(), "results",
                            model_dir_name, "finetune", now)
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


if __name__ == "__main__":
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

    main(args)

# model_path = "saved_models/MSE/25-02-2020_08:54:06/CAE_mvtec2_b12.h5"
# min_area = 200

# # Histogramm to visualize the ResMap distribution
# fig = plt.figure(figsize=(8, 5))
# plt.hist(resmaps_val_1d, bins=100, density=True, stacked=True)
# plt.title("Validation ResMap pixel value distribution")
# plt.xlabel("pixel intensity")
# plt.ylabel("probability")
# plt.savefig(os.path.join(save_dir, "histogram_val.png"))

# # save three images
# fig, axarr = plt.subplots(3, 1, figsize=(5, 18))
# try:
#     axarr[0].imshow(imgs_val_input[0])
# except TypeError:
#     axarr[0].imshow(imgs_val_input[0, :, :, 0], cmap=plt.cm.gray)
# axarr[0].set_title("original defect-free val image")
# try:
#     axarr[1].imshow(imgs_val_pred[0])
# except TypeError:
#     axarr[1].imshow(imgs_val_pred[0, :, :, 0], cmap=plt.cm.gray)
# axarr[1].set_title("reconstruction defect-free val image")
# try:
#     axarr[2].imshow(resmaps_val[0])
# except TypeError:
#     axarr[2].imshow(resmaps_val[0, :, :, 0], cmap=plt.cm.gray)
# axarr[2].set_title("ResMap defect-free val image")
# fig.savefig(os.path.join(save_dir, "3_val_musketeers.png"))


# parser.add_argument(
#     "-i", "--image", type=str, required=True, metavar="", help="path to test image"
# )

# python3 validate.py -p saved_models/MSE/21-02-2020_17:47:13/CAE_mvtec_b12.h5

# python3 test.py -p saved_models/MSE/21-02-2020_17:47:13/CAE_mvtec_b12.h5 -i "poke/000.png"
