import os
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import custom_loss_functions
import utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy import expand_dims
import scipy.stats

import requests
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import datetime
import csv
import pandas as pd
import json

import argparse

# import validation functions
from validate import threshold_images as threshold_images
from validate import label_images as label_images

from skimage.util import img_as_ubyte

# =========================================================================


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


# =========================================================================


def main(args):
    model_path = args.path

    # load model, setup and history
    model, setup, history = utils.load_model_HDF5(model_path)

    # data setup
    directory = setup["data_setup"]["directory"]
    val_data_dir = os.path.join(directory, "train")
    nb_training_images = setup["data_setup"]["nb_training_images"]
    nb_validation_images = setup["data_setup"]["nb_validation_images"]

    # preprocessing_setup
    rescale = setup["preprocessing_setup"]["rescale"]
    shape = setup["preprocessing_setup"]["shape"]
    preprocessing = setup["preprocessing_setup"]["preprocessing"]

    # train_setup
    color_mode = setup["train_setup"]["color_mode"]
    learning_rate = setup["train_setup"]["learning_rate"]
    decay = setup["train_setup"]["decay"]
    epochs_trained = setup["train_setup"]["epochs_trained"]
    nb_training_images_aug = setup["train_setup"]["nb_training_images_aug"]
    epochs = setup["train_setup"]["epochs"]
    batch_size = setup["train_setup"]["batch_size"]
    channels = setup["train_setup"]["channels"]
    validation_split = setup["train_setup"]["validation_split"]
    architecture = setup["train_setup"]["architecture"]
    loss = setup["train_setup"]["loss"]

    tag = setup["tag"]

    # create directory to save results
    model_dir_name = os.path.basename(str(Path(model_path).parent))
    save_dir = os.path.join(os.getcwd(), "results", model_dir_name, "vasualize")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Plot and save loss and val_loss
    plot = pd.DataFrame(history[["loss", "val_loss"]]).plot(figsize=(8, 5))
    fig = plot.get_figure()
    fig.savefig(os.path.join(save_dir, "train_val_losses.png"))

    # This will do preprocessing
    if architecture in ["mvtec", "mvtec2"]:
        preprocessing_function = None
    elif architecture == "resnet":
        preprocessing_function = keras.applications.inception_resnet_v2.preprocess_input
    elif architecture == "nasnet":
        preprocessing_function = keras.applications.nasnet.preprocess_input

    # same preprocessing as in training
    validation_datagen = ImageDataGenerator(
        rescale=rescale,
        data_format="channels_last",
        validation_split=validation_split,
        preprocessing_function=preprocessing_function,
    )

    # retrieve preprocessed validation images as a numpy array
    validation_generator = validation_datagen.flow_from_directory(
        directory=val_data_dir,
        target_size=shape,
        color_mode=color_mode,
        batch_size=nb_validation_images,
        shuffle=False,
        class_mode="input",
        subset="validation",
    )
    imgs_val_input = validation_generator.next()[0]
    np.save(
        file=os.path.join(save_dir, "imgs_val_input.npy"),
        arr=imgs_val_input,
        allow_pickle=True,
    )

    # retrieve image_names
    filenames = validation_generator.filenames

    # get reconstructed images (i.e predictions) on validation dataset
    imgs_val_pred = model.predict(imgs_val_input)
    np.save(
        file=os.path.join(save_dir, "imgs_val_pred.npy"),
        arr=imgs_val_pred,
        allow_pickle=True,
    )

    # compute residual maps on validation dataset
    resmaps_val = imgs_val_input - imgs_val_pred
    np.save(
        file=os.path.join(save_dir, "resmaps_val.npy"),
        arr=resmaps_val,
        allow_pickle=True,
    )

    # COMMENT
    resmaps_val = np.load(
        "results/25-02-2020_08:54:06/validation/02-03-2020_11:13:25/resmaps_val.npy",
        allow_pickle=True,
    )

    # Convert to 8-bit unsigned int
    # (unnecessary if working exclusively with scikit image, see .img_as_float())
    resmaps_val = img_as_ubyte(resmaps_val)

    nb_regions = []
    mean_area = []
    std_area = []

    for threshold in range(np.amax(resmaps_val)):
        # threshold residual maps
        resmaps_th = threshold_images(resmaps_val, threshold)

        # compute connected components
        resmaps_labeled, areas_all = label_images(resmaps_th)

        # check if area of largest anomalous region is below the minimum area
        areas_all_flat = [item for sublist in areas_all for item in sublist]
        nb_regions.append(len(areas_all_flat))
        mean_area.append(np.mean(areas_all_flat))
        std_area.append(np.std(areas_all_flat))

        print("current threshold = {}".format(threshold))

    df_out = pd.DataFrame.from_dict(
        {
            "threshold": list(range(255)),
            "nb_regions": nb_regions,
            "mean_area": mean_area,
            "std_area": std_area,
        }
    )
    df_out.plot(x="threshold", y=["mean_area", "std_area"])

    df_out.plot(x="threshold", y=["nb_regions"])

    # ===================================================================

    # COMMENT
    resmaps_val = np.load(
        "results/25-02-2020_08:54:06/validation/02-03-2020_11:13:25/resmaps_val.npy",
        allow_pickle=True,
    )

    # Convert to 8-bit unsigned int
    # (unnecessary if working exclusively with scikit image, see .img_as_float())
    resmaps_val = img_as_ubyte(resmaps_val)
    counts = []

    for threshold in range(np.amax(resmaps_val)):
        # threshold residual maps
        resmaps_th = threshold_images(resmaps_val, threshold)

        # compute connected components
        resmaps_labeled, areas_all = label_images(resmaps_th)

        # flatten area
        areas_all_flat = [item for sublist in areas_all for item in sublist]

        if threshold == 0:
            areas_all_flat.sort(reverse=True)
            min_area = areas_all_flat[-1]
            max_area = areas_all_flat[0]

        count, bins, ignored = plt.hist(
            areas_all_flat,
            range=(min_area, max_area/4),
            bins=200, # max_area - min_area
            density=False,
        )
        # count, bins, ignored = plt.hist(areas_all_flat, bins="auto", density=False)
        # counts.append(count)
        break
    
    # just plot histogramm for threshold == 0, pick a good value for area and run validate.py to get best threshold, then run test.py

        plt.plot()

    fig = plt.figure()
    plt.plot()

    # ===================================================================

    # Histogramm to visualize the ResMap distribution
    fig = plt.figure(figsize=(8, 5))
    plt.hist(resmaps_val_1d, bins=100, density=True, stacked=True)
    plt.title("Validation ResMap pixel value distribution")
    plt.xlabel("pixel intensity")
    plt.ylabel("probability")
    plt.savefig(os.path.join(save_dir, "histogram_val.png"))

    # save three images
    fig, axarr = plt.subplots(3, 1, figsize=(5, 18))
    try:
        axarr[0].imshow(imgs_val_input[0])
    except TypeError:
        axarr[0].imshow(imgs_val_input[0, :, :, 0], cmap=plt.cm.gray)
    axarr[0].set_title("original defect-free val image")
    try:
        axarr[1].imshow(imgs_val_pred[0])
    except TypeError:
        axarr[1].imshow(imgs_val_pred[0, :, :, 0], cmap=plt.cm.gray)
    axarr[1].set_title("reconstruction defect-free val image")
    try:
        axarr[2].imshow(resmaps_val[0])
    except TypeError:
        axarr[2].imshow(resmaps_val[0, :, :, 0], cmap=plt.cm.gray)
    axarr[2].set_title("ResMap defect-free val image")
    fig.savefig(os.path.join(save_dir, "3_val_musketeers.png"))


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )
    args = parser.parse_args()

    main(args)

model_path = "saved_models/MSE/25-02-2020_08:54:06/CAE_mvtec2_b12.h5"
resmaps_val = np.load(
    "results/25-02-2020_08:54:06/validation/02-03-2020_11:13:25/resmaps_val.npy",
    allow_pickle=True,
)

# parser.add_argument(
#     "-i", "--image", type=str, required=True, metavar="", help="path to test image"
# )

# python3 validate.py -p saved_models/MSE/21-02-2020_17:47:13/CAE_mvtec_b12.h5

# python3 test.py -p saved_models/MSE/21-02-2020_17:47:13/CAE_mvtec_b12.h5 -i "poke/000.png"
