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

import requests
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import datetime
import csv
import pandas as pd
import json

import argparse

# import computer vision functions
import cv2 as cv
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


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
        # areas_all.extend(areas)
        areas_all.append(areas)
    return images_labeled, areas_all


def main(args):
    model_path = args.path
    min_area = args.area

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

    # create a results directory if not existent
    model_dir_name = os.path.basename(str(Path(model_path).parent))
    now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    save_dir = os.path.join(os.getcwd(), "results", model_dir_name, "validation", now)
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
    # np.save(
    #     file=os.path.join(save_dir, "imgs_val_input.npy"),
    #     arr=imgs_val_input,
    #     allow_pickle=True,
    # )

    # retrieve image_names
    filenames = validation_generator.filenames

    # get reconstructed images (i.e predictions) on validation dataset
    imgs_val_pred = model.predict(imgs_val_input)
    # np.save(
    #     file=os.path.join(save_dir, "imgs_val_pred.npy"),
    #     arr=imgs_val_pred,
    #     allow_pickle=True,
    # )

    # compute residual maps on validation dataset
    resmaps_val = imgs_val_input - imgs_val_pred
    np.save(
        file=os.path.join(save_dir, "resmaps_val.npy"),
        arr=resmaps_val,
        allow_pickle=True,
    )

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
        areas_all_flat = [item for sublist in areas_all for item in sublist]
        areas_all_flat.sort(reverse=True)
        if min_area > areas_all_flat[0]:
            break

        threshold = threshold + 1
        print("current threshold = {}".format(threshold))

    # save threshold value and min_area
    val_results = {
        "threshold": str(threshold),
        "area": str(min_area),
    }
    with open(os.path.join(save_dir, "val_results.json"), "w") as json_file:
        json.dump(val_results, json_file)


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

# Example of command to initiate validation

# python3 validate.py -p saved_models/MSE/25-02-2020_08:54:06/CAE_mvtec2_b12.h5 -a 200
