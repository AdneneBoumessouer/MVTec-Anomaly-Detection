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


def plot_img(imgs, index, save=False):
    """Takes an image tensor and plots one image according to index"""
    if imgs.shape[-1] == 3:
        plt.imshow(imgs[index])
    else:
        plt.imshow(imgs[index, :, :, 0], cmap=plt.cm.gray)
    plt.show()


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
    epochs_trained = setup["train_setup"]["epochs_trained"]
    nb_training_images_aug = setup["train_setup"]["nb_training_images_aug"]
    epochs = setup["train_setup"]["epochs"]
    batch_size = setup["train_setup"]["batch_size"]
    channels = setup["train_setup"]["channels"]
    validation_split = setup["train_setup"]["validation_split"]
    architecture = setup["train_setup"]["architecture"]
    loss = setup["train_setup"]["loss"]

    comment = setup["comment"]

    # create directory to save results
    parent_dir = str(Path(model_path).parent)
    save_dir = os.path.join(parent_dir, "val_results")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Plot and save loss and val_loss
    plot = pd.DataFrame(history[["loss", "val_loss"]]).plot(figsize=(8, 5))
    fig = plot.get_figure()
    fig.savefig(os.path.join(save_dir, "losses.png"))

    # This will do preprocessing
    if architecture == "mvtec":
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

    # retrieve preprocessed input images as a numpy array
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

    # get reconstructed images (predictions)
    imgs_val_pred = model.predict(imgs_val_input)

    # compute residual maps
    imgs_val_diff = imgs_val_input - imgs_val_pred

    # determine threshold
    imgs_val_diff_1d = imgs_val_diff.flatten()
    mean = np.mean(imgs_val_diff_1d)
    std = np.std(imgs_val_diff_1d)

    # k = 1.65 # confidence 90%
    # k = 1.96 # confidence 95%
    k = 2.58  # confidence 99%
    # k = 3.00 # confidence 99.73%
    # k = 3.30 # confidence 99.90%

    threshold = mean + k*std

    # save validation results
    val_results = {
        "mean": str(mean),
        "std": str(std),
        "threshold": str(threshold),
        "pixel_max_value": str(np.amax(imgs_val_pred)),
        "pixel_min_value": str(np.amin(imgs_val_pred)),
    }

    with open(os.path.join(save_dir, "val_results.json"), "w") as json_file:
        json.dump(val_results, json_file)

    # save images
    plt.imsave(os.path.join(save_dir, "img_input.png"), imgs_val_input[0])
    plt.imsave(os.path.join(save_dir, "img_pred.png"), imgs_val_pred[0])
    plt.imsave(os.path.join(save_dir, "img_diff.png"), imgs_val_diff[0])


# create parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
)
args = parser.parse_args()

if __name__ == "__main__":
    main(args)

# model_path = "saved_models/MSE/14-02-2020_15:10:24/CAE_mvtec_b24.h5"

# model, setup, _ = utils.load_SavedModel(model_path)

# python3 validate.py -p saved_models/MSE/17-02-2020_18:14:52/CAE_mvtec_b12.h5
