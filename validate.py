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
    decay = setup["train_setup"]["decay"]
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
    fig.savefig(os.path.join(save_dir, "train_val_losses.png"))

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
    imgs_val_diff = imgs_val_input - imgs_val_pred
    np.save(
        file=os.path.join(save_dir, "imgs_val_diff.npy"),
        arr=imgs_val_diff,
        allow_pickle=True,
    )

    # determine threshold on validation dataset
    imgs_val_diff_1d = imgs_val_diff.flatten()
    mean_val = np.mean(imgs_val_diff_1d)
    std_val = np.std(imgs_val_diff_1d)

    # k = 1.65 # confidence 90%
    # k = 1.96 # confidence 95%
    factor_val = 2.58  # confidence 99%
    # k = 3.00 # confidence 99.73%
    # k = 3.30 # confidence 99.90%

    threshold_val = mean_val + factor_val * std_val

    # save validation results
    val_results = {
        "mean_val": str(mean_val),
        "std_val": str(std_val),
        "threshold_val": str(threshold_val),
        "pixel_max_value": str(np.amax(imgs_val_pred)),
        "pixel_min_value": str(np.amin(imgs_val_pred)),
    }

    with open(os.path.join(save_dir, "val_results.json"), "w") as json_file:
        json.dump(val_results, json_file)

    # ===================================================================
    # compute scores on validation images
    output_test = {"filenames": filenames, "scores": [], "mean": [], "std": []}
    for img_val_diff in imgs_val_diff:
        score, mean, std = utils.get_image_score(img_val_diff, factor_val)
        output_test["scores"].append(score)
        output_test["mean"].append(mean)
        output_test["std"].append(std)
        # assert length compatibility

    # format test results in a pd DataFrame
    df_test = pd.DataFrame.from_dict(output_test)
    df_test.to_pickle(os.path.join(save_dir, "df_val.pkl"))
    # display DataFrame
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_test)
    # ===================================================================

    # Histogramm to visualize the ResMap distribution
    fig = plt.figure(figsize=(8, 5))
    plt.hist(imgs_val_diff_1d, bins=100, density=True, stacked=True)
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
        axarr[2].imshow(imgs_val_diff[0])
    except TypeError:
        axarr[2].imshow(imgs_val_diff[0, :, :, 0], cmap=plt.cm.gray)
    axarr[2].set_title("ResMap defect-free val image")
    fig.savefig(os.path.join(save_dir, "3_val_musketeers.png"))


# create parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
)
args = parser.parse_args()

if __name__ == "__main__":
    main(args)

# python3 validate.py -p saved_models/MSE/21-02-2020_17:47:13/CAE_mvtec_b12.h5
