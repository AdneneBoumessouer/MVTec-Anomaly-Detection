import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from modules import loss_functions as loss_functions
import os
import csv
import pandas as pd
import json

"""
SAVE AND LOAD LINKS:
https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format
https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model
https://www.tensorflow.org/api_docs/python/tf/saved_model/save
https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model


https://stackoverflow.com/questions/55364954/keras-load-model-cant-recognize-tensorflows-activation-functions
"""


def load_SavedModel(model_path):
    """Save model with SavedModel format.
    This format seems to trigger an error message upon loading the model:
    ValueError: An empty Model cannot be used as a Layer."""
    # load model
    model = tf.keras.models.load_model(model_path, compile=True)
    # load training history
    dir_name = os.path.dirname(model_path)
    history = pd.read_csv(os.path.join(dir_name, "history.csv"))
    # load training setup
    with open(os.path.join(dir_name, "train_setup.json"), "r") as read_file:
        train_setup = json.load(read_file)
    return model, train_setup, history


def get_model_setup(model_path):
    dir_name = os.path.dirname(model_path)
    with open(os.path.join(dir_name, "setup.json"), "r") as read_file:
        setup = json.load(read_file)
    return setup


def load_model_HDF5(model_path):
    """Loads model (HDF5 format), training setup and training history.
    This format makes it difficult to load a trained model for further training,
    but works good enough for one training round."""

    # load loss function used in training
    dir_name = os.path.dirname(model_path)
    setup = get_model_setup(model_path)
    loss = setup["train_setup"]["loss"]

    # load autoencoder
    if loss == "MSSIM":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "mssim": loss_functions.mssim,
            },
            compile=True,
        )

    elif loss == "SSIM":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "ssim": loss_functions.ssim,
            },
            compile=True,
        )

    else:
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "l2": loss_functions.l2,
            },
            compile=True,
        )

    # load training history
    history = pd.read_csv(os.path.join(dir_name, "history.csv"))

    return model, setup, history


def save_np(arr, save_dir, filename):
    np.save(
        file=os.path.join(save_dir, filename), arr=arr, allow_pickle=True,
    )


def extend_dict(dict1, dict2):
    dict3 = {}
    for key in list(dict1.keys()):
        dict3[key] = []
        dict3[key].extend(dict1[key])
        dict3[key].extend(dict2[key])
    return dict3


def get_epochs_trained(history_dict):
    key = list(history_dict.keys())[0]
    return len(history_dict[key])


def get_total_number_test_images(test_data_dir):
    total_number = 0
    sub_dir_names = os.listdir(test_data_dir)
    for sub_dir_name in sub_dir_names:
        sub_dir_path = os.path.join(test_data_dir, sub_dir_name)
        filenames = os.listdir(sub_dir_path)
        number = len(filenames)
        total_number = total_number + number
    return total_number


def plot_input_pred_resmaps_val(inputs, preds, resmaps, index_val):
    fig, axarr = plt.subplots(3, 1, figsize=(5, 18))
    try:
        axarr[0].imshow(inputs[index_val])
    except TypeError:
        axarr[0].imshow(inputs[index_val, :, :, 0], cmap=plt.cm.gray)
    axarr[0].set_title("original defect-free val image")
    try:
        axarr[1].imshow(preds[index_val])
    except TypeError:
        axarr[1].imshow(preds[index_val, :, :, 0], cmap=plt.cm.gray)
    axarr[1].set_title("reconstruction defect-free val image")
    try:
        axarr[2].imshow(resmaps[index_val])
    except TypeError:
        axarr[2].imshow(resmaps[index_val, :, :, 0], cmap=plt.cm.gray)
    axarr[2].set_title("ResMap defect-free val image")

    return fig


def plot_input_pred_resmaps_test(inputs, preds, resmaps, index_test):
    fig, axarr = plt.subplots(3, 1, figsize=(5, 18))
    try:
        axarr[0].imshow(inputs[index_test])
    except TypeError:
        axarr[0].imshow(inputs[index_test, :, :, 0], cmap=plt.cm.gray)
    axarr[0].set_title("original sample test image")
    try:
        axarr[1].imshow(preds[index_test])
    except TypeError:
        axarr[1].imshow(preds[index_test, :, :, 0], cmap=plt.cm.gray)
    axarr[1].set_title("reconstruction test image")
    try:
        axarr[2].imshow(resmaps[index_test])
    except TypeError:
        axarr[2].imshow(resmaps[index_test, :, :, 0], cmap=plt.cm.gray)
    axarr[2].set_title("ResMap test image")

    return fig


def get_preprocessing_function(architecture):
    if architecture in ["mvtec", "mvtec2"]:
        preprocessing_function = None
    elif architecture == "resnet":
        preprocessing_function = keras.applications.inception_resnet_v2.preprocess_input
    elif architecture == "nasnet":
        preprocessing_function = keras.applications.nasnet.preprocess_input
    return preprocessing_function
