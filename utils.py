from numpy import expand_dims
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import custom_loss_functions
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


def load_model_HDF5(model_path):
    """Loads model (HDF5 format), training setup and training history.
    This format makes it difficult to load a trained model for further training,
    but works good enough for one training round."""
    # load autoencoder
    loss = model_path.split("/")[1]
    if loss == "MSSIM":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "mssim": custom_loss_functions.mssim,
            },
            compile=True,
        )

    elif loss == "SSIM":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "ssim": custom_loss_functions.ssim,
            },
            compile=True,
        )

    else:
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "l2": custom_loss_functions.l2,
            },
            compile=True,
        )

    # load training history
    dir_name = os.path.dirname(model_path)
    history = pd.read_csv(os.path.join(dir_name, "history.csv"))

    # # load model configuration
    # with open(os.path.join(dir_name, "model_config.json"), "r") as read_file:
    #     model_config = json.load(read_file)

    # load training setup
    with open(os.path.join(dir_name, "setup.json"), "r") as read_file:
        setup = json.load(read_file)

    return model, setup, history


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


def get_image_score(image, factor):
    image_1d = image.flatten()
    mean_image = np.mean(image_1d)
    std_image = np.std(image_1d)
    score = mean_image + factor*std_image
    return score, mean_image, std_image
