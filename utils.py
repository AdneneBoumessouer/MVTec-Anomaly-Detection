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

    # load training setup
    with open(os.path.join(dir_name, "train_setup.json"), "r") as read_file:
        train_setup = json.load(read_file)

    return model, train_setup, history
