"""
Implementation of mvtec architecture inspired by:
https://github.com/cheapthrillandwine/Improving_Unsupervised_Defect_Segmentation/blob/master/Improving_AutoEncoder_Samples.ipynb
"""

import tensorflow as tf
from tensorflow import keras

# Preprocessing parameters
RESCALE = 1.0 / 255
SHAPE = (256, 256)
PREPROCESSING_FUNCTION = None
PREPROCESSING = None
VMIN = 0.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN


def build_model(color_mode):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3

    # define model
    input_img = keras.layers.Input(shape=(*SHAPE, channels))
    # Encode-----------------------------------------------------------
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(
        input_img
    )
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(128, (4, 4), strides=2, activation="relu", padding="same")(
        x
    )
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    encoded = keras.layers.Conv2D(1, (8, 8), strides=1, padding="same")(x)

    # Decode---------------------------------------------------------------------
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(
        encoded
    )
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (4, 4), strides=2, activation="relu", padding="same")(
        x
    )
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((4, 4))(x)
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (8, 8), activation="relu", padding="same")(x)

    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(
        channels, (8, 8), activation="sigmoid", padding="same"
    )(x)

    model = keras.models.Model(input_img, decoded)

    return model
