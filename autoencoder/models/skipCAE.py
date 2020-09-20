"""
Model inspired by: https://towardsdatascience.com/using-skip-connections-to-enhance-denoising-autoencoder-algorithms-849e049c0ac9
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    LeakyReLU,
    Flatten,
    Reshape,
    Add,
)
from tensorflow.keras.models import Model
import numpy as np

# Preprocessing parameters
RESCALE = 1.0 / 255
SHAPE = (256, 256)
PREPROCESSING_FUNCTION = None
PREPROCESSING = None
VMIN = 0.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN


# Helper function to apply activation and batch normalization to the
# output added with output of residual connection from the encoder
def lrelu_bn(inputs):
    lrelu = LeakyReLU()(inputs)
    bn = BatchNormalization()(lrelu)
    return bn


def build_model(color_mode):

    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3
    img_dim = (*SHAPE, channels)

    # added ----------------------------------------------------------------
    input_img = Input(shape=img_dim)
    # ----------------------------------------------------------------------

    # Encoder

    # # added ----------------------------------------------------------------
    y = Conv2D(32, (3, 3), padding="same", strides=(2, 2))(input_img)
    y = LeakyReLU()(y)
    # # ----------------------------------------------------------------------

    y = Conv2D(32, (3, 3), padding="same", strides=(2, 2))(y)
    y = LeakyReLU()(y)
    y1 = Conv2D(64, (3, 3), padding="same", strides=(2, 2))(y)  # skip-1 64
    y = LeakyReLU()(y1)
    y = Conv2D(64, (3, 3), padding="same", strides=(2, 2))(y)
    y = LeakyReLU()(y)
    y2 = Conv2D(128, (3, 3), padding="same", strides=(2, 2))(y)  # skip-2 #128
    y = LeakyReLU()(y2)
    y = Conv2D(128, (3, 3), padding="same", strides=(2, 2))(y)
    y = LeakyReLU()(y)
    y = Conv2D(256, (3, 3), padding="same", strides=(2, 2))(y)
    y = LeakyReLU()(y)
    y = Conv2D(256, (3, 3), padding="same", strides=(2, 2))(y)
    y = LeakyReLU()(y)

    # Flattening for the bottleneck
    vol = y.shape
    x = Flatten()(y)
    latent = Dense(64, activation="relu")(x)

    # Decoder
    y = Dense(np.prod(vol[1:]), activation="relu")(latent)
    y = Reshape((vol[1], vol[2], vol[3]))(y)
    y = Conv2DTranspose(256, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(256, (3, 3), padding="same", strides=(2, 2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(128, (3, 3), padding="same", strides=(2, 2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(128, (3, 3), padding="same", strides=(2, 2))(y)
    y = Add()([y2, y])  # second skip connection added here
    y = lrelu_bn(y)
    y = Conv2DTranspose(64, (3, 3), padding="same", strides=(2, 2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(64, (3, 3), padding="same", strides=(2, 2))(y)
    y = Add()([y1, y])  # first skip connection added here
    y = lrelu_bn(y)
    y = Conv2DTranspose(32, (3, 3), padding="same", strides=(2, 2))(y)
    y = LeakyReLU()(y)

    # # added ----------------------------------------------------------------
    y = Conv2DTranspose(32, (3, 3), padding="same", strides=(2, 2))(y)
    y = LeakyReLU()(y)
    # # ----------------------------------------------------------------------

    y = Conv2DTranspose(
        img_dim[2], (3, 3), activation="sigmoid", padding="same", strides=(2, 2)
    )(y)

    # model
    autoencoder = Model(input_img, y)
    return autoencoder
