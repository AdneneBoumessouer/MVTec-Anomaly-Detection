"""
Model inspired by: https://github.com/natasasdj/anomalyDetection
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    LeakyReLU,
    Activation,
    concatenate,
    Flatten,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


# Preprocessing parameters
RESCALE = 1.0 / 255
SHAPE = (256, 256)
PREPROCESSING_FUNCTION = None
PREPROCESSING = None
VMIN = 0.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN


def inception_layer(x, filters):
    # 1x1 convolution
    x0 = Conv2D(
        filters, (1, 1), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x0 = BatchNormalization()(x0)
    x0 = LeakyReLU(alpha=0.1)(x0)
    # 3x3 convolution
    x1 = Conv2D(
        filters, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)
    # 5x5 convolution
    x2 = Conv2D(
        filters, (5, 5), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(alpha=0.1)(x2)
    # Max Pooling
    x3 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    x3 = Conv2D(
        filters, (1, 1), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x3)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU(alpha=0.1)(x3)
    output = concatenate([x0, x1, x2, x3], axis=3)
    return output


##### Inception-like Convolutional AutoEncoder #####


def build_model(color_mode, filters=[32, 64, 128]):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3
    img_dim = (*SHAPE, channels)

    # input
    input_img = Input(shape=img_dim)

    # encoder
    x = inception_layer(input_img, filters[0])
    x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

    # added -------------------------------------
    x = inception_layer(x, filters[0])
    x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
    # -------------------------------------------

    x = inception_layer(x, filters[1])
    x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

    # added -------------------------------------
    x = inception_layer(x, filters[1])
    x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
    # -------------------------------------------

    x = inception_layer(x, filters[2])
    x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

    # added -------------------------------------
    x = inception_layer(x, filters[2])
    x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
    # -------------------------------------------

    # encoded = x

    # decoder
    x = inception_layer(x, filters[2])
    x = UpSampling2D((2, 2))(x)

    # added -----------------------------
    x = inception_layer(x, filters[2])
    x = UpSampling2D((2, 2))(x)
    # -----------------------------------

    x = inception_layer(x, filters[1])
    x = UpSampling2D((2, 2))(x)

    # added -----------------------------
    x = inception_layer(x, filters[1])
    x = UpSampling2D((2, 2))(x)
    # -----------------------------------

    x = inception_layer(x, filters[0])
    x = UpSampling2D((2, 2))(x)

    # added -----------------------------
    x = inception_layer(x, filters[0])
    x = UpSampling2D((2, 2))(x)
    # -----------------------------------

    x = Conv2D(
        img_dim[2], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)

    decoded = x
    # model
    autoencoder = Model(input_img, decoded)
    return autoencoder
