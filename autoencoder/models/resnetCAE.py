from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Add,
    ReLU,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.regularizers import l2
from autoencoder.models.resnet.resnet import ResnetBuilder

# Preprocessing variables
RESCALE = 1 / 255
SHAPE = (256, 256)
PREPROCESSING_FUNCTION = None
PREPROCESSING = None
VMIN = 0.0  # -1.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN


def build_model(color_mode):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3

    # encoder
    resnet = ResnetBuilder.build_resnet_18((*SHAPE, channels))
    x = Conv2D(512, (1, 1), strides=1, activation="relu", padding="valid")(
        resnet.output
    )
    encoded = Conv2D(512, (1, 1), strides=1, activation="relu", padding="valid")(x)

    # decoder
    layer_1 = Conv2DTranspose(
        512, kernel_size=4, strides=2, padding="same", activation=None,
    )(encoded)
    layer_2 = BatchNormalization()(layer_1)
    layer_3 = ReLU()(layer_2)
    layer_4 = Conv2DTranspose(
        512, kernel_size=3, strides=1, padding="SAME", activation=None,
    )(layer_3)
    layer_5 = BatchNormalization()(layer_4)
    layer_6 = ReLU()(layer_5)
    ####
    add_1 = Add()([layer_1, layer_6])
    ####
    layer_7 = Conv2DTranspose(
        256, kernel_size=4, strides=2, padding="same", activation=None,
    )(add_1)
    layer_8 = BatchNormalization()(layer_7)
    layer_9 = ReLU()(layer_8)
    layer_10 = Conv2DTranspose(
        256, kernel_size=3, strides=1, padding="SAME", activation=None,
    )(layer_9)
    layer_11 = BatchNormalization()(layer_10)
    layer_12 = ReLU()(layer_11)
    ####
    add_2 = Add()([layer_7, layer_12])
    ####
    layer_13 = Conv2DTranspose(
        128, kernel_size=4, strides=2, padding="SAME", activation=None,
    )(add_2)
    layer_14 = BatchNormalization()(layer_13)
    layer_15 = ReLU()(layer_14)
    layer_16 = Conv2DTranspose(
        128, kernel_size=3, strides=1, padding="SAME", activation=None,
    )(layer_15)
    layer_17 = BatchNormalization()(layer_16)
    layer_18 = ReLU()(layer_17)
    ####
    add_3 = Add()([layer_13, layer_18])
    ####
    layer_19 = Conv2DTranspose(
        64, kernel_size=4, strides=2, padding="same", activation=None,
    )(add_3)
    layer_20 = BatchNormalization()(layer_19)
    layer_21 = ReLU()(layer_20)
    layer_22 = Conv2DTranspose(
        64, kernel_size=3, strides=1, padding="SAME", activation=None,
    )(layer_21)
    layer_23 = BatchNormalization()(layer_22)
    layer_24 = ReLU()(layer_23)
    ####
    add_4 = Add()([layer_19, layer_24])
    ####
    decoded = Conv2DTranspose(
        channels, kernel_size=4, strides=2, padding="same", activation="sigmoid",
    )(add_4)

    model = Model(resnet.input, decoded)

    return model
