import tensorflow as tf
from tensorflow import keras

# Preprocessing variables
RESCALE = None
SHAPE = (224, 224)  # check if correct
PREPROCESSING_FUNCTION = keras.applications.nasnet.preprocess_input
PREPROCESSING = "keras.applications.nasnet.NASNetMobile.preprocess_input"
VMIN = -1.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN


def build_model():
    base_encoder_nasnet = tf.keras.applications.NASNetMobile(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        input_tensor=None,
    )

    for layer in base_encoder_nasnet.layers:
        layer.trainable = False

    model_nasnet = keras.Sequential([base_encoder_nasnet])

    model_nasnet.add(
        keras.layers.Conv2DTranspose(
            256, kernel_size=4, strides=2, activation=None, padding="same"
        )
    )
    model_nasnet.add(keras.layers.BatchNormalization())
    model_nasnet.add(keras.layers.ReLU())

    model_nasnet.add(
        keras.layers.Conv2DTranspose(
            128, kernel_size=4, strides=2, activation=None, padding="same"
        )
    )
    model_nasnet.add(keras.layers.BatchNormalization())
    model_nasnet.add(keras.layers.ReLU())

    model_nasnet.add(
        keras.layers.Conv2DTranspose(
            64, kernel_size=4, strides=2, activation=None, padding="same"
        )
    )
    model_nasnet.add(keras.layers.BatchNormalization())
    model_nasnet.add(keras.layers.ReLU())
    model_nasnet.summary()

    model_nasnet.add(
        keras.layers.Conv2DTranspose(
            32, kernel_size=4, strides=2, activation=None, padding="same"
        )
    )
    model_nasnet.add(keras.layers.BatchNormalization())
    model_nasnet.add(keras.layers.ReLU())
    model_nasnet.summary()

    model_nasnet.add(
        keras.layers.Conv2DTranspose(
            3, kernel_size=4, strides=2, activation="sigmoid", padding="same"
        )
    )
    model_nasnet.summary()

    return model_nasnet
