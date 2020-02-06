import tensorflow as tf
from tensorflow import keras


def autoencoder_0(channels):
    """old model"""
    conv_encoder = keras.models.Sequential(
        [
            # keras.layers.InputLayer(input_shape=(256, 256, channels)),
            keras.layers.Conv2D(
                32,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
                input_shape=(256, 256, channels),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                32,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                32,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                32,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                64,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                128,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # CONV6
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                32,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                100, kernel_size=8, strides=1, padding="VALID", activation="relu"
            ),
            # keras.layers.BatchNormalization(),
            # keras.layers.Flatten(),
        ]
    )

    conv_decoder = keras.models.Sequential(
        [
            # keras.layers.InputLayer(input_shape=(1, 1, 100)),
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=3,
                strides=8,
                padding="VALID",
                # activation="relu",
                activation=keras.layers.LeakyReLU(0.2),
                input_shape=(1, 1, 100),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(
                64,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(
                128,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(
                64,
                kernel_size=3,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(
                64,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=3,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            # keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(
                channels,
                kernel_size=4,
                strides=2,
                padding="SAME",
                # activation=keras.layers.LeakyReLU(0.2),
                activation="relu",
            ),
        ]
    )
    model = keras.models.Sequential([conv_encoder, conv_decoder])
    print(conv_encoder.summary())
    print(conv_decoder.summary())
    print(model.summary())

    description_dict = {"pretrained": False, "configuation": "MVTec", "comments": None}

    return model, description_dict


# ==================================================================
# ==================================================================
# ==================================================================


def autoencoder_pretrained(channels):

    base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(299, 299, 3),
        pooling=None,
    )
    print(base_model.summary())

    conv_encoder = keras.models.Sequential(
        [
            base_model,
            keras.layers.Conv2D(
                512, kernel_size=1, strides=1, padding="VALID", activation=None,
            ),
            keras.layers.Conv2D(
                512, kernel_size=1, strides=1, padding="VALID", activation=None,
            ),
        ]
    )

    # decoder
    inputs = keras.layers.Input(shape=tuple(conv_encoder.output.shape[-3:]))

    layer_1 = keras.layers.Conv2DTranspose(
        512, kernel_size=4, strides=2, padding="VALID", activation=None,
    )(inputs)
    layer_2 = keras.layers.BatchNormalization()(layer_1)
    layer_3 = keras.layers.ReLU()(layer_2)

    layer_4 = keras.layers.Conv2DTranspose(
        512, kernel_size=3, strides=1, padding="VALID", activation=None,
    )(inputs)
    layer_5 = keras.layers.BatchNormalization()(layer_4)
    layer_6 = keras.layers.ReLU()(layer_5)

    add_1 = keras.layers.Add()([layer_1.output + layer_6.output])

    layer_1 = keras.layers.Conv2DTranspose(
        512, kernel_size=4, strides=2, padding="VALID", activation=None,
    )(inputs)
    layer_2 = keras.layers.BatchNormalization()(layer_1)
    layer_3 = keras.layers.ReLU()(layer_2)

    conv_decoder = keras.models.Model(inputs=inputs, outputs=layer_3)


# ==================================================================
# ==================================================================
# ==================================================================

