import tensorflow as tf
from tensorflow import keras


def autoencoder_0(channels):
    """old model"""
    conv_encoder = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                32,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
                input_shape=[256, 256, channels],
            ),
            keras.layers.Conv2D(
                32,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2D(
                32,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2D(
                32,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2D(
                64,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2D(
                128,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # CONV6
            keras.layers.Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2D(
                32,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2D(
                100, kernel_size=8, strides=1, padding="VALID", activation="relu"
            ),
        ]
    )

    conv_decoder = keras.models.Sequential(
        [
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=3,
                strides=8,
                padding="VALID",
                activation="relu",
                input_shape=[1, 1, 100],
            ),
            keras.layers.Conv2DTranspose(
                64,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2DTranspose(
                128,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2DTranspose(
                64,
                kernel_size=3,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2DTranspose(
                64,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=3,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Conv2DTranspose(
                channels,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
        ]
    )
    model = keras.models.Sequential([conv_encoder, conv_decoder])
    print(conv_encoder.summary())
    print(conv_decoder.summary())
    print(model.summary())

    return model


def autoencoder_1(channels):
    """current model"""
    conv_encoder = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                32,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
                input_shape=[256, 256, channels],
            ),  # CONV0 (added layer)
            keras.layers.MaxPool2D(pool_size=2, padding="same"),  # 128
            keras.layers.Conv2D(
                32,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # CONV1
            keras.layers.MaxPool2D(pool_size=2, padding="same"),  # 64
            keras.layers.Conv2D(
                32,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # CONV2
            keras.layers.MaxPool2D(pool_size=2, padding="same"),  # 32
            keras.layers.Conv2D(
                32,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # CONV3
            keras.layers.MaxPool2D(pool_size=1, padding="same"),  # 32
            keras.layers.Conv2D(
                64,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # CONV4
            keras.layers.MaxPool2D(pool_size=2, padding="same"),  # 16
            keras.layers.Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # CONV5
            keras.layers.MaxPool2D(pool_size=1, padding="same"),  # 16
            keras.layers.Conv2D(
                128,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # CONV6
            keras.layers.MaxPool2D(pool_size=2, padding="same"),  # 8
            keras.layers.Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # CONV7
            keras.layers.MaxPool2D(pool_size=1, padding="same"),  # 8
            keras.layers.Conv2D(
                32,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # CONV8
            keras.layers.MaxPool2D(pool_size=1, padding="same"),  # 8
            keras.layers.Conv2D(
                100, kernel_size=8, strides=1, padding="VALID", activation="relu"
            ),  # CONV9
        ]
    )

    conv_decoder = keras.models.Sequential(
        [
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=3,
                strides=8,
                padding="VALID",
                activation="relu",
                input_shape=[1, 1, 100],
            ),  # (None, 8, 8, 32)
            keras.layers.Conv2DTranspose(
                64,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # (None, 8, 8, 32)
            keras.layers.Conv2DTranspose(
                128,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # (None, 8, 8, 32)
            keras.layers.Conv2DTranspose(
                64,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # (None, 16, 8, 32)
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2DTranspose(
                64,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # (None, 16, 8, 32)
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=3,
                strides=2,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # (None, 32, 8, 32)
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # (None, 32, 8, 32)
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),  # (None, 64, 8, 32)
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2DTranspose(
                32,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2DTranspose(
                channels,
                kernel_size=4,
                strides=1,
                padding="SAME",
                activation=keras.layers.LeakyReLU(0.2),
            ),
        ]
    )

    model = keras.models.Sequential([conv_encoder, conv_decoder])

    print(conv_encoder.summary())
    print(conv_decoder.summary())
    print(model.summary())

    return model


def autoencoder_2():
    """Prototype for transfer learning"""

    base_encoder = keras.applications.xception.Xception(
        input_shape=(256, 256, 3),
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
    )
    # LAST LAYER: block14_sepconv2_act (Activation) (None, 8, 8, 2048)

    conv_encoder = keras.models.Sequential(
        [
            base_encoder,
            keras.layers.Conv2D(
                512,
                kernel_size=1,
                strides=1,
                padding="VALID",
                activation=keras.layers.LeakyReLU(0.2),
            ),
        ]
    )

    conv_decoder = keras.models.Sequential(
        [
            keras.layers.Conv2DTranspose(
                512,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=None,
                input_shape=[8, 8, 512],
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLu(),
            keras.layers.Conv2DTranspose(
                256,
                kernel_size=4,
                strides=2,
                padding="SAME",
                activation=None,
                input_shape=[8, 8, 512],
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLu(),
        ]
    )

    # base_encoder = keras.applications.inception_resnet_v2.InceptionResNetV2(
    #     input_shape=(256, 256, 3),
    #     include_top=False,
    #     weights="imagenet",
    #     input_tensor=None,
    #     pooling="avg",
    # )
    # # LAST LAYER: conv_7b_ac (Activation)         (None, 6, 6, 1536)

    # base_encoder = keras.applications.resnet_v2.ResNet50V2(
    #     input_shape=(256, 256, 3),
    #     include_top=False,
    #     weights="imagenet",
    #     input_tensor=None,
    #     pooling=None,
    # )
    # # LAST LAYER: post_relu (Activation)          (None, 8, 8, 2048)

    # base_encoder = keras.applications.resnet.ResNet50(
    #     input_shape=(256, 256, 3),
    #     include_top=False,
    #     weights="imagenet",
    #     input_tensor=None,
    #     pooling=None,
    # )
    # # LAST LAYER:

    base_encoder.summary()

