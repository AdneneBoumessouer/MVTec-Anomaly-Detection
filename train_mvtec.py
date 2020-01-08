#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:46:17 2019

@author: adnene33
"""

from numpy import expand_dims
import requests
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

from keras.preprocessing.image import (
    ImageDataGenerator,
)  # unknown issue for this import

# from tensorflow.keras.preprocessing.image import (
#     ImageDataGenerator,
# )  # see in: tensorflow.python
import os
import numpy as np


# ========================= LOAD TRAINING DATASET =======================
X_train_full = np.load("X_train.npy")

X_train, X_valid = X_train_full[:-363], X_train_full[-363:]

# plt.imshow(X_train[1000])
# plt.show()


# ============================ DEFINE MODEL =================================
tf.random.set_seed(42)
np.random.seed(42)

conv_encoder = keras.models.Sequential(
    [
        keras.layers.Conv2D(
            32,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
            input_shape=[256, 256, 1],  # SSIM
        ),  # CONV0 (added layer)
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            32,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV1
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            32,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV2
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            32,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV3
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV4
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV5
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            128,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV6
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV7
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            32,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV8
        # keras.layers.MaxPool2D(pool_size=2),
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
            1,  ## SSIM
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),
    ]
)


conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

# conv_encoder.summary()
# conv_decoder.summary()
# conv_ae.summary()

# =============================== TRAINING SETUP =================================
batch_size = 5  # 32
epochs = 10
data_augmentation = True
flow_from_directory = False
loss = "MSSIM"
# num_predictions = 20

if loss == "SSIM":
    # ---------------------------- TESTING IN PROGRESS-----------------------------
    def ssim_loss(img_true, img_pred):
        return -1 * K.mean(tf.image.ssim(img_true, img_pred, 1.0), axis=-1)

    # SSIM only works with grayscale
    X_train = tf.image.rgb_to_grayscale(X_train)
    X_valid = tf.image.rgb_to_grayscale(X_valid)

    optimizer = keras.optimizers.Adam(
        learning_rate=2e-4, beta_1=0.9, beta_2=0.999, amsgrad=False
    )

    conv_ae.compile(
        loss=ssim_loss, optimizer=optimizer, metrics=[ssim_loss, "mean_squared_error"],
    )

elif loss == "MSSIM":
    # TESTING INTENDED LATER
    def mssim_loss(img_true, img_pred):
        return -1 * K.mean(tf.image.ssim_multiscale(img_true, img_pred, 1.0), axis=-1)

    optimizer = keras.optimizers.Adam(
        learning_rate=2e-4, beta_1=0.9, beta_2=0.999, amsgrad=False
    )

    conv_ae.compile(
        loss=mssim_loss,
        optimizer=optimizer,
        metrics=[mssim_loss, "mean_squared_error"],
    )

else:
    loss_function = "mean_squared_error"

    optimizer = keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False
    )

    conv_ae.compile(
        loss=loss_function, optimizer=optimizer, metrics=["mean_squared_error"]
    )


# =============================== TRAINING =================================

if not data_augmentation:
    print("Not using data augmentation.")
    conv_ae.fit(
        X_train,
        X_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_valid, X_valid),
        shuffle=True,
    )
    # specify model name and directory to save model to
    model_name = "CAE_" + str(epochs) + ".h5"
    save_dir = os.path.join(os.getcwd(), "saved_models/" + loss)

else:
    print("Using real-time data augmentation.")
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=90,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.2,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.2,
        shear_range=0.0,  # set range for random shear
        zoom_range=0.0,  # set range for random zoom 0.15
        channel_shift_range=0.0,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode="nearest",
        cval=0.0,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format="channels_last",
        # fraction of images reserved for validation (strictly between 0 and 1)
        # P.S: change depending on flow or flow_from_directory
        validation_split=0.0,
    )
    # specify model name and directory to save model to
    model_name = "CAE_" + str(epochs) + "_datagen.h5"
    save_dir = os.path.join(os.getcwd(), "saved_models/" + loss)

    if flow_from_directory:
        """
        !!! NOT YET IMPLEMENTED !!!
        Example of directory structure (see: https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
        
        data/
            train/
                good/
                    screw_001.jpg
                    screw_002.jpg
                    ...

            validation/
                good/
                    dog001.jpg
                    dog002.jpg
                    ...
                defect/
                    cat001.jpg
                    cat002.jpg"""

        # Fit the model on the batches generated by datagen.flow_from_directory()
        train_generator = datagen.flow_from_directory(
            directory="datasets/mvtec",
            target_size=(256, 256),
            color_mode="rgb",
            # classes=classes,  # does not work because of subdirectory architecture, see https://keras.io/preprocessing/image/#flow_from_directory
            batch_size=32,
            class_mode="binary",
        )

        history = conv_ae.fit_generator(
            generator=train_generator,
            epochs=epochs,
            validation_data=(X_valid, X_valid),
            workers=-1,
        )
    else:
        # Fit the model on the batches generated by datagen.flow().
        train_generator = datagen.flow(
            x=X_train, y=X_train, batch_size=batch_size, shuffle=True
        )

        history = conv_ae.fit_generator(
            generator=train_generator,
            epochs=epochs,
            steps_per_epoch=np.ceil(len(X_train) / batch_size),
            validation_data=(X_valid, X_valid),
            workers=-1,
        )


# ====================== SEE RESULTS OF TRAINED AE =======================

# show reconstructed image sample--------------------------------


import utils

utils.show_original_and_reconstructed_img_gray(X_train[100], conv_ae)

# ----------------------------------------------------------------


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
conv_ae.save(model_path)
print("Saved trained model at %s " % model_path)


# Score trained model.
# scores = conv_ae.evaluate(X_test, X_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

