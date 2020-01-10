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
import custom_loss_functions
import utils
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np


# ========================= LOAD TRAINING DATASET =======================

X_train, X_valid, _, _ = utils.load_mvtec_data_as_tensor(
    dir_path="datasets/tensors", numpy=False
)

# ============================ DEFINE MODEL =============================
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
            input_shape=[256, 256, 3],  # SSIM
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
            3,  ## SSIM
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

# adjust architecture, see: https://github.com/keras-team/keras/issues/3923

# =============================== TRAINING SETUP =================================
batch_size = 5  # 32
epochs = 10
data_augmentation = True
flow_from_directory = True
loss = "MSSIM"  # 'MSSIM'(input: RGB), 'SSIM'(input:GrayScale) or 'MSE'(equivalent to L2, input: RGB)

# preprocess input tensors
X_train = utils.preprocess_tensor(X_train, loss=loss)
X_valid = utils.preprocess_tensor(X_valid, loss=loss)

# set loss function, optimizer and metric
if loss == "SSIM":
    loss_function = custom_loss_functions.ssim_loss

    optimizer = keras.optimizers.Adam(
        learning_rate=2e-4, beta_1=0.9, beta_2=0.999, amsgrad=False
    )
    conv_ae.compile(
        loss=loss_function,
        optimizer=optimizer,
        metrics=[loss_function, "mean_squared_error"],
    )

elif loss == "MSSIM":
    loss_function = custom_loss_functions.mssim_loss
    optimizer = keras.optimizers.Adam(
        learning_rate=2e-4, beta_1=0.9, beta_2=0.999, amsgrad=False
    )
    conv_ae.compile(
        loss=loss_function,
        optimizer=optimizer,
        metrics=[loss_function, "mean_squared_error"],
    )

elif loss == "MSE":
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
    if flow_from_directory:
        print("Using Keras's flow_from_directory method.")
        # This will do preprocessing and realtime data augmentation:
        train_datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=45,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.0,  # set range for random shear
            zoom_range=0.0,  # set range for random zoom 0.15
            channel_shift_range=0.0,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode="nearest",
            cval=0.0,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=1.0 / 255,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format="channels_last",
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0,
        )

        # For validation dataset, only rescaling
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Generate training batches with datagen.flow_from_directory()
        train_generator = train_datagen.flow_from_directory(
            directory="datasets/data/train",
            target_size=(256, 256),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="input",
        )

        validation_generator = validation_datagen.flow_from_directory(
            directory="datasets/data/validation",
            target_size=(256, 256),
            color_mode="rgb",
            batch_size=1,  # previously == batch_size
            class_mode="input",
        )

        # specify model name and directory to save model to
        model_name = "CAE_" + str(epochs) + "flow_from_dir" + "_datagen.h5"
        save_dir = os.path.join(os.getcwd(), "saved_models/" + loss)

        # Fit the model on the batches generated by datagen.flow_from_directory()
        history = conv_ae.fit_generator(
            generator=train_generator,
            epochs=epochs,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            workers=-1,
        )
    else:
        print("Using Keras's flow")
        train_datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=45,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
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
            validation_split=0.0,
        )
        # Fit the model on the batches generated by datagen.flow().
        train_generator = train_datagen.flow(
            x=X_train, y=X_train, batch_size=batch_size, shuffle=True
        )

        # specify model name and directory to save model to
        model_name = "CAE_" + str(epochs) + "flow_from_dir" + "_datagen.h5"
        save_dir = os.path.join(os.getcwd(), "saved_models/" + loss)

        # Fit the model on the batches generated by datagen.flow()
        history = conv_ae.fit_generator(
            generator=train_generator,
            epochs=epochs,
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=(X_valid, X_valid),
            workers=-1,
        )


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

