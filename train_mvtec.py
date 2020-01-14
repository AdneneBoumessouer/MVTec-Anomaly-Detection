#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:46:17 2019

@author: adnene33
"""
import os
import sys

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import custom_loss_functions
import utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy import expand_dims

import requests
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import datetime
import csv
import pandas as pd
import json

import argparse


def main(args):
    # ========================= LOAD TRAINING DATASET =======================
    # NEW TRAINING
    if args.command == "new":
        epochs = args.epochs
        batch_size = args.batch
        loss = args.loss.upper()

        if loss == "SSIM":
            channels = 1
            color_mode = "grayscale"
        else:
            channels = 3
            color_mode = "rgb"

        # ============================ DEFINE MODEL =============================
        # tf.random.set_seed(42)
        # np.random.seed(42)

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
        # adjust architecture, see: https://github.com/keras-team/keras/issues/3923

        # =============================== TRAINING SETUP =================================

        # set loss function, optimizer and metric
        if loss == "SSIM":
            loss_function = custom_loss_functions.ssim_loss

            optimizer = keras.optimizers.Adam(
                learning_rate=2e-4, beta_1=0.9, beta_2=0.999, amsgrad=False
            )
            model.compile(
                loss=loss_function,
                optimizer=optimizer,
                metrics=[loss_function, "mean_squared_error"],
            )

        elif loss == "MSSIM":
            loss_function = custom_loss_functions.mssim_loss
            optimizer = keras.optimizers.Adam(
                learning_rate=2e-4, beta_1=0.9, beta_2=0.999, amsgrad=False
            )
            model.compile(
                loss=loss_function,
                optimizer=optimizer,
                metrics=[loss_function, "mean_squared_error"],
            )

        elif loss == "MSE":
            loss_function = "mean_squared_error"
            optimizer = keras.optimizers.Adam(
                learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False
            )
            model.compile(
                loss=loss_function, optimizer=optimizer, metrics=["mean_squared_error"]
            )

    # RESUME TRAINING
    elif args.command == "resume":
        model_path = args.path
        epochs = args.epochs
        batch_size = args.batch_size        
        
        # load model
        # model, train_setup, history_resume = utils.load_model_HDF5(model_path)
        model, train_setup, history_resume = utils.load_SavedModel(model_path)
        # batch_size = train_setup['batch_size'] # unnecessary with new SavedModel format
        color_mode = train_setup['color_mode'] # has to be consistent with previous model
        # loss = train_setup['loss'] # unnecessary with new SavedModel format
        

    # =============================== TRAINING =================================
    if batch_size != 1:
        print("Using real-time data augmentation.")
    else:
        print("Not using data augmentation")

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
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="input",
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory="datasets/data/validation",
        target_size=(256, 256),
        color_mode=color_mode,
        batch_size=1, 
        class_mode="input",
    )    

    # Fit the model on the batches generated by datagen.flow_from_directory()
    history = model.fit_generator(
        generator=train_generator,
        epochs=epochs,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        workers=-1,
    )

    # specify model name and directory to save model to
    # model_name = 'CAE_e{}_b{}_0.h5'.format(epochs,batch_size) # not using HDF5
    model_name = 'CAE_e{}_b{}_0'.format(epochs,batch_size)
    now = datetime.datetime.now()
    save_dir = os.path.join(
        os.getcwd(), "saved_models", loss, now.strftime("%d-%m-%Y_%H:%M:%S")
    )

    # save model using SavedModel format (not HDF5)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print("Saved trained model at %s " % model_path)

    # save training history
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = os.path.join(save_dir, "history.csv")
    with open(hist_csv_file, mode="w") as f:
        hist_df.to_csv(f)
    print("Saved training history at %s " % hist_csv_file)

    # save training setup
    if args.command == 'new':
        train_dict = {
            "epochs": epochs,
            "batch_size": batch_size,
            "loss": loss,
            "color_mode": color_mode,
            "channels": channels,
        }
    elif argparse.command == 'resume':
        train_dict = {
            "epochs": epochs,
            "batch_size": batch_size,
            "loss": loss,
            "color_mode": color_mode,
            "channels": channels,
            'path_to_previous_model': args.path,
        }
    with open(os.path.join(save_dir, "train_setup.json"), "w") as json_file:
        json.dump(train_dict, json_file)


# create top level parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(
    help="help for subcommand", title="commands", dest="command"
)
# create the subparser to begin training a new model
parser_new_training = subparsers.add_parser("new")
parser_new_training.add_argument(
    "-e",
    "--epochs",
    type=int,
    required=True,
    metavar="",
    help="number of training epochs",
)
parser_new_training.add_argument(
    "-b", "--batch", type=int, required=True, metavar="", help="batch size"
)
parser_new_training.add_argument(
    "-l",
    "--loss",
    type=str,
    required=True,
    metavar="",
    choices=["mssim", "ssim", "mse"],
    help="loss function used during training",
)
# create the subparser to resume the training of an existing model
parser_resume_training = subparsers.add_parser("resume")
parser_resume_training.add_argument(
    "-p", "--path", type=str, required=True, metavar="", help="path to existing model"
)
parser_resume_training.add_argument(
    "-e",
    "--epochs",
    type=int,
    required=True,
    metavar="",
    help="number of training epochs",
)
parser_resume_training.add_argument(
    "-b", "--batch", type=int, required=True, metavar="", help="batch size"
)

# args = parser.parse_args(["new", "-e", "50", "-b", "4", "-l", "mssim"])
args = parser.parse_args()

if __name__ == "__main__":
    main(args)


# Score trained model.
# scores = model.evaluate(X_test, X_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
