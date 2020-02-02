#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:46:17 2019

@author: adnene33
This Script is meant to train on a single object category of MVTec, unlike train_mvtec.py
"""
import os
import sys

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import custom_loss_functions
import architectures
import utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import requests
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import datetime
import csv
import pandas as pd
import json

import argparse


def main(args):
    # ========================= LOAD TRAINING SETUP =======================
    directory = args.directory
    train_data_dir = os.path.join(directory, "train")
    epochs = args.epochs
    batch_size = args.batch
    validation_split = 0.1  # incorporate to args

    # NEW TRAINING
    if args.command == "new":
        loss = args.loss.upper()

        if loss == "SSIM":
            channels = 1
            color_mode = "grayscale"
        else:
            channels = 3
            color_mode = "rgb"

        # ==================== GET MODEL ARCHITECTURE =======================

        model = architectures.autoencoder_0(channels)

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
            loss_function = custom_loss_functions.l2_loss
            optimizer = keras.optimizers.Adam(
                learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False
            )
            model.compile(
                loss=loss_function, optimizer=optimizer, metrics=["mean_squared_error"]
            )

    # RESUME TRAINING
    elif args.command == "resume":
        model_path = args.model

        # load model
        model, train_setup, _ = utils.load_SavedModel(model_path)
        color_mode = train_setup["color_mode"]
        validation_split = train_setup["validation_split"]

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
        rotation_range=15,
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
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # randomly change brightness (darker < 1 < brighter)
        brightness_range=[0.9, 1.1],
        # set rescaling factor (applied before any other transformation)
        rescale=1.0 / 255,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format="channels_last",
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=validation_split,
    )

    # For validation dataset, only rescaling
    # validation_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.1)

    # Generate training batches with datagen.flow_from_directory()
    train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=(256, 256),
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="input",
        subset="training",
    )

    # Generate validation batches with datagen.flow_from_directory()
    validation_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=(256, 256),
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="input",
        subset="validation",
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
    model_name = "CAE_e{}_b{}_0".format(epochs, batch_size)
    now = datetime.datetime.now()
    save_dir = os.path.join(
        os.getcwd(), "saved_models", loss, now.strftime("%d-%m-%Y_%H:%M:%S")
    )

    # save model
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    # using SavedModel format
    # model.save(model_path) # variant 1
    # tf.keras.models.save_model(
    #     model, model_path, include_optimizer=True, save_format="tf"
    # )  # variant 2

    # using HDF5 format
    # model.save(model_path+'.h5') # variant 1
    tf.keras.models.save_model(
        model, model_path, include_optimizer=True, save_format="h5"
    )  # variant 2

    print("Saved trained model at %s " % model_path)

    # save training history
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = os.path.join(save_dir, "history.csv")
    with open(hist_csv_file, mode="w") as f:
        hist_df.to_csv(f)
    print("Saved training history at %s " % hist_csv_file)

    # save training setup
    if args.command == "new":
        train_dict = {
            "directory": directory,
            "epochs": epochs,
            "batch_size": batch_size,
            "loss": loss,
            "color_mode": color_mode,
            "channels": channels,
            "validation_split": validation_split,
        }
    elif args.command == "resume":
        train_dict = {
            "directory": directory,
            "epochs": epochs,
            "batch_size": batch_size,
            "loss": loss,
            "color_mode": color_mode,
            "channels": channels,
            "validation_split": validation_split,
            "path_to_previous_model": args.model,
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

parser_new_training.add_argument(
    "-d", "--directory", type=str, required=True, metavar="", help="training directory"
)

# create the subparser to resume the training of an existing model
parser_resume_training = subparsers.add_parser("resume")
parser_resume_training.add_argument(
    "-m", "--model", type=str, required=True, metavar="", help="path to existing model"
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

# args = parser.parse_args(["new", '-d', "mvtec/hazelnut/train",  "-e", "50", "-b", "4", "-l", "mssim"])
args = parser.parse_args()

if __name__ == "__main__":
    if tf.test.is_gpu_available():
        print("GPU was detected.")
    else:
        print("No GPU was detected. CNNs can be very slow without a GPU.")
    print("Tensorflow version: {}".format(tf.__version__))
    print("Keras version: {}".format(keras.__version__))
    main(args)

# python3 train.py new -d mvtec/hazelnut -e 1 -b 1 -l mse
# python3 train.py new -d mvtec/hazelnut -e 100 -b 24 -l mse
