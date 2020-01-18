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
    model_path = args.model
    val_data_dir = args.valdir

    # load model and setup
    model, train_setup, _ = utils.load_SavedModel(model_path)

    color_mode = train_setup["color_mode"]
    validation_split = train_setup["validation_split"]
    channels = train_setup["channels"]
    loss = train_setup["loss"]

    # This will do preprocessing
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255, validation_split=validation_split
    )

    # Generate validation batches with datagen.flow_from_directory()
    validation_generator = validation_datagen.flow_from_directory(
        directory=val_data_dir,
        target_size=(256, 256),
        color_mode=color_mode,
        batch_size=1,
        shuffle=False,
        class_mode="input",
        subset="validation",
    )

    # Generate input-image batches with datagen.flow_from_directory()
    input_generator = validation_datagen.flow_from_directory(
        directory=val_data_dir,
        target_size=(256, 256),
        color_mode=color_mode,
        batch_size=1,
        shuffle=False,
        class_mode="input",
        subset="validation",
    )

    # retrieve preprocessed input images as a numpy array
    nb_images = input_generator.samples
    inputs = np.zeros(shape=(nb_images, 256, 256, channels))
    for i in range(nb_images):
        input_image = validation_generator.next()[0]
        input_tensor[i, :, :, :] = input_image

    # get reconstructed images (predictions)
    reconstructions = model.predict_generator(
        validation_generator, steps=validation_generator.samples, verbose=1,
    )

    # compute Residual Maps
    resmaps = utils.residual_maps(inputs, reconstructions, loss=loss)

    # approximate threshold


if __name__ == "__main__":
    main()
