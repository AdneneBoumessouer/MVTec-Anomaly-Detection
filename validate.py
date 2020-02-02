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

model_path = "saved_models/MSE/02-02-2020_16:32:49/CAE_e150_b12_0"


def main(args):
    model_path = args.model

    # load model and setup
    # model, train_setup, _ = utils.load_SavedModel(model_path)
    model, train_setup, _ = utils.load_model_HDF5(model_path)

    directory = train_setup["directory"]
    val_data_dir = os.path.join(directory, "train")
    color_mode = train_setup["color_mode"]
    validation_split = train_setup["validation_split"]
    channels = train_setup["channels"]
    loss = train_setup["loss"]
    batch_size = train_setup["batch_size"]

    # This will do preprocessing
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255, validation_split=validation_split, zca_epsilon=1e-06,
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
        input_image = input_generator.next()[0]
        inputs[i, :, :, :] = input_image

    # get reconstructed images (predictions)
    reconstructions = model.predict_generator(
        validation_generator,
        steps=validation_generator.samples / batch_size,
        verbose=1,
    )

    recon = model.predict(inputs)

    index = 1
    plt.imshow(inputs[index])
    img_original = tf.expand_dims(inputs[index], 0)
    img_reconstruction = model.predict(img_original)
    plt.imshow(img_reconstruction[0])

    utils.compare_images(inputs[index], reconstructions[index])

    # compute Residual Maps
    resmaps = utils.residual_maps(inputs, reconstructions, loss=loss)

    # determine threshold

    # save threshold


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, required=True, metavar="", help="path to existing model"
)

args = parser.parse_args()

if __name__ == "__main__":
    main()
