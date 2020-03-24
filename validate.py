import os
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

from modules import utils as utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import argparse

from skimage.util import img_as_ubyte
from modules.cv import scale_pixel_values as scale_pixel_values
from modules.cv import filter_gauss_images as filter_gauss_images
from modules.cv import filter_median_images as filter_median_images
from modules.cv import threshold_images as threshold_images
from modules.cv import label_images as label_images


def main(args):
    model_path = args.path
    min_area = args.area
    save = args.save

    # load model, setup and history
    model, setup, history = utils.load_model_HDF5(model_path)

    # data setup
    directory = setup["data_setup"]["directory"]
    val_data_dir = os.path.join(directory, "train")
    nb_training_images = setup["data_setup"]["nb_training_images"]
    nb_validation_images = setup["data_setup"]["nb_validation_images"]

    # preprocessing_setup
    rescale = setup["preprocessing_setup"]["rescale"]
    shape = setup["preprocessing_setup"]["shape"]
    preprocessing = setup["preprocessing_setup"]["preprocessing"]

    # train_setup
    color_mode = setup["train_setup"]["color_mode"]
    learning_rate = setup["train_setup"]["learning_rate"]
    decay = setup["train_setup"]["decay"]
    epochs_trained = setup["train_setup"]["epochs_trained"]
    nb_training_images_aug = setup["train_setup"]["nb_training_images_aug"]
    epochs = setup["train_setup"]["epochs"]
    batch_size = setup["train_setup"]["batch_size"]
    channels = setup["train_setup"]["channels"]
    validation_split = setup["train_setup"]["validation_split"]
    architecture = setup["train_setup"]["architecture"]
    loss = setup["train_setup"]["loss"]

    tag = setup["tag"]

    # create a results directory if not existent
    model_dir_name = os.path.basename(str(Path(model_path).parent))

    save_dir = os.path.join(
        os.getcwd(),
        "results",
        directory,
        architecture,
        loss,
        model_dir_name,
        "validation",
        "a_" + str(min_area),
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Plot and save loss and val_loss
    plot = pd.DataFrame(history[["loss", "val_loss"]]).plot(figsize=(8, 5))
    fig = plot.get_figure()
    fig.savefig(os.path.join(save_dir, "train_val_losses.png"))

    # This will do preprocessing
    if architecture in ["mvtec", "mvtec2"]:
        preprocessing_function = None
    elif architecture == "resnet":
        preprocessing_function = keras.applications.inception_resnet_v2.preprocess_input
    elif architecture == "nasnet":
        preprocessing_function = keras.applications.nasnet.preprocess_input

    # same preprocessing as in training
    validation_datagen = ImageDataGenerator(
        rescale=rescale,
        data_format="channels_last",
        validation_split=validation_split,
        preprocessing_function=preprocessing_function,
    )

    # retrieve preprocessed validation images as a numpy array
    validation_generator = validation_datagen.flow_from_directory(
        directory=val_data_dir,
        target_size=shape,
        color_mode=color_mode,
        batch_size=nb_validation_images,
        shuffle=False,
        class_mode="input",
        subset="validation",
    )
    imgs_val_input = validation_generator.next()[0]

    # retrieve validation image_names
    filenames = validation_generator.filenames

    # get reconstructed images (i.e predictions) on validation dataset
    imgs_val_pred = model.predict(imgs_val_input)

    # compute residual maps on validation dataset
    resmaps_val = imgs_val_input - imgs_val_pred

    if color_mode == "rgb":
        resmaps_val = tf.image.rgb_to_grayscale(resmaps_val)

    if save:
        utils.save_np(imgs_val_input, save_dir, "imgs_val_input.npy")
        utils.save_np(imgs_val_pred, save_dir, "imgs_val_pred.npy")
        utils.save_np(resmaps_val, save_dir, "resmaps_val.npy")

    # scale pixel values linearly to [0,1]
    resmaps_val = scale_pixel_values(architecture, resmaps_val)

    # Convert to 8-bit unsigned int
    resmaps_val = img_as_ubyte(resmaps_val)

    threshold_min = np.amin(resmaps_val)
    threshold_max = np.amax(resmaps_val)

    for threshold in range(threshold_min, threshold_max):
        # threshold residual maps
        resmaps_th = threshold_images(resmaps_val, threshold)
        print("current threshold = {}".format(threshold))

        # filter images to remove salt noise
        resmaps_fil = filter_median_images(resmaps_th, kernel_size=3)

        # compute connected components
        resmaps_labeled, areas_all = label_images(resmaps_fil)

        # check if area of largest anomalous region is below the minimum area
        areas_all_flat = [item for sublist in areas_all for item in sublist]
        areas_all_flat.sort(reverse=True)
        if min_area > areas_all_flat[0]:
            break

    # save threshold value and min_area
    val_results = {
        "threshold": str(threshold),
        "area": str(min_area),
    }
    print(val_results)

    with open(os.path.join(save_dir, "val_results.json"), "w") as json_file:
        json.dump(val_results, json_file)


if __name__ == "__main__":

    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )
    parser.add_argument(
        "-a",
        "--area",
        type=int,
        required=True,
        metavar="",
        help="minimum area for a connected component",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=bool,
        required=False,
        default=False,
        metavar="",
        help="save inputs, predictions and reconstructions of validation dataset",
    )
    args = parser.parse_args()

    main(args)

# Example of command to initiate validation

# python3 validate.py -p saved_models/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/CAE_mvtec2_b12.h5 -a 55
