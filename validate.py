import os
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from processing.preprocessing import Preprocessor
from processing.preprocessing import get_preprocessing_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import argparse
import time

from skimage.util import img_as_ubyte

from processing import utils as utils
from processing.resmaps import calculate_resmaps as calculate_resmaps
from processing.cv import threshold_images as threshold_images
from processing.cv import label_images as label_images
from processing.utils import printProgressBar as printProgressBar


def main(args):
    # Get validation arguments
    model_path = args.path
    save = args.save
    min_area = args.area

    # ========================= SETUP ==============================

    # load model and info
    model, info, _ = utils.load_model_HDF5(model_path)

    input_directory = info["data"]["input_directory"]
    architecture = info["model"]["architecture"]
    loss = info["model"]["loss"]
    rescale = info["preprocessing"]["rescale"]
    shape = info["preprocessing"]["shape"]
    color_mode = info["preprocessing"]["color_mode"]
    nb_validation_images = info["data"]["nb_validation_images"]

    val_data_dir = os.path.join(input_directory, "train")

    # create a results directory if not existent
    model_dir_name = os.path.basename(str(Path(model_path).parent))

    save_dir = os.path.join(
        os.getcwd(),
        "results",
        input_directory,
        architecture,
        loss,
        model_dir_name,
        "validation",
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # ============================= PREPROCESSING ===============================
    # get the correct preprocessing function
    preprocessing_function = get_preprocessing_function(architecture)

    # initialize preprocessor
    preprocessor = Preprocessor(
        input_directory=input_directory,
        rescale=rescale,
        shape=shape,
        color_mode=color_mode,
        preprocessing_function=preprocessing_function,
    )

    # get validation generator
    validation_generator = preprocessor.get_val_generator(
        batch_size=nb_validation_images, shuffle=True
    )

    # retrieve validation images from generator
    imgs_val_input = validation_generator.next()[0]

    # retrieve validation image_names
    filenames = validation_generator.filenames

    # get reconstructed images (i.e predictions) on validation dataset
    imgs_val_pred = model.predict(imgs_val_input)

    # compute residual maps on validation dataset
    resmaps_val = calculate_resmaps(imgs_val_input, imgs_val_pred, method="SSIM")

    if color_mode == "rgb":
        resmaps_val = tf.image.rgb_to_grayscale(resmaps_val)

    if args.area is None:
        print("[INFO] exiting")
        exit()

    # Convert to 8-bit unsigned int
    resmaps_val = img_as_ubyte(resmaps_val)

    # ========================= VALIDATION ALGORITHM ==============================

    # set threshold boundaries
    threshold_min = 160  # np.amin(resmaps_val) + 1
    threshold_max = np.amax(resmaps_val)
    thresholds = list(range(threshold_min, threshold_max + 1))

    # initialize progress bar
    l = len(thresholds)
    printProgressBar(0, l, prefix="Progress:", suffix="Complete", length=50)

    for i, threshold in enumerate(thresholds):
        # threshold residual maps
        resmaps_th = threshold_images(resmaps_val, threshold)[:, :, :, 0]

        # compute connected components
        resmaps_labeled, areas_all = label_images(resmaps_th)

        # check if area of largest anomalous region is below the minimum area
        areas_all_flat = [item for sublist in areas_all for item in sublist]
        areas_all_flat.sort(reverse=True)
        try:
            if min_area > areas_all_flat[0]:
                time.sleep(0.1)
                printProgressBar(l, l, prefix="Progress:", suffix="Complete", length=50)
                break
        except IndexError:
            continue

        # print progress bar
        time.sleep(0.1)
        printProgressBar(i + 1, l, prefix="Progress:", suffix="Complete", length=50)

    # save area and threshold pair
    validation_result = {"min_area": min_area, "threshold": threshold}
    print("[INFO] validation results: {}".format(validation_result))

    # save validation result
    with open(os.path.join(save_dir, "validation_result.json"), "w") as json_file:
        json.dump(validation_result, json_file, indent=4, sort_keys=False)
    print("[INFO] validation results saved at {}".format(save_dir))

    if save:
        utils.save_np(imgs_val_input, save_dir, "imgs_val_input.npy")
        utils.save_np(imgs_val_pred, save_dir, "imgs_val_pred.npy")
        utils.save_np(resmaps_val, save_dir, "resmaps_val.npy")


if __name__ == "__main__":

    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )

    parser.add_argument(
        "-a",
        "--area",
        # nargs="+",
        type=int,
        required=True,
        metavar="",
        help="minimum area for a connected component to be classified as anomalous",
    )

    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="save inputs, predictions and reconstructions of validation dataset",
    )

    args = parser.parse_args()

    main(args)

# Example of command to initiate validation

# python3 validate.py -p saved_models/mvtec/capsule/mvtec2/SSIM/19-04-2020_14-14-36/CAE_mvtec2_b8.h5 -a 10
