import os
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import custom_loss_functions
import utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy import expand_dims

from sklearn.metrics import confusion_matrix

import requests
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import datetime
import csv
import pandas as pd
import json

import argparse
from skimage.util import img_as_ubyte

# import validation functions
from validate import threshold_images as threshold_images
from validate import label_images as label_images


def is_defective(areas, min_area):
    """Decides if image is defective given the areas of its connected components"""
    areas = np.array(areas)
    if areas[areas >= min_area].shape[0] > 0:
        return 1
    return 0


def classify(areas_all, min_area):
    """Decides if images are defective given the areas of their connected components"""
    y_pred = []
    for areas in areas_all:
        y_pred.append(is_defective(areas, min_area))
    return y_pred


def main(args):
    model_path = args.path
    threshold = args.threshold
    min_area = args.area
    save = args.save

    # load model, setup and history
    model, setup, history = utils.load_model_HDF5(model_path)

    # data setup
    directory = setup["data_setup"]["directory"]
    test_data_dir = os.path.join(directory, "test")
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

    if threshold == None or min_area == None:
        # get path to validation results directory
        parent_dir = str(Path(model_path).parent)
        val_result_dir = os.path.join(parent_dir, "val_results")
        if not os.path.isdir(val_result_dir):
            raise Exception

        # get threshold and area from validation results
        with open(os.path.join(val_result_dir, "val_results.json"), "r") as read_file:
            val_results = json.load(read_file)

        if threshold == None:
            # should be parsed ARGUMENT
            threshold = float(val_results["threshold"])
            print("using validation threshold")
        if min_area == None:
            min_area = float(val_results["area"])  # should be parsed ARGUMENT
            print("using validation area")

    # create directory to save test results
    model_dir_name = os.path.basename(str(Path(model_path).parent))
    # now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    # save_dir = os.path.join(os.getcwd(), "results",
    #                         model_dir_name, "test", now)

    save_dir = os.path.join(
        os.getcwd(),
        "results",
        directory,
        architecture,
        loss,
        model_dir_name,
        "test",
        "th_" + str(threshold) + "_a_" + str(min_area),
    )

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # This will do preprocessing
    if architecture in ["mvtec", "mvtec2"]:
        preprocessing_function = None
    elif architecture == "resnet":
        preprocessing_function = keras.applications.inception_resnet_v2.preprocess_input
    elif architecture == "nasnet":
        preprocessing_function = keras.applications.nasnet.preprocess_input

    test_datagen = ImageDataGenerator(
        rescale=rescale,
        data_format="channels_last",
        preprocessing_function=preprocessing_function,
    )

    total_number = utils.get_total_number_test_images(test_data_dir)

    # retrieve preprocessed test images as a numpy array
    test_generator = test_datagen.flow_from_directory(
        directory=test_data_dir,
        target_size=shape,
        color_mode=color_mode,
        batch_size=total_number,
        shuffle=False,
        class_mode="input",
    )
    imgs_test_input = test_generator.next()[0]

    # predict on test images
    imgs_test_pred = model.predict(imgs_test_input)

    # converts rgb to grayscale
    if color_mode == "rgb":
        imgs_val_input = tf.image.rgb_to_grayscale(imgs_val_input)
        imgs_val_pred = tf.image.rgb_to_grayscale(imgs_val_pred)

    # compute residual maps on test set
    resmaps_test = imgs_test_input - imgs_test_pred

    if save:
        utils.save_np(imgs_test_input, save_dir, "imgs_val_input.npy")
        utils.save_np(imgs_test_pred, save_dir, "imgs_val_pred.npy")
        utils.save_np(resmaps_test, save_dir, "resmaps_val.npy")

    # retrieve test image_names
    filenames = test_generator.filenames

    # Convert to 8-bit unsigned int
    # (unnecessary if working exclusively with scikit image, see .img_as_float())
    # must be consistent with validation
    resmaps_test = img_as_ubyte(resmaps_test)

    # threshold residual maps with the given threshold
    resmaps_th = threshold_images(resmaps_test, threshold)

    # compute connected components
    resmaps_labeled, areas_all = label_images(resmaps_th)

    # classify images
    y_pred = classify(areas_all, min_area)

    # retrieve ground truth
    y_true = [1 if "good" not in filename.split("/") else 0 for filename in filenames]

    # format test results in a pd DataFrame
    classification = {"filenames": filenames, "predictions": y_pred, "truth": y_true}
    df_clf = pd.DataFrame.from_dict(classification)
    # df_clf.to_pickle(os.path.join(save_dir, "df_clf.pkl"))
    # with open(os.path.join(save_dir, "classification.txt"), "a") as f:
    #     f.write(df_clf.to_string(header=True, index=True))

    # print DataFrame to console
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_clf)

    # detection ratios: https://en.wikipedia.org/wiki/Confusion_matrix
    # condition positive (P)
    P = y_true.count(1)

    # condition negative (N)
    N = y_true.count(0)

    # true positive (TP)
    TP = np.sum([1 if y_pred[i] == y_true[i] == 1 else 0 for i in range(total_number)])

    # true negative (TN)
    TN = np.sum(
        [1 if y_pred[i] == y_true[i] == False else 0 for i in range(total_number)]
    )

    # sensitivity, recall, hit rate, or true positive rate (TPR)
    TPR = TP / P

    # specificity, selectivity or true negative rate (TNR)
    TNR = TN / N

    # confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, normalize="true").ravel()

    # save test results
    test_results = {
        "TPR": TPR,
        "TNR": TNR,
        "threshold": threshold,
        "min_area": min_area,
    }

    with open(os.path.join(save_dir, "test_results.json"), "w") as json_file:
        json.dump(test_results, json_file)

    print("test_results: \n{}".format(test_results))
    print()
    print("confusion matrix: \n{}".format(conf_matrix))


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=None,
        metavar="",
        help="number of training images",
    )
    parser.add_argument(
        "-a",
        "--area",
        type=int,
        default=None,
        metavar="",
        help="number of training images",
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

# Examples of command to initiate testing

# using threshold and area from validation results:
# python3 test.py -p saved_models/MSE/25-02-2020_08:54:06/CAE_mvtec2_b12.h5

# using passed arguments for threshold and area
# python3 test.py -p saved_models/MSE/25-02-2020_08:54:06/CAE_mvtec2_b12.h5 -t 28 -a 50
