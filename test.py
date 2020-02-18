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


def get_image_score(image, k):
    image_1d = image.flatten()
    mean_image = np.mean(image_1d)
    std_image = np.std(image_1d)
    score = mean_image + k*std_image
    return score


def main(args):
    model_path = args.path

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
    epochs_trained = setup["train_setup"]["epochs_trained"]
    nb_training_images_aug = setup["train_setup"]["nb_training_images_aug"]
    epochs = setup["train_setup"]["epochs"]
    batch_size = setup["train_setup"]["batch_size"]
    channels = setup["train_setup"]["channels"]
    validation_split = setup["train_setup"]["validation_split"]
    architecture = setup["train_setup"]["architecture"]
    loss = setup["train_setup"]["loss"]

    # comment
    comment = setup["comment"]

    # This will do preprocessing
    if architecture == "mvtec":
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

    test_data_dir = os.path.join(directory, "test")
    total_number = utils.get_total_number_test_images(test_data_dir)

    # retrieve preprocessed input images as a numpy array
    test_generator = test_datagen.flow_from_directory(
        directory=test_data_dir,
        target_size=shape,
        color_mode=color_mode,
        batch_size=total_number,
        shuffle=False,
        class_mode="input",
    )
    imgs_test_input = test_generator.next()[0]

    # retrieve image_names
    filenames = test_generator.filenames

    # predict on test images
    imgs_test_pred = model.predict(imgs_test_input)

    # compute scores on test images
    scores = np.array([get_image_score(img_test_pred)
                       for img_test_pred in imgs_test_pred])

    # get path to validation results directory
    parent_dir = Path(model_path).parent
    val_result_dir = os.path.join(parent_dir, "val_results")
    if not os.path.isdir(val_result_dir):
        raise Exception

    # get threshold from validation results
    with open(os.path.join(val_result_dir, "val_results.json"), "r") as read_file:
        val_results = json.load(read_file)

    threshold = val_results["threshold"]

    # classification: defective if score exceeds threshold, positive class == defect
    y_pred_bool = scores > threshold
    y_pred_class = ["defect" if boolean ==
                    True else "defect_free" for boolean in y_pred_bool]

    y_true_bool = [True if "good" not in filename.split(
        "/") else False for filename in filenames]
    y_true_class = ["defect" if boolean ==
                    True else "defect_free" for boolean in y_true_bool]

    # detection ratios: https://en.wikipedia.org/wiki/Confusion_matrix
    # condition positive (P)
    P = y_true_bool.count(True)

    # condition negative (N)
    N = y_true_bool.count(False)

    # true positive (TP)
    TP = np.sum([1 if y_pred_bool[i] == y_true_bool[i] ==
                 True else 0 for i in range(total_number)])

    # true negative (TN)
    TN = np.sum([1 if y_pred_bool[i] == y_true_bool[i] ==
                 False else 0 for i in range(total_number)])

    # sensitivity, recall, hit rate, or true positive rate (TPR)
    TPR = TP / P

    # specificity, selectivity or true negative rate (TNR)
    TNR = TN / N

    # confusion matrix
    conf_matrix = confusion_matrix(
        y_true_class, y_pred_class, labels=["defect", "defect_free"], normalize="true")

    # create directory to save test_results
    parent_dir = Path(model_path).parent
    save_dir = os.path.join(parent_dir, "test_results")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save test results
    test_results = {
        "TPR": TPR,
        "TNR": TNR,
    }

    with open(os.path.join(save_dir, "test_results.json"), "w") as json_file:
        json.dump(test_results, json_file)

    print("test_results: \n{}".format(test_results))
    print()
    print("confusion matrix: \n{}".format(conf_matrix))


# create parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
)
args = parser.parse_args()

if __name__ == "__main__":
    main(args)

model_path = "saved_models/MSE/02-02-2020_16:32:49/CAE_e150_b12_0"
