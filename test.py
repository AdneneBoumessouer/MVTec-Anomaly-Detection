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


def get_image_score(image, factor):
    image_1d = image.flatten()
    mean_image = np.mean(image_1d)
    std_image = np.std(image_1d)
    score = mean_image + factor * std_image
    return score, mean_image, std_image


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

    # create directory to save test results
    parent_dir = str(Path(model_path).parent)
    save_dir = os.path.join(parent_dir, "test_results")
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

    test_data_dir = os.path.join(directory, "test")
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
    np.save(
        file=os.path.join(save_dir, "imgs_test_input.npy"),
        arr=imgs_test_input,
        allow_pickle=True,
    )

    # retrieve image_names
    filenames = test_generator.filenames

    # predict on test images
    imgs_test_pred = model.predict(imgs_test_input)
    np.save(
        file=os.path.join(save_dir, "imgs_test_pred.npy"),
        arr=imgs_test_pred,
        allow_pickle=True,
    )
    print("TYPE imgs_test_pred:{}".format(type(imgs_test_pred)))

    # ===============================================================
    # ===============================================================

    # compute residual maps
    imgs_test_diff = imgs_test_input - imgs_test_pred
    np.save(
        file=os.path.join(save_dir, "imgs_test_diff.npy"),
        arr=imgs_test_diff,
        allow_pickle=True,
    )

    # determine threshold test
    imgs_test_diff_1d = imgs_test_diff.flatten()
    mean_test = np.mean(imgs_test_diff_1d)
    std_test = np.std(imgs_test_diff_1d)

    # k = 1.65 # confidence 90%
    # k = 1.96 # confidence 95%
    factor_test = 2.58  # confidence 99%
    # k = 3.00 # confidence 99.73%
    # k = 3.30 # confidence 99.90%

    threshold_test = mean_test + factor_test * std_test

    # Histogramm to visualize the ResMap distribution
    fig = plt.figure(figsize=(8, 5))
    plt.hist(imgs_test_diff_1d, bins=100, density=True, stacked=True)
    plt.title("Test ResMap pixel value distribution")
    plt.xlabel("pixel intensity")
    plt.ylabel("probability")
    plt.savefig(os.path.join(save_dir, "histogram_test.png"))

    # show reconstruction for one particular image
    test_image_name = args.image  # change name in the future
    index = filenames.index(test_image_name)
    print(filenames[index])

    fig, axarr = plt.subplots(3, 1, figsize=(5, 18))
    try:
        axarr[0].imshow(imgs_test_input[index])
    except TypeError:
        axarr[0].imshow(imgs_test_input[index, :, :, 0], cmap=plt.cm.gray)
    axarr[0].set_title("original defect test image")
    try:
        axarr[1].imshow(imgs_test_pred[index])
    except:
        axarr[1].imshow(imgs_test_pred[index, :, :, 0], cmap=plt.cm.gray)
    axarr[1].set_title("reconstruction defect test image")
    try:
        axarr[2].imshow(imgs_test_diff[index])
    except:
        axarr[2].imshow(imgs_test_diff[index, :, :, 0], cmap=plt.cm.gray)
    axarr[2].set_title("ResMap defect test image")
    fig.savefig(os.path.join(save_dir, "3_test_musketeers.png"))

    # ===============================================================
    # ===============================================================

    # compute scores on test images
    output_test = {"filenames": filenames, "scores": [], "mean": [], "std": []}
    for img_test_diff in imgs_test_diff:
        score, mean, std = utils.get_image_score(img_test_diff, factor_test)
        output_test["scores"].append(score)
        output_test["mean"].append(mean)
        output_test["std"].append(std)
        # assert length compatibility

    # format test results in a pd DataFrame
    df_test = pd.DataFrame.from_dict(output_test)
    df_test.to_pickle(os.path.join(save_dir, "df_test.pkl"))
    # display DataFrame
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_test)

    # get path to validation results directory
    parent_dir = str(Path(model_path).parent)
    val_result_dir = os.path.join(parent_dir, "val_results")
    if not os.path.isdir(val_result_dir):
        raise Exception

    # get threshold from validation results
    with open(os.path.join(val_result_dir, "val_results.json"), "r") as read_file:
        val_results = json.load(read_file)

    threshold_val = float(val_results["threshold_val"])
    mean_val = float(val_results["mean_val"])
    std_val = float(val_results["std_val"])

    # ======================= CLASSIFICATION ===========================
    # classification: defective if score exceeds threshold, positive class == defect
    scores = np.array(output_test["scores"])
    y_pred_bool = scores > threshold_val
    y_pred_class = [
        "defect" if boolean == True else "defect_free" for boolean in y_pred_bool
    ]

    y_true_bool = [
        True if "good" not in filename.split("/") else False for filename in filenames
    ]
    y_true_class = [
        "defect" if boolean == True else "defect_free" for boolean in y_true_bool
    ]

    # detection ratios: https://en.wikipedia.org/wiki/Confusion_matrix
    # condition positive (P)
    P = y_true_bool.count(True)

    # condition negative (N)
    N = y_true_bool.count(False)

    # true positive (TP)
    TP = np.sum(
        [
            1 if y_pred_bool[i] == y_true_bool[i] == True else 0
            for i in range(total_number)
        ]
    )

    # true negative (TN)
    TN = np.sum(
        [
            1 if y_pred_bool[i] == y_true_bool[i] == False else 0
            for i in range(total_number)
        ]
    )

    # sensitivity, recall, hit rate, or true positive rate (TPR)
    TPR = TP / P

    # specificity, selectivity or true negative rate (TNR)
    TNR = TN / N

    # confusion matrix
    conf_matrix = confusion_matrix(
        y_true_class, y_pred_class, labels=["defect", "defect_free"]
    )  # normalize="true"

    # save test results
    test_results = {
        "mean_val": str(mean_val),
        "mean_test": str(mean_test),
        "std_val": str(std_val),
        "std_test": str(std_test),
        "threshold_val": str(threshold_val),
        "threshold_test": str(threshold_test),
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
parser.add_argument(
    "-i", "--image", type=str, required=True, metavar="", help="path to test image"
)

# parser_new_training.add_argument(
#     "-c", "--comment", type=str, help="write comment regarding training")
args = parser.parse_args()

if __name__ == "__main__":
    main(args)

# python3 test.py -p saved_models/MSE/21-02-2020_17:47:13/CAE_mvtec_b12.h5 -i "poke/000.png"
