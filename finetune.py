from skimage.util import img_as_ubyte
from validate import label_images as label_images
from validate import threshold_images as threshold_images
import argparse
import json
import pandas as pd
import csv
import datetime
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
import scipy.stats

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main(args):
    # Get finetuning parameters
    model_path = args.path
    save = args.save
    img_val = args.val
    img_test = args.test
    thresholds_to_plot = list(set(args.list))
    thresholds_to_plot.sort()

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

    # create directory to save results
    model_dir_name = os.path.basename(str(Path(model_path).parent))
    save_dir = os.path.join(
        os.getcwd(),
        "results",
        directory,
        architecture,
        loss,
        model_dir_name,
        "finetune",
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

    # get reconstructed images (i.e predictions) on validation dataset
    print("computing reconstructions of validation images...")
    imgs_val_pred = model.predict(imgs_val_input)

    # converts rgb to grayscale
    # if color_mode == "rgb":
    #     imgs_val_input = tf.image.rgb_to_grayscale(imgs_val_input)
    #     imgs_val_pred = tf.image.rgb_to_grayscale(imgs_val_pred)

    # compute residual maps on validation dataset
    resmaps_val = imgs_val_input - imgs_val_pred
    if color_mode == "rgb":
        resmaps_val = tf.image.rgb_to_grayscale(resmaps_val)

    if save:
        utils.save_np(imgs_val_input, save_dir, "imgs_val_input.npy")
        utils.save_np(imgs_val_pred, save_dir, "imgs_val_pred.npy")
        utils.save_np(resmaps_val, save_dir, "resmaps_val.npy")

    # plot a sample validation image alongside its corresponding reconstruction and resmap for inspection
    if img_val != None:
        plt.style.use("default")
        # compute image index
        index_val = validation_generator.filenames.index(img_val)
        fig = utils.plot_input_pred_resmaps_val(
            imgs_val_input, imgs_val_pred, resmaps_val, index_val
        )
        fig.savefig(os.path.join(save_dir, "val_plots.png"))
        print("figure saved at {}".format(os.path.join(save_dir, "val_plots.png")))

    # Convert to 8-bit unsigned int for further processing
    # (unnecessary if working exclusively with scikit image, see .img_as_float())
    resmaps_val = img_as_ubyte(resmaps_val)

    nb_regions = []
    mean_area_size = []
    std_area_size = []
    threshold_max = np.amax(resmaps_val)

    # compute and plot number of anomalous regions and their area sizes with increasing thresholds
    print("computing anomalous regions and area sizes with increasing thresholds...")
    for threshold in range(threshold_max):
        # threshold residual maps
        resmaps_th = threshold_images(resmaps_val, threshold)

        # compute anomalous regions
        resmaps_labeled, areas_all = label_images(resmaps_th)

        # check if area of largest anomalous region is below the minimum area
        areas_all_flat = [item for sublist in areas_all for item in sublist]
        nb_regions.append(len(areas_all_flat))
        mean_area_size.append(np.mean(areas_all_flat))
        std_area_size.append(np.std(areas_all_flat))

        print("current threshold: {}".format(threshold))

    plt.style.use("seaborn-darkgrid")

    df_out = pd.DataFrame.from_dict(
        {
            "threshold": list(range(np.amax(resmaps_val))),
            "nb_regions": nb_regions,
            "mean_area_size": mean_area_size,
            "std_area_size": std_area_size,
        }
    )
    df_out.plot(
        x="threshold",
        y=["mean_area_size", "std_area_size"],
        xticks=np.arange(start=0, stop=threshold_max, step=5, dtype=int),
        title="mean and standard deviation of areas with increasing thresholds",
        figsize=(8, 5),
    )
    plt.savefig(os.path.join(save_dir, "stat_area.pdf"))
    # plt.show()
    df_out.plot(
        x="threshold",
        y=["nb_regions"],
        xticks=np.arange(start=0, stop=threshold_max, step=5, dtype=int),
        title="number of regions with increasing thresholds",
        figsize=(8, 5),
    )
    plt.savefig(os.path.join(save_dir, "nb_of_regions.pdf"))
    # plt.show()

    # ===================================================================
    # ===================================================================

    # COMMENT -----------------------------------------------------------
    # resmaps_val = np.load(
    #     "results/25-02-2020_08:54:06/validation/02-03-2020_11:13:25/resmaps_val.npy",
    #     allow_pickle=True,
    # )

    # Convert to 8-bit unsigned int
    # (unnecessary if working exclusively with scikit image, see .img_as_float())
    # resmaps_val = img_as_ubyte(resmaps_val)
    # -------------------------------------------------------------------

    counts = []
    nb_bins = 500
    max_pixel_value = np.amax(resmaps_val)

    # compute residual maps for threshold = 0
    resmaps_th = threshold_images(images=resmaps_val, threshold=0)

    # compute anomalous regions
    resmaps_labeled, areas_all = label_images(resmaps_th)

    # flatten area
    areas_all_flat = [item for sublist in areas_all for item in sublist]
    max_range_hist = np.amax(np.array(areas_all_flat))  # for plotting

    # compute and plot distribution of anomaly areas's sizes
    fig4 = plt.figure(num=4, figsize=(12, 8))
    count, bins, ignored = plt.hist(areas_all_flat, bins=nb_bins, density=False,)
    plt.title(
        "Distribution of anomaly areas' sizes for validation ResMaps with Threshold = 0"
    )
    plt.xlabel("area size in pixel")
    plt.ylabel("count")
    # plt.show()
    fig4.savefig(os.path.join(save_dir, "distr_area_th_0.pdf"))

    # compute residual maps for multiple thresholds
    for threshold in thresholds_to_plot:
        # threshold residual maps
        resmaps_th = threshold_images(resmaps_val, threshold)

        # compute anomalous regions
        resmaps_labeled, areas_all = label_images(resmaps_th)

        # flatten area
        areas_all_flat = [item for sublist in areas_all for item in sublist]

        fig5 = plt.figure(num=5, figsize=(12, 5))
        count, bins, ignored = plt.hist(
            areas_all_flat,
            bins=nb_bins,
            density=False,
            range=(0, max_range_hist),
            histtype="barstacked",
            label="threshold = {}".format(threshold),
        )

        # count, edges = np.histogram(
        #     areas_all_flat, bins=nb_bins, density=False, range=(0, max_range_hist)
        # )
        # bins_middle = edges[:-1] + ((edges[1] - edges[0]) / 2)
        # plt.plot(
        #     bins_middle,
        #     count,
        #     linestyle="-",
        #     linewidth=0.5,
        #     # marker="o",
        #     # markersize=0.5,
        #     label="threshold = {}".format(threshold),
        # )
        # plt.fill_between(bins_middle, count)

    plt.title(
        "Distribution of anomaly areas' sizes for validation ResMaps with various Thresholds"
    )
    plt.legend()
    plt.xlabel("area size in pixel")
    plt.ylabel("count")
    plt.show(block=True)
    fig5.savefig(os.path.join(save_dir, "distr_area_th_multiple.pdf"))

    # plot a sample test image alongside its corresponding reconstruction and resmap for inspection
    if img_test != None:
        plt.style.use("default")
        test_data_dir = os.path.join(directory, "test")
        total_number = utils.get_total_number_test_images(test_data_dir)

        test_datagen = ImageDataGenerator(
            rescale=rescale,
            data_format="channels_last",
            preprocessing_function=preprocessing_function,
        )

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
        print("computing reconstructions of validation images...")
        imgs_test_pred = model.predict(imgs_test_input)

        # compute residual maps on test set
        resmaps_test = imgs_test_input - imgs_test_pred

        if color_mode == "rgb":
            resmaps_test = tf.image.rgb_to_grayscale(resmaps_test)

        # compute image index
        index_test = test_generator.filenames.index(img_test)

        # save three images
        fig = utils.plot_input_pred_resmaps_test(
            imgs_test_input, imgs_test_pred, resmaps_test, index_test
        )
        fig.savefig(os.path.join(save_dir, "test_plots.png"))
        print("figure saved at {}".format(os.path.join(save_dir, "test_plots.png")))


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
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
    parser.add_argument(
        "-v",
        "--val",
        type=str,
        default=None,
        metavar="",
        help="path to sample test image relative to validation directory for visualization",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=str,
        default=None,
        metavar="",
        help="path to sample test image relative to test directory for visualization",
    )
    parser.add_argument(
        "-l",
        "--list",
        nargs="+",
        type=int,
        required=False,
        default=[20, 30, 40],
        metavar="",
        help="thresholds to plot",
    )
    args = parser.parse_args()

    main(args)


# model_path = "saved_models/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/CAE_mvtec2_b12.h5"


# python3 finetune.py -p saved_models/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/CAE_mvtec2_b12.h5 -v "good/000.png" -t "poke/000.png"
# python3 finetune.py -p saved_models/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/CAE_mvtec2_b12.h5 -l 20 30 40
