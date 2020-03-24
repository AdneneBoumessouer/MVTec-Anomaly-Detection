import os
import sys
from pathlib import Path

import argparse
import json
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt

from skimage.util import img_as_ubyte
from modules.cv import scale_pixel_values as scale_pixel_values
from modules.cv import filter_gauss_images as filter_gauss_images
from modules.cv import filter_median_images as filter_median_images
from modules.cv import threshold_images as threshold_images
from modules.cv import label_images as label_images


def main(args):
    # Get finetuning parameters
    model_path = args.path
    save = args.save
    img_val = args.val
    img_test = args.test
    thresholds_to_plot = list(set(args.list))
    thresholds_to_plot.sort()

    # ========================= SETUP ==============================

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

    # ============================= PREPROCESSING ===============================

    # get preprocessing function corresponding to model
    preprocessing_function = utils.get_preprocessing_function(architecture)

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

    # ============== RECONSTRUCT IMAGES AND COMPUTE RESIDUAL MAPS ==============

    # get reconstructed images (i.e predictions) on validation dataset
    print("computing reconstructions of validation images...")
    imgs_val_pred = model.predict(imgs_val_input)

    # compute residual maps on validation dataset
    resmaps_val = imgs_val_input - imgs_val_pred

    if color_mode == "rgb":
        resmaps_val = tf.image.rgb_to_grayscale(resmaps_val)

    if save:
        utils.save_np(imgs_val_input, save_dir, "imgs_val_input.npy")
        utils.save_np(imgs_val_pred, save_dir, "imgs_val_pred.npy")
        utils.save_np(resmaps_val, save_dir, "resmaps_val.npy")

    # plot a sample validation image alongside its corresponding reconstruction and resmap for inspection
    if img_val is not None:
        plt.style.use("default")
        # compute image index
        index_val = validation_generator.filenames.index(img_val)
        fig = utils.plot_input_pred_resmaps_val(
            imgs_val_input, imgs_val_pred, resmaps_val, index_val
        )
        fig.savefig(os.path.join(save_dir, "val_plots.png"))
        print("figure saved at {}".format(os.path.join(save_dir, "val_plots.png")))

    # scale pixel values linearly to [0,1]
    resmaps_val = scale_pixel_values(architecture, resmaps_val)

    # Convert to 8-bit unsigned int for further processing
    resmaps_val = img_as_ubyte(resmaps_val)

    # ======= NUMBER OF REGIONS, THEIR MEAN SIZE AND STD DEVIATION WITH INCREASING THRESHOLDS =======

    dict_out = {
        "threshold": [],
        "nb_regions": [],
        "mean_areas_size": [],
        "std_areas_size": [],
        "sum_areas_size": [],
    }

    # compute descriptive values
    min_pixel_value = np.amin(resmaps_val)
    max_pixel_value = np.amax(resmaps_val)
    mu = resmaps_val.flatten().mean()
    sigma = resmaps_val.flatten().std()

    # set relevant threshold interval
    threshold_min = int(round(scipy.stats.norm(mu, sigma).ppf(0.97), 1)) - 1
    threshold_max = max_pixel_value

    # compute and plot number of anomalous regions and their area sizes with increasing thresholds
    print("computing anomalous regions and area sizes with increasing thresholds...")
    for threshold in range(threshold_min, threshold_max):
        # threshold residual maps
        resmaps_th = threshold_images(resmaps_val, threshold)

        # filter images to remove salt noise
        resmaps_fil = filter_median_images(resmaps_th, kernel_size=3)

        # compute anomalous regions for current threshold
        resmaps_labeled, areas_all = label_images(resmaps_fil)

        # compute characteristics
        areas_all_1d = [item for sublist in areas_all for item in sublist]

        nb_regions = len(areas_all_1d)
        if nb_regions == 0:
            break

        mean_areas_size = np.mean(areas_all_1d)
        std_areas_size = np.std(areas_all_1d)
        sum_areas_size = np.sum(areas_all_1d)

        # append values to dictionnary
        dict_out["threshold"].append(threshold)
        dict_out["nb_regions"].append(nb_regions)
        dict_out["mean_areas_size"].append(mean_areas_size)
        dict_out["std_areas_size"].append(std_areas_size)
        dict_out["sum_areas_size"].append(sum_areas_size)
        print("threshold: {}".format(threshold))

    plt.style.use("seaborn-darkgrid")

    # print DataFrame to console
    df_out = pd.DataFrame.from_dict(dict_out)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_out)

    # save DataFrame (as text AND as pkl)
    with open(os.path.join(save_dir, "test_results_all.txt"), "a") as f:
        f.truncate(0)
        f.write(df_out.to_string(header=True, index=True))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(df_out.threshold, df_out.sum_areas_size, "#1f77b4")
    ax1.set_ylabel("sum of anomalous region's area size", color="#1f77b4")
    for tl in ax1.get_yticklabels():
        tl.set_color("#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(df_out.threshold, df_out.nb_regions, "#ff7f0e")
    ax2.set_ylabel("number of anomalous regions", color="#ff7f0e")
    for tl in ax2.get_yticklabels():
        tl.set_color("#ff7f0e")

    fig.savefig(os.path.join(save_dir, "plot_stat.pdf"))

    # plt.show()

    # plot a sample test image alongside its corresponding reconstruction and resmap for inspection
    if img_test is not None:
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
        default=[133, 143, 153],
        metavar="",
        help="thresholds to plot",
    )
    args = parser.parse_args()

    main(args)


# model_path = "saved_models/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/CAE_mvtec2_b12.h5"


# python3 finetune.py -p saved_models/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/CAE_mvtec2_b12.h5 -v "good/000.png" -t "poke/000.png"
# python3 finetune.py -p saved_models/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/CAE_mvtec2_b12.h5 -l 113 143 153

# ===================================================================

# counts = []
# nb_bins = 500
# max_pixel_value = np.amax(resmaps_val)

# # compute residual maps for threshold = 127 # 113
# resmaps_th = threshold_images(images=resmaps_val, threshold=127)  # 113

# # compute anomalous regions
# resmaps_labeled, areas_all = label_images(resmaps_th)

# # flatten area
# areas_all_1d = [item for sublist in areas_all for item in sublist]
# max_range_hist = np.amax(np.array(areas_all_1d))  # for plotting

# # compute and plot distribution of anomaly regions' sizes
# fig4 = plt.figure(num=4, figsize=(12, 8))
# count, bins, ignored = plt.hist(areas_all_1d, bins=nb_bins, density=False,)
# plt.title(
#     "Distribution of anomaly regions' sizes for validation ResMaps with Threshold = 127"
# )
# plt.xlabel("area size in pixel")
# plt.ylabel("count")
# # plt.show()
# fig4.savefig(os.path.join(save_dir, "distr_area_th_0.pdf"))

# # compute residual maps for multiple thresholds
# for threshold in thresholds_to_plot:
#     # threshold residual maps
#     resmaps_th = threshold_images(resmaps_val, threshold)

#     # compute anomalous regions
#     resmaps_labeled, areas_all = label_images(resmaps_th)

#     # flatten area
#     areas_all_1d = [item for sublist in areas_all for item in sublist]

#     fig5 = plt.figure(num=5, figsize=(12, 5))
#     count, bins, ignored = plt.hist(
#         areas_all_1d,
#         bins=nb_bins,
#         density=False,
#         range=(0, max_range_hist),
#         histtype="barstacked",
#         label="threshold = {}".format(threshold),
#     )

#     # count, edges = np.histogram(
#     #     areas_all_1d, bins=nb_bins, density=False, range=(0, max_range_hist)
#     # )
#     # bins_middle = edges[:-1] + ((edges[1] - edges[0]) / 2)
#     # plt.plot(
#     #     bins_middle,
#     #     count,
#     #     linestyle="-",
#     #     linewidth=0.5,
#     #     # marker="o",
#     #     # markersize=0.5,
#     #     label="threshold = {}".format(threshold),
#     # )
#     # plt.fill_between(bins_middle, count)

# plt.title(
#     "Distribution of anomaly areas' sizes for validation ResMaps with various Thresholds"
# )
# plt.legend()
# plt.xlabel("area size in pixel")
# plt.ylabel("count")
# plt.show(block=True)
# fig5.savefig(os.path.join(save_dir, "distr_area_th_multiple.pdf"))
