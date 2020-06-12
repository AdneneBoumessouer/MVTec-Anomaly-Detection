import sys
import os
import argparse
import json
import pandas as pd
import csv
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from modules import utils as utils
from modules.utils import printProgressBar as printProgressBar
from modules.resmaps import calculate_resmaps
from skimage.util import img_as_ubyte
from modules import metrics as custom_metrics
from modules import loss_functions as loss_functions
import keras.backend as K
from tensorflow import keras

import time
from autoencoder import AutoEncoder
import ktrain
import models
from preprocessing import Preprocessor

"""
Created on Tue Dec 10 19:46:17 2019

@author: adnene33


Valid input arguments for color_mode and loss:

                        +----------------+----------------+
                        |       Model Architecture        |
                        +----------------+----------------+
                        | mvtec, mvtec2  | Resnet, Nasnet |
========================+================+================+
        ||              |                |                |
        ||   grayscale  | SSIM, L2, MSE  |   Not Valid    |
Color   ||              |                |                |
Mode    ----------------+----------------+----------------+
        ||              |                |                |
        ||      RGB     | MSSIM, L2, MSE | MSSIM, L2, MSE |
        ||              |                |                |
--------+---------------+----------------+----------------+
"""


def check_arguments(architecture, color_mode, loss):
    if architecture == "resnet" and color_mode == "grayscale":
        raise ValueError("ResNet expects rgb images")
    if architecture == "nasnet" and color_mode == "grayscale":
        raise ValueError("NasNet expects rgb images")
    if loss == "MSSIM" and color_mode == "grayscale":
        raise ValueError("MSSIM works only with rgb images")
    if loss == "SSIM" and color_mode == "rgb":
        raise ValueError("SSIM works only with grayscale images")
    return


def main(args):

    # get parsed arguments from user
    input_directory = args.directory
    train_data_dir = os.path.join(input_directory, "train")
    batch_size = args.batch
    color_mode = args.color
    # loss = args.loss.upper()
    loss = args.loss
    architecture = args.architecture
    tag = args.tag

    # check arguments
    check_arguments(architecture, color_mode, loss)

    # get autoencoder
    autoencoder = AutoEncoder(
        input_directory, architecture, color_mode, loss, batch_size
    )

    # load data as generators that yield batches of preprocessed images
    preprocessor = Preprocessor(
        input_directory,
        rescale=autoencoder.rescale,
        shape=autoencoder.shape,
        color_mode=autoencoder.color_mode,
        preprocessing_function=autoencoder.preprocessing_function,
    )
    train_generator = preprocessor.get_train_generator(
        batch_size=autoencoder.batch_size, shuffle=True
    )
    validation_generator = preprocessor.get_val_generator(
        batch_size=autoencoder.batch_size, shuffle=True
    )

    nb_train_images = validation_generator.samples
    nb_validation_images = validation_generator.samples

    # find best learning rates for training
    autoencoder.find_opt_lr(train_generator, validation_generator)

    # train
    autoencoder.fit()

    #

    # save model
    autoencoder.save()

    # setup and model configuration

    # save setup
    with open(os.path.join(save_dir, "setup.json"), "w") as json_file:
        json.dump(setup, json_file, indent=4, sort_keys=False)

    # Save model
    tf.keras.models.save_model(
        model, model_path, include_optimizer=True, save_format="h5"
    )
    print("Saved trained model at %s " % model_path)

    # save training history
    hist_dict = dict((key, history.history[key]) for key in hist_keys)
    hist_df = pd.DataFrame(hist_dict)
    hist_csv_file = os.path.join(save_dir, "history.csv")
    with open(hist_csv_file, mode="w") as f:
        hist_df.to_csv(f)
    print("Saved training history at %s " % hist_csv_file)

    # save loss plot
    plt.figure()
    learner.plot(plot_type="loss")
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    print("loss plot saved at {} ".format(save_dir))

    # save lr plot
    plt.figure()
    learner.plot(plot_type="lr")
    plt.savefig(os.path.join(save_dir, "lr_plot.png"))
    print("learning rate plot saved at {} ".format(save_dir))

    if args.inspect:
        # -------------- INSPECTING VALIDATION IMAGES --------------
        print("[INFO] inspecting validation images...")

        # create a directory to save inspection plots
        inspection_val_dir = os.path.join(save_dir, "inspection_val")
        if not os.path.isdir(inspection_val_dir):
            os.makedirs(inspection_val_dir)
        inspection_val_generator = preprocessor.get_val_generator(
            batch_size=nb_validation_images, shuffle=False
        )
        imgs_val_input = inspection_val_generator.next()[0]
        # imgs_val_input = next(inspection_val_generator)
        filenames = inspection_val_generator.filenames

        # predict on validation images
        print("[INFO] reconstructing validation images...")
        imgs_val_pred = model.predict(imgs_val_input)

        # convert to grayscale if RGB
        if color_mode == "rgb":
            imgs_val_input = tf.image.rgb_to_grayscale(imgs_val_input).numpy()
            imgs_val_pred = tf.image.rgb_to_grayscale(imgs_val_pred).numpy()

        # save input and pred arrays
        print(
            "[INFO] saving input and pred validation images at {}...".format(
                inspection_val_dir
            )
        )
        utils.save_np(imgs_val_input, inspection_val_dir, "imgs_val_input.npy")
        utils.save_np(imgs_val_pred, inspection_val_dir, "imgs_val_pred.npy")

        # compute resmaps using the ssim method
        resmaps_val_ssim = calculate_resmaps(
            imgs_val_input, imgs_val_pred, method="SSIM"
        )

        # Convert to 8-bit unsigned int
        resmaps_val_ssim = img_as_ubyte(resmaps_val_ssim)

        # generate and save inspection images
        print("[INFO] generating inspection plots on validation images...")
        l = len(filenames)
        printProgressBar(0, l, prefix="Progress:", suffix="Complete", length=50)
        for i in range(len(imgs_val_input)):
            f, axarr = plt.subplots(3, 1)
            f.set_size_inches((4, 9))

            im00 = axarr[0].imshow(
                imgs_val_input[i, :, :, 0], cmap="gray", vmin=vmin, vmax=vmax
            )
            axarr[0].set_title("input")
            axarr[0].set_axis_off()
            f.colorbar(im00, ax=axarr[0])

            im10 = axarr[1].imshow(
                imgs_val_pred[i, :, :, 0], cmap="gray", vmin=vmin, vmax=vmax
            )
            axarr[1].set_title("pred")
            axarr[1].set_axis_off()
            f.colorbar(im10, ax=axarr[1])

            im20 = axarr[2].imshow(
                resmaps_val_ssim[i, :, :, 0], cmap="inferno", vmin=0, vmax=255
            )
            axarr[2].set_title("resmap_ssim")
            axarr[2].set_axis_off()
            f.colorbar(im20, ax=axarr[2])

            plt.suptitle("VALIDATION\n" + filenames[i])
            plot_name = utils.get_plot_name(filenames[i], suffix="inspection")
            f.savefig(os.path.join(inspection_val_dir, plot_name))
            plt.close(fig=f)
            # print progress bar
            time.sleep(0.1)
            printProgressBar(i + 1, l, prefix="Progress:", suffix="Complete", length=50)

        # -------------- INSPECTING TEST IMAGES --------------

        print("[INFO] inspecting test images...")

        # create a directory to save inspection plots
        inspection_test_dir = os.path.join(save_dir, "inspection_test")
        if not os.path.isdir(inspection_test_dir):
            os.makedirs(inspection_test_dir)

        nb_test_images = utils.get_total_number_test_images(test_data_dir)

        inspection_test_generator = preprocessor.get_test_generator(
            batch_size=nb_test_images, shuffle=False
        )
        imgs_test_input = inspection_val_generator.next()[0]
        filenames = inspection_val_generator.filenames

        # predict on test images
        print("[INFO] reconstructing test images...")
        imgs_test_pred = model.predict(imgs_test_input)

        # convert to grayscale if RGB
        if color_mode == "rgb":
            imgs_test_input = tf.image.rgb_to_grayscale(imgs_test_input).numpy()
            imgs_test_pred = tf.image.rgb_to_grayscale(imgs_test_pred).numpy()

        # save input and pred arrays
        print(
            "[INFO] saving input and pred test images at {}...".format(
                inspection_test_dir
            )
        )
        utils.save_np(imgs_test_input, inspection_test_dir, "imgs_test_input.npy")
        utils.save_np(imgs_test_pred, inspection_test_dir, "imgs_test_pred.npy")

        # compute resmaps using the ssim method
        resmaps_test_ssim = calculate_resmaps(
            imgs_test_input, imgs_test_pred, method="SSIM"
        )

        # Convert to 8-bit unsigned int
        resmaps_test_ssim = img_as_ubyte(resmaps_test_ssim)

        # generate and save inspection images
        print("[INFO] generating inspection plots on test images...")
        l = len(filenames)
        printProgressBar(0, l, prefix="Progress:", suffix="Complete", length=50)
        for i in range(len(imgs_test_input)):
            f, axarr = plt.subplots(3, 1)
            f.set_size_inches((4, 9))

            im00 = axarr[0].imshow(
                imgs_test_input[i, :, :, 0], cmap="gray", vmin=vmin, vmax=vmax
            )
            axarr[0].set_title("input")
            axarr[0].set_axis_off()
            f.colorbar(im00, ax=axarr[0])

            im10 = axarr[1].imshow(
                imgs_test_pred[i, :, :, 0], cmap="gray", vmin=vmin, vmax=vmax
            )
            axarr[1].set_title("pred")
            axarr[1].set_axis_off()
            f.colorbar(im10, ax=axarr[1])

            im20 = axarr[2].imshow(
                resmaps_test_ssim[i, :, :, 0], cmap="inferno", vmin=0, vmax=255
            )
            axarr[2].set_title("resmap_ssim")
            axarr[2].set_axis_off()
            f.colorbar(im20, ax=axarr[2])

            plt.suptitle("TEST\n" + filenames[i])
            plot_name = utils.get_plot_name(filenames[i], suffix="inspection")
            f.savefig(os.path.join(inspection_test_dir, plot_name))
            plt.close(fig=f)
            # print progress bar
            time.sleep(0.1)
            printProgressBar(i + 1, l, prefix="Progress:", suffix="Complete", length=50)

        print("[INFO] done.")
        print("[INFO] all generated files are saved at: \n{}".format(save_dir))
        print("exiting script...")


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        metavar="",
        help="training directory",
    )

    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        required=True,
        metavar="",
        choices=["mvtec", "mvtec2", "resnet", "nasnet"],
        help="model to use in training",
    )

    parser.add_argument(
        "-n",
        "--nb-images",
        type=int,
        default=10000,
        metavar="",
        help="number of training images",
    )
    parser.add_argument(
        "-b", "--batch", type=int, required=True, metavar="", help="batch size"
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        required=True,
        metavar="",
        choices=["mssim", "ssim", "l2", "mse"],
        help="loss function used during training",
    )

    parser.add_argument(
        "-c",
        "--color",
        type=str,
        required=True,
        metavar="",
        choices=["rgb", "grayscale"],
        help="color mode",
    )

    parser.add_argument(
        "-i",
        "--inspect",
        action="store_true",
        help="whether or not to reconstruct validation and test images after training",
    )

    parser.add_argument(
        "-t", "--tag", type=str, help="give a tag to the model to be trained"
    )

    args = parser.parse_args()
    if tf.test.is_gpu_available():
        print("[INFO] GPU was detected...")
    else:
        print("[INFO] No GPU was detected. CNNs can be very slow without a GPU...")
    print("[INFO] Tensorflow version: {} ...".format(tf.__version__))
    print("[INFO] Keras version: {} ...".format(keras.__version__))
    main(args)

# Examples of commands to initiate training with resnet architecture

# python3 train.py -d mvtec/capsule -a resnet -b 8 -l l2 -c rgb --inspect
# python3 train.py -d mvtec/capsule -a mvtec2 -b 8 -l l2 -c grayscale --inspect

# python3 train.py -d werkstueck/data_a30_nikon_schwarz_ooc_cut -a mvtec2 -b 4 -l ssim -c grayscale --inspect


# Examples of commands to initiate training with mvtec architecture

# python3 train.py -d mvtec/capsule -a mvtec2 -b 8 -l ssim -c grayscale --inspect
# python3 train.py -d werkstueck/data_a30_nikon_weiss_edit -a mvtec2 -b 12 -l l2 -c grayscale --inspect
