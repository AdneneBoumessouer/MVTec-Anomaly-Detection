import os
import argparse
import tensorflow as tf
from tensorflow import keras
from autoencoder.autoencoder import AutoEncoder
from processing.preprocessing import Preprocessor
from processing.utils import printProgressBar as printProgressBar
from processing import utils
from processing import resmaps


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
        ||   grayscale  |    SSIM, L2    |   Not Valid    |
Color   ||              |                |                |
Mode    ----------------+----------------+----------------+
        ||              |                |                |
        ||      RGB     |    MSSIM, L2   |    MSSIM, L2   |
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
    input_directory = args.input_directory
    train_data_dir = os.path.join(input_directory, "train")
    batch_size = args.batch
    color_mode = args.color
    # loss = args.loss.upper()
    loss = args.loss
    architecture = args.architecture

    # check arguments
    check_arguments(architecture, color_mode, loss)

    # get autoencoder
    autoencoder = AutoEncoder(
        input_directory, architecture, color_mode, loss, batch_size
    )

    # load data as generators that yield batches of preprocessed images
    preprocessor = Preprocessor(
        input_directory=input_directory,
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

    # find best learning rates for training
    autoencoder.find_opt_lr(train_generator, validation_generator)

    # train
    autoencoder.fit()

    # save model
    autoencoder.save()

    if args.inspect:
        # -------------- INSPECTING VALIDATION IMAGES --------------
        print("[INFO] inspecting validation images...")

        # create a directory to save inspection plots
        inspection_val_dir = os.path.join(autoencoder.save_dir, "inspection_val")
        if not os.path.isdir(inspection_val_dir):
            os.makedirs(inspection_val_dir)

        inspection_val_generator = preprocessor.get_val_generator(
            batch_size=autoencoder.learner.val_data.samples, shuffle=False
        )

        imgs_val_input = inspection_val_generator.next()[0]
        filenames_val = inspection_val_generator.filenames

        # get reconstructed images (i.e predictions) on validation dataset
        print("[INFO] reconstructing validation images...")
        imgs_val_pred = autoencoder.model.predict(imgs_val_input)

        # convert to grayscale if RGB
        if color_mode == "rgb":
            imgs_val_input = tf.image.rgb_to_grayscale(imgs_val_input).numpy()
            imgs_val_pred = tf.image.rgb_to_grayscale(imgs_val_pred).numpy()

        # remove last channel since images are grayscale
        imgs_val_input = imgs_val_input[:, :, :, 0]
        imgs_val_pred = imgs_val_pred[:, :, :, 0]

        # instantiate TensorImages object to compute validation resmaps
        tensor_val = resmaps.TensorImages(
            imgs_input=imgs_val_input,
            imgs_pred=imgs_val_pred,
            vmin=autoencoder.vmin,
            vmax=autoencoder.vmax,
            method="SSIM",
            dtype="float64",
            filenames=filenames_val,
        )

        # generate and save inspection validation plots
        tensor_val.generate_inspection_plots(
            group="validation", save_dir=inspection_val_dir
        )

        # -------------- INSPECTING TEST IMAGES --------------
        print("[INFO] inspecting test images...")

        # create a directory to save inspection plots
        inspection_test_dir = os.path.join(autoencoder.save_dir, "inspection_test")
        if not os.path.isdir(inspection_test_dir):
            os.makedirs(inspection_test_dir)

        nb_test_images = preprocessor.get_total_number_test_images()

        inspection_test_generator = preprocessor.get_test_generator(
            batch_size=nb_test_images, shuffle=False
        )

        imgs_test_input = inspection_test_generator.next()[0]
        filenames_test = inspection_test_generator.filenames

        # get reconstructed images (i.e predictions) on validation dataset
        print("[INFO] reconstructing test images...")
        imgs_test_pred = autoencoder.model.predict(imgs_test_input)

        # convert to grayscale if RGB
        if color_mode == "rgb":
            imgs_test_input = tf.image.rgb_to_grayscale(imgs_test_input).numpy()
            imgs_test_pred = tf.image.rgb_to_grayscale(imgs_test_pred).numpy()

        # remove last channel since images are grayscale
        imgs_test_input = imgs_test_input[:, :, :, 0]
        imgs_test_pred = imgs_test_pred[:, :, :, 0]

        # instantiate TensorImages object to compute test resmaps
        tensor_test = resmaps.TensorImages(
            imgs_input=imgs_test_input,
            imgs_pred=imgs_test_pred,
            vmin=autoencoder.vmin,
            vmax=autoencoder.vmax,
            method="SSIM",
            dtype="float64",
            filenames=filenames_test,
        )

        # generate and save inspection test plots
        tensor_test.generate_inspection_plots(
            group="test", save_dir=inspection_test_dir
        )

    print("exiting script...")
    return


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--input-directory",
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

    args = parser.parse_args()
    if tf.test.is_gpu_available():
        print("[INFO] GPU was detected...")
    else:
        print("[INFO] No GPU was detected. CNNs can be very slow without a GPU...")
    print("[INFO] Tensorflow version: {} ...".format(tf.__version__))
    print("[INFO] Keras version: {} ...".format(keras.__version__))
    main(args)

# Examples of commands to initiate training with mvtec architecture

# python3 train.py -d mvtec/capsule -a mvtec2 -b 8 -l ssim -c grayscale --inspect

# python3 train.py -d werkstueck/data_a30_nikon_weiss_edit -a mvtec2 -b 8 -l l2 -c grayscale --inspect
