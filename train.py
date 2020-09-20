"""
Created on Tue Dec 10 19:46:17 2019

@author: Adnene Boumessouer
"""

import os
import argparse
import tensorflow as tf
from tensorflow import keras
from autoencoder.autoencoder import AutoEncoder
from processing.preprocessing import Preprocessor
from processing.utils import printProgressBar as printProgressBar
from processing import utils
from processing import postprocessing
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
Valid combinations for input arguments for architecture, color_mode and loss:
                        +----------------+----------------+
                        |       Model Architecture        |
                        +----------------+----------------+
                        |  mvtecCAE      |   ResnetCAE    |
                        |  baselineCAE   |                |
                        |  inceptionCAE  |                |
========================+================+================+
        ||              |                |                |
        ||   grayscale  |    ssim, l2    |    ssim, l2    |
Color   ||              |                |                |
Mode    ----------------+----------------+----------------+
        ||              |                |                |
        ||      RGB     |    mssim, l2   |    mssim, l2   |
        ||              |                |                |
--------+---------------+----------------+----------------+
"""


def check_arguments(architecture, color_mode, loss):
    if loss == "mssim" and color_mode == "grayscale":
        raise ValueError("MSSIM works only with rgb images")
    if loss == "ssim" and color_mode == "rgb":
        raise ValueError("SSIM works only with grayscale images")
    return


def main(args):

    # get parsed arguments from user
    input_dir = args.input_dir
    architecture = args.architecture
    color_mode = args.color
    loss = args.loss
    batch_size = args.batch

    # check arguments
    check_arguments(architecture, color_mode, loss)

    # get autoencoder
    autoencoder = AutoEncoder(input_dir, architecture, color_mode, loss, batch_size)

    # load data as generators that yield batches of preprocessed images
    preprocessor = Preprocessor(
        input_directory=input_dir,
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
    autoencoder.find_lr_opt(train_generator, validation_generator)

    # train
    autoencoder.fit(lr_opt=autoencoder.lr_opt)

    # save model
    autoencoder.save()

    if args.inspect:
        # -------------- INSPECTING VALIDATION IMAGES --------------
        logger.info("generating inspection plots of validation images...")

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
        logger.info("reconstructing validation images...")
        imgs_val_pred = autoencoder.model.predict(imgs_val_input)

        # instantiate TensorImages object to compute validation resmaps
        tensor_val = postprocessing.TensorImages(
            imgs_input=imgs_val_input,
            imgs_pred=imgs_val_pred,
            vmin=autoencoder.vmin,
            vmax=autoencoder.vmax,
            method=autoencoder.loss,
            dtype="float64",
            filenames=filenames_val,
        )

        # generate and save inspection validation plots
        tensor_val.generate_inspection_plots(
            group="validation", save_dir=inspection_val_dir
        )

        # -------------- INSPECTING TEST IMAGES --------------
        logger.info("generating inspection plots of test images...")

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
        logger.info("reconstructing test images...")
        imgs_test_pred = autoencoder.model.predict(imgs_test_input)

        # instantiate TensorImages object to compute test resmaps
        tensor_test = postprocessing.TensorImages(
            imgs_input=imgs_test_input,
            imgs_pred=imgs_test_pred,
            vmin=autoencoder.vmin,
            vmax=autoencoder.vmax,
            method=autoencoder.loss,
            dtype="float64",
            filenames=filenames_test,
        )

        # generate and save inspection test plots
        tensor_test.generate_inspection_plots(
            group="test", save_dir=inspection_test_dir
        )

    logger.info("done.")
    return


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="Train an AutoEncoder on an image dataset.",
        epilog="Example usage: python3 train.py -d mvtec/capsule -a mvtec2 -b 8 -l ssim -c grayscale",
    )
    parser.add_argument(
        "-d",
        "--input-dir",
        type=str,
        required=True,
        metavar="",
        help="directory containing training images",
    )

    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        required=False,
        metavar="",
        choices=["mvtecCAE", "baselineCAE", "inceptionCAE", "resnetCAE", "skipCAE"],
        default="mvtec2",
        help="architecture of the model to use for training: 'mvtecCAE', 'baselineCAE', 'inceptionCAE', 'resnetCAE' or 'skipCAE'",
    )

    parser.add_argument(
        "-c",
        "--color",
        type=str,
        required=False,
        metavar="",
        choices=["rgb", "grayscale"],
        default="grayscale",
        help="color mode for preprocessing images before training: 'rgb' or 'grayscale'",
    )

    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        required=False,
        metavar="",
        choices=["mssim", "ssim", "l2"],
        default="ssim",
        help="loss function to use for training: 'mssim', 'ssim' or 'l2'",
    )

    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        required=False,
        metavar="",
        default=8,
        help="batch size to use for training",
    )

    parser.add_argument(
        "-i",
        "--inspect",
        action="store_true",
        help="generate inspection plots after training",
    )

    args = parser.parse_args()
    if tf.test.is_gpu_available():
        logger.info("GPU was detected...")
    else:
        logger.info("No GPU was detected. CNNs can be very slow without a GPU...")
    logger.info("Tensorflow version: {} ...".format(tf.__version__))
    logger.info("Keras version: {} ...".format(keras.__version__))
    main(args)

# Examples of commands to initiate training with mvtec architecture

# python3 train.py -d mvtec/capsule -a mvtecCAE -b 8 -l ssim -c grayscale --inspect
# python3 train.py -d mvtec/hazelnut -a resnetCAE -b 8 -l mssim -c rgb --inspect
# python3 train.py -d mvtec/pill -a inceptionCAE -b 8 -l mssim -c rgb --inspect
