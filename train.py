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
import modules.models.resnet as resnet
import modules.models.mvtec_2 as mvtec_2
import modules.models.mvtec as mvtec
from modules import metrics as custom_metrics
from modules import loss_functions as loss_functions
import keras.backend as K
from tensorflow import keras
import tensorflow as tf
import sys
import os
import time
import ktrain

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


def main(args):

    # ========================= SETUP ==============================

    # Get training data setup
    directory = args.directory
    train_data_dir = os.path.join(directory, "train")
    nb_training_images_aug = args.nb_images
    batch_size = args.batch
    color_mode = args.color
    loss = args.loss.upper()
    validation_split = 0.1
    architecture = args.architecture
    tag = args.tag

    # check input arguments
    if architecture == "resnet" and color_mode == "grayscale":
        raise ValueError("ResNet expects rgb images")
    if architecture == "nasnet" and color_mode == "grayscale":
        raise ValueError("NasNet expects rgb images")
    if loss == "MSSIM" and color_mode == "grayscale":
        raise ValueError("MSSIM works only with rgb images")
    if loss == "SSIM" and color_mode == "rgb":
        raise ValueError("SSIM works only with grayscale images")

    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3

    # build model
    if architecture == "mvtec":
        model = mvtec.build_model(channels)
        dynamic_range = 1.0
    elif architecture == "mvtec2":
        model = mvtec_2.build_model(channels)
        dynamic_range = 1.0
    elif architecture == "resnet":
        model, base_encoder = resnet.build_model()
        dynamic_range = 2.0
    elif architecture == "nasnet":
        # model, base_encoder = models.build_nasnet()
        # dynamic_range = 2
        raise Exception("Nasnet ist not yet implemented.")

    # set metrics to monitor training
    if color_mode == "grayscale":
        metrics = [custom_metrics.ssim_metric(dynamic_range)]
        hist_keys = ("loss", "val_loss", "ssim", "val_ssim")
    elif color_mode == "rgb":
        metrics = [custom_metrics.mssim_metric(dynamic_range)]
        hist_keys = ("loss", "val_loss", "mssim", "val_mssim")

    # set loss function
    if loss == "SSIM":
        loss_function = loss_functions.ssim_loss(dynamic_range)
    elif loss == "MSSIM":
        loss_function = loss_functions.mssim_loss(dynamic_range)
    elif loss == "L2":
        loss_function = loss_functions.l2_loss
    elif loss == "MSE":
        loss_function = keras.losses.mean_squared_error

    # specify model name and directory to save model
    now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_dir = os.path.join(
        os.getcwd(), "saved_models", directory, architecture, loss, now
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_name = "CAE_" + architecture + "_b{}".format(batch_size)
    model_path = os.path.join(save_dir, model_name + ".h5")

    # specify logging directory for tensorboard visualization
    log_dir = os.path.join(save_dir, "logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # set callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, mode="min", verbose=1,
    )
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=False,  # True
        save_weights_only=False,
        period=1,
    )
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=log_dir, write_graph=True, update_freq="epoch"
    )

    # ============================= PREPROCESSING ===============================

    if architecture in ["mvtec", "mvtec2"]:
        rescale = 1.0 / 255
        shape = (256, 256)
        preprocessing_function = None
        preprocessing = None
    elif architecture == "resnet":
        rescale = None
        shape = (299, 299)
        preprocessing_function = keras.applications.inception_resnet_v2.preprocess_input
        preprocessing = "keras.applications.inception_resnet_v2.preprocess_input"
    elif architecture == "nasnet":
        rescale = None
        shape = (224, 224)
        preprocessing_function = keras.applications.nasnet.preprocess_input
        preprocessing = "keras.applications.inception_resnet_v2.preprocess_input"

    print("[INFO] Using Keras's flow_from_directory method...")
    # This will do preprocessing and realtime data augmentation:
    train_datagen = ImageDataGenerator(
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=5,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.05,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.05,
        # set mode for filling points outside the input boundaries
        fill_mode="nearest",
        # value used for fill_mode = "constant"
        cval=0.0,
        # randomly change brightness (darker < 1 < brighter)
        brightness_range=[0.95, 1.05],
        # set rescaling factor (applied before any other transformation)
        rescale=rescale,
        # set function that will be applied on each input
        preprocessing_function=preprocessing_function,
        # image data format, either "channels_first" or "channels_last"
        data_format="channels_last",
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=validation_split,
    )

    # For validation dataset, only rescaling
    validation_datagen = ImageDataGenerator(
        rescale=rescale,
        data_format="channels_last",
        validation_split=validation_split,
        preprocessing_function=preprocessing_function,
    )

    # Generate training batches with datagen.flow_from_directory()
    train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=shape,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="input",
        subset="training",
        shuffle=True,
    )

    # Generate validation batches with datagen.flow_from_directory()
    validation_generator = validation_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=shape,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="input",
        subset="validation",
        shuffle=True,
    )

    # Print command to paste in browser for visualizing in Tensorboard
    print("\ntensorboard --logdir={}\n".format(log_dir))

    # calculate epochs
    epochs = nb_training_images_aug // train_generator.samples

    # =============================== TRAINING =================================

    # define configuration for LR_find
    print("[INFO] initializing LR_find configuration...")
    if loss in ["SSIM", "MSSIM"]:
        stop_factor = -6
    elif loss == "L2":
        stop_factor = 6
    max_epochs = 10
    start_lr = 1e-7
    plt.close("all")

    if architecture in ["mvtec", "mvtec2"]:

        # initialize the optimizer and compile model
        print("[INFO] compiling model...")
        optimizer = keras.optimizers.Adam(learning_rate=start_lr)
        model.compile(
            loss=loss_function, optimizer=optimizer, metrics=metrics,
        )

        # wrap model and data in ktrain.Learner object
        learner = ktrain.get_learner(
            model=model,
            train_data=train_generator,
            val_data=validation_generator,
            # workers=8,
            use_multiprocessing=False,
            batch_size=batch_size,
        )

        # find good learning rate
        learner.lr_find(
            start_lr=start_lr,
            lr_mult=1.01,
            max_epochs=max_epochs,
            stop_factor=stop_factor,
            show_plot=False,
            # callbacks=[early_stopping_cb]
            verbose=1,
        )
        plt.figure()  # added
        learner.lr_plot()
        plt.savefig(os.path.join(save_dir, "lr_find_plot.png"))
        plt.show(block=True)
        plt.close()  # added
        print("[INFO] learning rate finder complete")

        # prompt user to enter max learning rate
        print("[INFO] examine plot and choose max learning rates...")
        max_lr = float(input("Enter max learning rate: "))

        # start training
        history = learner.fit_onecycle(
            lr=max_lr,
            epochs=epochs,
            cycle_momentum=True,
            max_momentum=0.95,
            min_momentum=0.85,
            verbose=1,
        )

        # setup and model configuration
        setup = {
            "data_setup": {
                "directory": directory,
                "nb_training_images": train_generator.samples,
                "nb_validation_images": validation_generator.samples,
            },
            "preprocessing_setup": {
                "rescale": rescale,
                "shape": shape,
                "preprocessing": preprocessing,
            },
            "lr_finder": {
                "start_lr": start_lr,
                "max_lr": max_lr,
                "stop_factor": stop_factor,
                "max_epochs": max_epochs,
            },
            "train_setup": {
                "architecture": architecture,
                "nb_training_images_aug": nb_training_images_aug,
                "epochs": epochs,
                "max_lr": max_lr,
                "min_lr": max_lr / 10,
                "batch_size": batch_size,
                "loss": loss,
                "dynamic_range": dynamic_range,
                "color_mode": color_mode,
                "channels": channels,
                "validation_split": validation_split,
            },
            "tag": tag,
        }

    elif architecture in ["resnet", "nasnet"]:

        # ------------------- PHASE 1 ---------------------

        print("[INFO] PHASE 1: training base encoder...")
        # freeze base
        print("[INFO] freezing base encoder's layers...")
        for layer in base_encoder.layers:
            layer.trainable = False
        epochs_1 = int(np.ceil(0.7 * epochs))

        # initialize the optimizer and compile model
        print("[INFO] compiling model...")
        optimizer = keras.optimizers.Adam(learning_rate=start_lr)
        model.compile(
            loss=loss_function, optimizer=optimizer, metrics=metrics,
        )

        # wrap model and data in ktrain.Learner object
        learner = ktrain.get_learner(
            model=model,
            train_data=train_generator,
            val_data=validation_generator,
            # workers=8,
            use_multiprocessing=False,
            batch_size=batch_size,
        )

        # find good learning rate
        learner.lr_find(
            start_lr=start_lr,
            lr_mult=1.01,
            max_epochs=max_epochs,
            stop_factor=stop_factor,
            show_plot=True,
            verbose=1,
        )
        # plt.close("all")
        # learner.lr_plot()
        # plt.xscale("log")
        plt.savefig(os.path.join(save_dir, "lr_find_plot_1.png"))
        plt.close("all")
        print("[INFO] learning rate finder for PHASE 1 complete")

        # prompt user to enter max learning rate
        print("[INFO] examine plot and choose max learning rates...")
        max_lr_1 = float(input("Enter max learning rate: "))

        # Phase 1: start training
        history_1 = learner.fit_onecycle(
            lr=max_lr_1,
            epochs=epochs_1,
            cycle_momentum=True,
            max_momentum=0.95,
            min_momentum=0.85,
            # callbacks=[early_stopping_cb],
            verbose=1,
        )

        # ------------------- PHASE 2 ---------------------

        print("[INFO] PHASE 2: training entire model...")
        print("[INFO] unfreezing base encoder's layers...")
        for layer in base_encoder.layers:
            layer.trainable = True
        epochs_2 = epochs - epochs_1

        # initialize the optimizer and compile model for Phase 2
        print("[INFO] compiling model...")
        optimizer = keras.optimizers.Adam(learning_rate=start_lr)
        model.compile(
            loss=loss_function, optimizer=optimizer, metrics=metrics,
        )

        # find good learning rate
        learner.lr_find(
            start_lr=start_lr,
            lr_mult=1.01,
            max_epochs=max_epochs,
            stop_factor=stop_factor,
            show_plot=False,
            verbose=1,
        )
        # plt.close("all")
        learner.lr_plot()
        # plt.xscale("log")
        plt.savefig(os.path.join(save_dir, "lr_find_plot_2.png"))
        plt.close("all")
        print("[INFO] learning rate finder for PHASE 2 complete")

        # prompt user to enter max learning rate
        print("[INFO] examine plot and choose max learning rates...")
        max_lr_2 = float(input("Enter max learning rate: "))

        # Phase 2: start training
        history_2 = learner.fit_onecycle(
            lr=max_lr_2,
            epochs=epochs_2,
            cycle_momentum=True,
            max_momentum=0.95,
            min_momentum=0.85,
            # callbacks=[early_stopping_cb],
            verbose=1,
        )

        # update history
        history = utils.update_history(history_1, history_2)
        learner.history = history

        # setup and model configuration
        setup = {
            "data_setup": {
                "directory": directory,
                "nb_training_images": train_generator.samples,
                "nb_validation_images": validation_generator.samples,
            },
            "preprocessing_setup": {
                "rescale": rescale,
                "shape": shape,
                "preprocessing": preprocessing,
            },
            "lr_finder": {
                "start_lr": start_lr,
                "max_lr_1": max_lr_1,
                "max_lr_2": max_lr_2,
                "stop_factor": stop_factor,
                "max_epochs": max_epochs,
            },
            "train_setup": {
                "architecture": architecture,
                "nb_training_images_aug": nb_training_images_aug,
                "epochs": epochs,
                "max_lr_1": max_lr_1,
                "min_lr_1": max_lr_1 / 10,
                "max_lr_2": max_lr_2,
                "min_lr_2": max_lr_2 / 10,
                "batch_size": batch_size,
                "loss": loss,
                "dynamic_range": dynamic_range,
                "color_mode": color_mode,
                "channels": channels,
                "validation_split": validation_split,
            },
            "tag": tag,
        }

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

        # create a generator that yields preprocessed validation images
        inspection_val_generator = validation_datagen.flow_from_directory(
            directory=train_data_dir,
            target_size=shape,
            color_mode=color_mode,
            batch_size=validation_generator.samples,
            class_mode=None,
            subset="validation",
            shuffle=False,
        )
        # imgs_val_input = inspection_val_generator.next()[0]
        imgs_val_input = next(inspection_val_generator)
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
                imgs_val_input[i, :, :, 0], cmap="gray", vmin=0.0, vmax=1.0
            )
            axarr[0].set_title("input")
            axarr[0].set_axis_off()
            f.colorbar(im00, ax=axarr[0])

            im10 = axarr[1].imshow(
                imgs_val_pred[i, :, :, 0], cmap="gray", vmin=0.0, vmax=1.0
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

        test_datagen = ImageDataGenerator(
            rescale=rescale,
            data_format="channels_last",
            preprocessing_function=preprocessing_function,
        )
        test_data_dir = os.path.join(directory, "test")
        total_number = utils.get_total_number_test_images(test_data_dir)

        # retrieve preprocessed test images as a numpy array
        inspection_test_generator = test_datagen.flow_from_directory(
            directory=test_data_dir,
            target_size=shape,
            color_mode=color_mode,
            batch_size=total_number,
            shuffle=False,
            class_mode=None,
        )
        # imgs_test_input = inspection_test_generator.next()[0]
        imgs_test_input = next(inspection_test_generator)
        filenames = inspection_test_generator.filenames

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
                imgs_test_input[i, :, :, 0], cmap="gray", vmin=0.0, vmax=1.0
            )
            axarr[0].set_title("input")
            axarr[0].set_axis_off()
            f.colorbar(im00, ax=axarr[0])

            im10 = axarr[1].imshow(
                imgs_test_pred[i, :, :, 0], cmap="gray", vmin=0.0, vmax=1.0
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

# python3 train.py -d mvtec/capsule -a resnet -b 8 -l l2 -c rgb -n 1100
# python3 train.py -d mvtec/capsule -a resnet -b 8 -l l2 -c rgb --inspect

# python3 train.py -d werkstueck/data_a30_nikon_schwarz_ooc_cut -a mvtec2 -b 4 -l ssim -c grayscale --inspect


# Examples of commands to initiate training with mvtec architecture

# python3 train.py -d mvtec/capsule -a mvtec2 -b 8 -l ssim -c grayscale --inspect
# python3 train.py -d werkstueck/data_a30_nikon_weiss_edit -a mvtec2 -b 12 -l l2 -c grayscale --inspect
