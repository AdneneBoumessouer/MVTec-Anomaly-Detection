# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:46:17 2019

@author: Adnene Boumessouer
"""
import sys
import os
import shutil
import datetime
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import ktrain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autoencoder.models import mvtec
from autoencoder.models import mvtec_2
from autoencoder.models import resnet
from autoencoder.models import nasnet
from autoencoder import metrics
from autoencoder import losses


# Learning Rate Finder Parameters
START_LR = 1e-5
LR_MAX_EPOCHS = 10


class AutoEncoder:
    def __init__(
        self,
        input_directory,
        architecture,
        color_mode,
        loss,
        batch_size=8,
        verbose=True,
    ):
        # path attrivutes
        self.input_directory = input_directory
        self.save_dir = None
        self.log_dir = None

        # model and data attributes
        self.architecture = architecture
        self.color_mode = color_mode
        self.loss = loss
        self.batch_size = batch_size

        # learning rate finder attributes
        self.opt_lr = None
        self.opt_lr_i = None
        self.base_lr = None
        self.base_lr_i = None

        # training attributes
        self.learner = None

        # results attributes
        self.hist = None
        self.epochs_trained = None

        # verbosity
        self.verbose = verbose

        # build model and preprocessing variables
        if architecture == "mvtec":
            self.model = mvtec.build_model(color_mode)
            self.rescale = mvtec.RESCALE
            self.shape = mvtec.SHAPE
            self.preprocessing_function = mvtec.PREPROCESSING_FUNCTION
            self.preprocessing = mvtec.PREPROCESSING
            self.vmin = mvtec.VMIN
            self.vmax = mvtec.VMAX
            self.dynamic_range = mvtec.DYNAMIC_RANGE
        elif architecture == "mvtec2":
            self.model = mvtec_2.build_model(color_mode)
            self.rescale = mvtec_2.RESCALE
            self.shape = mvtec_2.SHAPE
            self.preprocessing_function = mvtec_2.PREPROCESSING_FUNCTION
            self.preprocessing = mvtec_2.PREPROCESSING
            self.vmin = mvtec_2.VMIN
            self.vmax = mvtec_2.VMAX
            self.dynamic_range = mvtec_2.DYNAMIC_RANGE
        elif architecture == "resnet":
            self.model = resnet.build_model()
            self.rescale = resnet.RESCALE
            self.shape = resnet.SHAPE
            self.preprocessing_function = resnet.PREPROCESSING_FUNCTION
            self.preprocessing = resnet.PREPROCESSING
            self.vmin = resnet.VMIN
            self.vmax = resnet.VMAX
            self.dynamic_range = resnet.DYNAMIC_RANGE
        elif architecture == "nasnet":
            self.model = nasnet.build_model()
            self.rescale = nasnet.RESCALE
            self.shape = nasnet.SHAPE
            self.preprocessing_function = nasnet.PREPROCESSING_FUNCTION
            self.preprocessing = nasnet.PREPROCESSING
            self.vmin = nasnet.VMIN
            self.vmax = nasnet.VMAX
            self.dynamic_range = nasnet.DYNAMIC_RANGE
            raise NotImplementedError("nasnet not yet implemented.")

        # set loss function
        if loss == "ssim":
            self.loss_function = losses.ssim_loss(self.dynamic_range)
        elif loss == "mssim":
            self.loss_function = losses.mssim_loss(self.dynamic_range)
        elif loss == "l2":
            self.loss_function = losses.l2_loss
        elif loss == "mse":
            self.loss_function = keras.losses.mean_squared_error

        # set metrics to monitor training
        if color_mode == "grayscale":
            self.metrics = [metrics.ssim_metric(self.dynamic_range)]
            self.hist_keys = ("loss", "val_loss", "ssim", "val_ssim")
        elif color_mode == "rgb":
            self.metrics = [metrics.mssim_metric(self.dynamic_range)]
            self.hist_keys = ("loss", "val_loss", "mssim", "val_mssim")

        # create directory to save model and logs
        self.create_save_dir()

        # compile model
        optimizer = keras.optimizers.Adam(learning_rate=START_LR)
        self.model.compile(
            loss=self.loss_function, optimizer=optimizer, metrics=self.metrics
        )
        return

    ### Methods for training =================================================

    def find_opt_lr(self, train_generator, validation_generator):
        # initialize learner object
        self.learner = ktrain.get_learner(
            model=self.model,
            train_data=train_generator,
            val_data=validation_generator,
            batch_size=self.batch_size,
        )

        if self.loss in ["ssim", "mssim"]:
            stop_factor = -6
        elif self.loss == ["l2", "mse"]:
            stop_factor = 6

        # simulate training while recording learning rate and loss
        print("[INFO] initiating learning rate finder to determine best learning rate.")
        try:
            self.learner.lr_find(
                start_lr=START_LR,
                lr_mult=1.01,
                max_epochs=LR_MAX_EPOCHS,
                stop_factor=stop_factor,
                verbose=self.verbose,
                show_plot=True,  # False
            )
        except Exception:
            shutil.rmtree(self.save_dir)
            sys.exit("\nexiting script.")

        losses = np.array(self.learner.lr_finder.losses)
        lrs = np.array(self.learner.lr_finder.lrs)

        # find optimal learning rate
        min_loss = np.amin(losses)
        min_loss_i = np.argmin(losses)

        # retrieve segment containing decreasing losses
        segment = losses[: min_loss_i + 1]
        max_loss = np.amax(segment)

        # compute optimal loss
        optimal_loss = max_loss - 0.92 * (max_loss - min_loss)

        # get index corresponding to optimal loss
        self.opt_lr_i = np.argwhere(segment < optimal_loss)[0][0]

        # get optimal learning rate
        self.opt_lr = float(lrs[self.opt_lr_i])

        # get base learning rate
        self.base_lr = self.opt_lr / 10
        self.base_lr_i = np.argwhere(lrs[:min_loss_i] > self.base_lr)[0][0]
        print("[INFO] learning rate finder complete.")
        print(f"\tbase learning rate: {self.base_lr:.2E}")
        print(f"\toptimal learning rate: {self.opt_lr:.2E}")
        self.lr_find_plot(save=True)
        return

    def fit(self):
        # create tensorboard callback to monitor training
        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir=self.log_dir, write_graph=True, update_freq="epoch"
        )
        # Print command to paste in browser for visualizing in Tensorboard
        print(
            "[INFO] run the following command in a seperate terminal to monitor training on tensorboard:"
        )
        print("\ntensorboard --logdir={}\n".format(self.log_dir))

        # fit model using Cyclical Learning Rates
        try:
            self.hist = self.learner.autofit(
                self.opt_lr,
                epochs=None,
                early_stopping=6,
                reduce_on_plateau=3,
                reduce_factor=2,
                cycle_momentum=True,
                max_momentum=0.95,
                min_momentum=0.85,
                monitor="val_loss",  # Check this
                checkpoint_folder=None,
                verbose=self.verbose,
                callbacks=[tensorboard_cb],
            )
        except Exception:
            shutil.rmtree(self.save_dir)
            sys.exit("\nexiting script.")
        return

    ### Methods to create directory structure and save (and load?) model =================

    def create_save_dir(self):
        # create a directory to save model
        now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        # root_dir = str(Path(os.getcwd()).parent)
        save_dir = os.path.join(
            # root_dir,
            os.getcwd(),
            "saved_models",
            self.input_directory,
            self.architecture,
            self.loss,
            now,
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        # create a log directory for tensorboard
        log_dir = os.path.join(save_dir, "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        return

    def create_model_name(self):
        epochs_trained = self.get_best_epoch()
        model_name = (
            "CAE_"
            + self.architecture
            + "_b{}_e{}.hdf5".format(self.batch_size, epochs_trained)
        )
        # model_path = os.path.join(save_dir, model_name + ".h5")
        return model_name

    def save(self):
        # save model
        self.model.save(os.path.join(self.save_dir, self.create_model_name()))
        # tf.keras.models.save_model(
        #     model, model_path, include_optimizer=True, save_format="h5"
        # )
        # save trainnig info
        info = self.get_info()
        with open(os.path.join(self.save_dir, "info.json"), "w") as json_file:
            json.dump(info, json_file, indent=4, sort_keys=False)
        # save training plots
        self.loss_plot(save=True)
        # self.lr_find_plot(save=True)
        self.lr_schedule_plot(save=True)
        # save training history
        hist_dict = self.get_history_dict()
        hist_df = pd.DataFrame(hist_dict)
        hist_csv_file = os.path.join(self.save_dir, "history.csv")
        with open(hist_csv_file, mode="w") as csv_file:
            hist_df.to_csv(csv_file)
        # print("[INFO] training history has been successfully saved as csv file at %s " % hist_csv_file)
        print("[INFO] training history has been successfully saved as csv file.")
        print(
            "[INFO] all files have been successfully saved at: \n{}".format(
                self.save_dir
            )
        )
        return

    ### Methods for getting finished training process info =====================

    def get_history_dict(self):
        hist_dict = dict((key, self.hist.history[key]) for key in self.hist_keys)
        return hist_dict

    def get_info(self):
        info = {
            "data": {
                "input_directory": self.input_directory,
                "nb_training_images": self.learner.train_data.samples,
                "nb_validation_images": self.learner.val_data.samples,
                "validation_split": self.learner.train_data.image_data_generator._validation_split,
            },
            "model": {"architecture": self.architecture, "loss": self.loss,},
            "preprocessing": {
                "color_mode": self.color_mode,
                "rescale": self.rescale,
                "shape": self.shape,
                "vmin": self.vmin,
                "vmax": self.vmax,
                "dynamic_range": self.dynamic_range,
                "preprocessing": self.preprocessing,
            },
            "lr_finder": {"base_lr": self.base_lr, "opt_lr": self.opt_lr,},
            "training": {
                "batch_size": self.batch_size,
                "epochs_trained": self.get_best_epoch(),
                "nb_train_images_total": self.get_total_nb_training_images(),
            },
        }
        return info

    def get_best_epoch(self):
        """
        Returns the index of the epoch when the model had stopped training.
        This epoch corresponds to the lowest validation loss registered
        during training because of the use of Early Stopping Callback.
        """
        hist_dict = self.get_history_dict()
        best_epoch = int(np.argmin(np.array(hist_dict["val_loss"])))
        return best_epoch

    def get_best_val_loss(self):
        """
        Returns the (lowest) validation loss corresponding to the best epoch.
        """
        hist_dict = self.get_history_dict()
        epochs_trained = np.argmin(np.array(hist_dict["val_loss"]))
        best_val_loss = np.array(hist_dict["val_loss"])[epochs_trained]
        return best_val_loss

    def get_total_nb_training_images(self):
        epochs_trained = self.get_best_epoch()
        total_nb = int(epochs_trained * self.learner.train_data.samples)
        return total_nb

    ### Methods for plotting ============================================

    def lr_find_plot(self, save=False):
        losses = np.array(self.learner.lr_finder.losses)
        lrs = np.array(self.learner.lr_finder.lrs)
        i = self.opt_lr_i
        j = self.base_lr_i
        with plt.style.context("seaborn-darkgrid"):
            fig, ax = plt.subplots()
            plt.ylabel("loss")
            plt.xlabel("learning rate (log scale)")
            ax.plot(lrs[10:-1], losses[10:-1])
            plt.xscale("log")
            ax.plot(
                lrs[j],
                losses[j],
                markersize=10,
                marker="o",
                color="green",
                label="base_lr",
            )
            ax.plot(
                lrs[i],
                losses[i],
                markersize=10,
                marker="o",
                color="red",
                label="opt_lr",
            )
            plt.title(
                f"Learning Rate Plot \nbase learning rate: {lrs[j]:.2E}\noptimal learning rate: {lrs[i]:.2E}"
            )
            ax.legend()
            plt.show()
        if save:
            plt.close()
            fig.savefig(os.path.join(self.save_dir, "lr_plot.png"))
            print("[INFO] lr_plot.png successfully saved.")
        print(f"[INFO] base learning rate: {lrs[j]:.2E}")
        print(f"[INFO] optimal learning rate: {lrs[i]:.2E}")
        return

    def lr_schedule_plot(self, save=False):
        with plt.style.context("seaborn-darkgrid"):
            fig, _ = plt.subplots()
            self.learner.plot(plot_type="lr")
            plt.title("Cyclical Learning Rate Scheduler")
            plt.show()
        if save:
            plt.close()
            fig.savefig(os.path.join(self.save_dir, "lr_schedule_plot.png"))
            print("[INFO] lr_schedule_plot.png successfully saved.")
            # return fig
        return

    def loss_plot(self, save=False):
        hist_dict = self.get_history_dict()
        hist_df = pd.DataFrame(hist_dict)
        with plt.style.context("seaborn-darkgrid"):
            fig = hist_df.plot().get_figure()
            plt.title("Loss Plot")
            plt.show()
        if save:
            plt.close()
            fig.savefig(os.path.join(self.save_dir, "loss_plot.png"))
            print("[INFO] loss_plot.png successfully saved.")
        return

    ### Methods to load model (and data?) =================
