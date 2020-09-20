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

from autoencoder.models import mvtecCAE
from autoencoder.models import baselineCAE
from autoencoder.models import inceptionCAE
from autoencoder.models import resnetCAE
from autoencoder.models import skipCAE
from autoencoder import metrics
from autoencoder import losses

import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        # custom learning rate estimation attributes
        self.lr_opt = None
        self.lr_opt_i = None
        self.lr_base = None
        self.lr_base_i = None

        # ktrain learning rate estimation attributes
        self.lr_mg_i = None
        self.lr_mg = None
        self.lr_ml_10_i = None
        self.lr_ml_10 = None

        # training attributes
        self.learner = None

        # results attributes
        self.hist = None
        self.epochs_trained = None

        # build model and preprocessing variables
        if architecture == "mvtecCAE":
            # Preprocessing parameters
            self.model = mvtecCAE.build_model(color_mode)
            self.rescale = mvtecCAE.RESCALE
            self.shape = mvtecCAE.SHAPE
            self.preprocessing_function = mvtecCAE.PREPROCESSING_FUNCTION
            self.preprocessing = mvtecCAE.PREPROCESSING
            self.vmin = mvtecCAE.VMIN
            self.vmax = mvtecCAE.VMAX
            self.dynamic_range = mvtecCAE.DYNAMIC_RANGE

        elif architecture == "baselineCAE":
            # Preprocessing parameters
            self.model = baselineCAE.build_model(color_mode)
            self.rescale = baselineCAE.RESCALE
            self.shape = baselineCAE.SHAPE
            self.preprocessing_function = baselineCAE.PREPROCESSING_FUNCTION
            self.preprocessing = baselineCAE.PREPROCESSING
            self.vmin = baselineCAE.VMIN
            self.vmax = baselineCAE.VMAX
            self.dynamic_range = baselineCAE.DYNAMIC_RANGE

        elif architecture == "inceptionCAE":
            # Preprocessing parameters
            self.model = inceptionCAE.build_model(color_mode)
            self.rescale = inceptionCAE.RESCALE
            self.shape = inceptionCAE.SHAPE
            self.preprocessing_function = inceptionCAE.PREPROCESSING_FUNCTION
            self.preprocessing = inceptionCAE.PREPROCESSING
            self.vmin = inceptionCAE.VMIN
            self.vmax = inceptionCAE.VMAX
            self.dynamic_range = inceptionCAE.DYNAMIC_RANGE

        elif architecture == "resnetCAE":
            # Preprocessing parameters
            self.model = resnetCAE.build_model(color_mode)
            self.rescale = resnetCAE.RESCALE
            self.shape = resnetCAE.SHAPE
            self.preprocessing_function = resnetCAE.PREPROCESSING_FUNCTION
            self.preprocessing = resnetCAE.PREPROCESSING
            self.vmin = resnetCAE.VMIN
            self.vmax = resnetCAE.VMAX
            self.dynamic_range = resnetCAE.DYNAMIC_RANGE

        elif architecture == "skipCAE":
            # Preprocessing parameters
            self.model = skipCAE.build_model(color_mode)
            self.rescale = skipCAE.RESCALE
            self.shape = skipCAE.SHAPE
            self.preprocessing_function = skipCAE.PREPROCESSING_FUNCTION
            self.preprocessing = skipCAE.PREPROCESSING
            self.vmin = skipCAE.VMIN
            self.vmax = skipCAE.VMAX
            self.dynamic_range = skipCAE.DYNAMIC_RANGE

        # Learning Rate Finder parameters
        self.start_lr = config.START_LR
        self.lr_max_epochs = config.LR_MAX_EPOCHS
        self.lrf_decrease_factor = config.LRF_DECREASE_FACTOR

        # Training parameters
        self.early_stopping = config.EARLY_STOPPING
        self.reduce_on_plateau = config.REDUCE_ON_PLATEAU

        # verbosity
        self.verbose = verbose
        if verbose:
            self.model.summary()

        # set loss function
        if loss == "ssim":
            self.loss_function = losses.ssim_loss(self.dynamic_range)
        elif loss == "mssim":
            self.loss_function = losses.mssim_loss(self.dynamic_range)
        elif loss == "l2":
            self.loss_function = losses.l2_loss

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
        self.model.compile(
            loss=self.loss_function, optimizer="adam", metrics=self.metrics
        )
        return

    ### Methods for training =================================================

    def find_lr_opt(self, train_generator, validation_generator):
        # initialize learner object
        self.learner = ktrain.get_learner(
            model=self.model,
            train_data=train_generator,
            val_data=validation_generator,
            batch_size=self.batch_size,
        )

        # simulate training while recording learning rate and loss
        logger.info("initiating learning rate finder to determine best learning rate.")

        self.learner.lr_find(
            start_lr=self.start_lr,
            lr_mult=1.01,
            max_epochs=self.lr_max_epochs,
            stop_factor=6,
            verbose=self.verbose,
            show_plot=True,
            restore_weights_only=True,
        )
        self.ktrain_lr_estimate()
        self.custom_lr_estimate()
        self.lr_find_plot(n_skip_beginning=10, n_skip_end=1, save=True)
        return

    def ktrain_lr_estimate(self):
        # get losses and learning rates
        losses = np.array(self.learner.lr_finder.losses)
        lrs = np.array(self.learner.lr_finder.lrs)
        # minimum loss devided by 10
        self.ml_i = self.learner.lr_finder.ml
        self.lr_ml_10 = lrs[self.ml_i] / 10
        logger.info(f"lr with minimum loss divided by 10: {self.lr_ml_10:.2E}")
        try:
            min_loss_i = np.argmin(losses)
            self.lr_ml_10_i = np.argwhere(lrs[:min_loss_i] > self.lr_ml_10)[0][0]
        except:
            self.lr_ml_10_i = None
        # minimum gradient
        self.lr_mg_i = self.learner.lr_finder.mg
        if self.lr_mg_i is not None:
            self.lr_mg = lrs[self.lr_mg_i]
            logger.info(f"lr with minimum numerical gradient: {self.lr_mg:.2E}")
        return

    def custom_lr_estimate(self):
        # get losses and learning rates
        losses = np.array(self.learner.lr_finder.losses)
        lrs = np.array(self.learner.lr_finder.lrs)
        # find optimal learning rate
        min_loss = np.amin(losses)
        min_loss_i = np.argmin(losses)
        # retrieve segment containing decreasing losses
        segment = losses[: min_loss_i + 1]
        max_loss = np.amax(segment)
        # compute optimal loss
        optimal_loss = max_loss - self.lrf_decrease_factor * (max_loss - min_loss)
        # get optimal learning rate index (corresponding to optimal loss) and value
        self.lr_opt_i = np.argwhere(segment < optimal_loss)[0][0]
        self.lr_opt = float(lrs[self.lr_opt_i])
        # get base learning rate
        self.lr_base_i = np.argwhere(lrs[:min_loss_i] > self.lr_opt / 10)[0][0]
        self.lr_base = float(lrs[self.lr_base_i])
        # log to console
        logger.info(f"custom base learning rate: {lrs[self.lr_base_i]:.2E}")
        logger.info(f"custom optimal learning rate: {lrs[self.lr_opt_i]:.2E}")
        logger.info("learning rate finder complete.")
        return

    def fit(self, lr_opt):
        # create tensorboard callback to monitor training
        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir=self.log_dir, write_graph=True, update_freq="epoch"
        )
        # Print command to paste in browser for visualizing in Tensorboard
        logger.info(
            "run the following command in a seperate terminal to monitor training on tensorboard:"
            + "\ntensorboard --logdir={}\n".format(self.log_dir)
        )
        assert self.learner.model is self.model

        # fit model using Cyclical Learning Rates
        self.hist = self.learner.autofit(
            lr=lr_opt,
            epochs=None,
            early_stopping=self.early_stopping,
            reduce_on_plateau=self.reduce_on_plateau,
            reduce_factor=2,
            cycle_momentum=True,
            max_momentum=0.95,
            min_momentum=0.85,
            monitor="val_loss",
            checkpoint_folder=None,
            verbose=self.verbose,
            callbacks=[tensorboard_cb],
        )
        return

    ### Methods to create directory structure and save (and load?) model =================

    def create_save_dir(self):
        # create a directory to save model
        now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        save_dir = os.path.join(
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
        model_name = self.architecture + "_b{}_e{}.hdf5".format(
            self.batch_size, epochs_trained
        )
        return model_name

    def save(self):
        # save model
        self.model.save(os.path.join(self.save_dir, self.create_model_name()))
        # save trainnig info
        info = self.get_info()
        with open(os.path.join(self.save_dir, "info.json"), "w") as json_file:
            json.dump(info, json_file, indent=4, sort_keys=False)
        # save training plots
        self.loss_plot(save=True)
        self.lr_schedule_plot(save=True)
        # save training history
        hist_dict = self.get_history_dict()
        hist_df = pd.DataFrame(hist_dict)
        hist_csv_file = os.path.join(self.save_dir, "history.csv")
        with open(hist_csv_file, mode="w") as csv_file:
            hist_df.to_csv(csv_file)
        logger.info("training history has been successfully saved as csv file.")
        logger.info(
            "training files have been successfully saved at: \n{}".format(self.save_dir)
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
            "lr_finder": {"lr_base": self.lr_base, "lr_opt": self.lr_opt,},
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

    def lr_find_plot(self, n_skip_beginning=10, n_skip_end=1, save=False):
        # get losses and learning rates
        losses = np.array(self.learner.lr_finder.losses)
        lrs = np.array(self.learner.lr_finder.lrs)
        # get display indicies
        sb = n_skip_beginning
        se = n_skip_end
        # plot
        with plt.style.context("seaborn-darkgrid"):
            fig, ax = plt.subplots()
            plt.ylabel("loss")
            plt.xlabel("learning rate (log scale)")
            ax.plot(lrs[sb:-se], losses[sb:-se])
            plt.xscale("log")
            ax.plot(
                lrs[self.lr_base_i],
                losses[self.lr_base_i],
                markersize=10,
                marker="o",
                color="green",
                label="custom_lr_base",
            )
            ax.plot(
                lrs[self.lr_opt_i],
                losses[self.lr_opt_i],
                markersize=10,
                marker="o",
                color="red",
                label="custom_lr_opt",
            )
            if self.lr_ml_10_i is not None:
                ax.plot(
                    lrs[self.lr_ml_10_i],
                    losses[self.lr_ml_10_i],
                    markersize=7,
                    marker="s",
                    color="magenta",
                    label="lr_min_loss_div_10",
                )
            if self.lr_mg_i is not None:
                ax.plot(
                    lrs[self.lr_mg_i],
                    losses[self.lr_mg_i],
                    markersize=7,
                    marker="s",
                    color="blue",
                    label="lr_min_gradient",
                )
            plt.title(
                f"Learning Rate Plot \ncustom base learning rate: {lrs[self.lr_base_i]:.2E}\n"
                + f"custom optimal learning rate: {lrs[self.lr_opt_i]:.2E}"
            )
            ax.legend()
            plt.show()
        if save:
            plt.close()
            fig.savefig(os.path.join(self.save_dir, "lr_plot.png"))
            logger.info("lr_plot.png successfully saved.")
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
            logger.info("lr_schedule_plot.png successfully saved.")
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
            logger.info("loss_plot.png successfully saved.")
        return
