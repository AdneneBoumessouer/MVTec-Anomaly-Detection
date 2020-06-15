import os
import time

import numpy as np
from skimage.metrics import structural_similarity as ssim

from processing import utils
from processing.utils import printProgressBar as printProgressBar

import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte

# Segmentation Parameters

# float + SSIM
THRESH_MIN_FLOAT_SSIM = 0.35
THRESH_STEP_FLOAT_SSIM = 0.002

# float + L2
THRESH_MIN_FLOAT_L2 = 0.01
THRESH_STEP_FLOAT_L2 = 0.0005

# uint8 + SSIM
THRESH_MIN_UINT8_SSIM = 90
THRESH_STEP_UINT8_SSIM = 1

# uint8 + L2 are incompatible


class TensorImages:
    def __init__(
        self,
        imgs_input,
        imgs_pred,
        vmin,
        vmax,
        method,
        dtype="float64",
        filenames=None,
    ):
        assert imgs_input.ndim == 3
        assert imgs_pred.ndim == 3
        self.imgs_input = imgs_input
        self.imgs_pred = imgs_pred
        self.dtype = dtype

        # pixel min and max values depend on preprocessing function,
        # which in turn depends on the model used for training.
        self.vmin = vmin
        self.vmax = vmax

        # compute resmaps
        assert dtype in ["float64", "uint8"]
        assert method in ["L2", "SSIM"]
        self.resmaps = calculate_resmaps(self.imgs_input, self.imgs_pred, method, dtype)
        if dtype == "float64":
            if method == "SSIM":
                self.thresh_min = THRESH_MIN_FLOAT_SSIM
                self.step = THRESH_STEP_FLOAT_SSIM
                self.vmin_resmap = 0.0
                self.vmax_resmap = 1.0
            elif method == "L2":
                self.thresh_min = THRESH_MIN_FLOAT_L2
                self.step = THRESH_STEP_FLOAT_L2
                self.vmin_resmap = None
                self.vmax_resmap = None
        elif dtype == "uint8":
            if method == "SSIM":
                self.thresh_min = THRESH_MIN_UINT8_SSIM
                self.step = THRESH_STEP_UINT8_SSIM
                self.vmin_resmap = 0
                self.vmax_resmap = 255
            elif method == "L2":
                raise Exception("L2 Resmaps are incompatible with uint8 dtype.")
            # Convert to 8-bit unsigned int
            self.resmaps = img_as_ubyte(self.resmaps)
        self.thresh_max = np.amax(self.resmaps)
        # self.n_steps = (self.thresh_max - self.thresh_min) // self.step
        self.method = method
        self.filenames = filenames

    def generate_inspection_plots(self, group, save_dir=None):
        assert group in ["validation", "test"]
        l = len(self.filenames)
        printProgressBar(0, l, prefix="Progress:", suffix="Complete", length=50)
        for i in range(len(self.imgs_input)):
            self.plot_input_pred_resmap(i, group, save_dir)
            # print progress bar
            time.sleep(0.1)
            printProgressBar(i + 1, l, prefix="Progress:", suffix="Complete", length=50)
        return

    ### plottings methods for inspection

    def plot_input_pred_resmap(self, index, group, save_dir=None):
        assert group in ["validation", "test"]
        fig, axarr = plt.subplots(3, 1)
        fig.set_size_inches((4, 9))

        im00 = axarr[0].imshow(
            self.imgs_input[index], cmap="gray", vmin=self.vmin, vmax=self.vmax,
        )
        axarr[0].set_title("input")
        axarr[0].set_axis_off()
        fig.colorbar(im00, ax=axarr[0])

        im10 = axarr[1].imshow(
            self.imgs_pred[index], cmap="gray", vmin=self.vmin, vmax=self.vmax
        )
        axarr[1].set_title("pred")
        axarr[1].set_axis_off()
        fig.colorbar(im10, ax=axarr[1])

        im20 = axarr[2].imshow(
            self.resmaps[index],
            cmap="inferno",
            vmin=self.vmin_resmap,
            vmax=self.vmax_resmap,
        )
        axarr[2].set_title("resmap_" + self.method)
        axarr[2].set_axis_off()
        fig.colorbar(im20, ax=axarr[2])

        plt.suptitle(group.upper() + "\n" + self.filenames[index])

        if save_dir is not None:
            plot_name = get_plot_name(self.filenames[index], suffix="inspection")
            fig.savefig(os.path.join(save_dir, plot_name))
            plt.close(fig=fig)
        # return fig

    def plot_image(self, plot_type, index):
        assert plot_type in ["input", "pred", "resmap"]
        # select image to plot
        if plot_type == "input":
            image = self.imgs_input[index]
            cmap = "gray"
            vmin = self.vmin
            vmax = self.vmax
        elif plot_type == "pred":
            image = self.imgs_pred[index]
            cmap = "gray"
            vmin = self.vmin
            vmax = self.vmax
        elif plot_type == "resmap":
            image = self.resmaps[index]
            cmap = "inferno"
            vmin = self.vmin_resmap
            vmax = self.vmax_resmap
        # plot image
        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        fig.colorbar(im)
        title = plot_type + "\n" + self.filenames[index]
        plt.title(title)
        plt.show()


#### Image Processing Functions

## Functions for generating Resmaps


def calculate_resmaps(imgs_input, imgs_pred, method, dtype="float64"):
    if method == "L2":
        resmaps = resmaps_l2(imgs_input, imgs_pred)
    elif method == "SSIM":
        resmaps = resmaps_ssim(imgs_input, imgs_pred)
    if dtype == "uint8":
        resmaps = img_as_ubyte(resmaps)
    return resmaps


def resmaps_ssim(imgs_input, imgs_pred):
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    for index in range(len(imgs_input)):
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
        _, resmap = ssim(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
            full=True,
        )
        # resmap = np.expand_dims(resmap, axis=-1)
        resmaps[index] = 1 - resmap
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    return resmaps


def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2
    return resmaps


def segment_resmaps(resmaps, threshold):
    return resmaps > threshold


### utilitary functions


def get_plot_name(filename, suffix):
    filename_new, ext = os.path.splitext(filename)
    filename_new = "_".join(filename_new.split("/")) + "_" + suffix + ext
    return filename_new
