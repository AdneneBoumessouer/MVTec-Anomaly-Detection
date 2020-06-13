import os
import time

import numpy as np
from skimage.metrics import structural_similarity as ssim

from processing import utils
from processing.utils import printProgressBar as printProgressBar

import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte


VMIN_PRED = -1
VMAX_PRED = 1


class TensorImages:
    def __init__(
        self,
        imgs_input,
        imgs_pred,
        vmin,
        vmax,
        method,
        datatype="float",
        filenames=None,
    ):
        # pixel min and max values depend on preprocessing function,
        # which in turn depends on the model used for training.
        self.vmin_input = vmin
        self.vmax_input = vmax

        # compute resmaps
        self.resmaps = calculate_resmaps(imgs_input, imgs_pred, method)
        if datatype == "float":
            self.vmin_resmap = 0.0
            self.vmax_resmap = 1.0
            if method == "SSIM":
                self.step_seg = 1 / 255  # ~0.0039215
            elif method == "L2":
                self.step_seg = 1e-4  # ADJUST
        elif datatype == "uint8":
            self.step_seg = 1
            self.vmin_resmap = 0
            self.vmax_resmap = 255
            # Convert to 8-bit unsigned int
            self.resmaps = img_as_ubyte(self.resmaps)
        self.method = method
        self.filenames = filenames

    def generate_inspection_plots(
        self, imgs_input, imgs_pred, group="validation", save_dir=None
    ):
        l = len(self.filenames)
        printProgressBar(0, l, prefix="Progress:", suffix="Complete", length=50)
        for i in range(len(imgs_input)):
            f, axarr = plt.subplots(3, 1)
            f.set_size_inches((4, 9))

            im00 = axarr[0].imshow(
                imgs_input[i, :, :, 0],
                cmap="gray",
                vmin=self.vmin_input,
                vmax=self.vmax_input,
            )
            axarr[0].set_title("input")
            axarr[0].set_axis_off()
            f.colorbar(im00, ax=axarr[0])

            im10 = axarr[1].imshow(
                imgs_pred[i, :, :, 0], cmap="gray", vmin=VMIN_PRED, vmax=VMAX_PRED
            )
            axarr[1].set_title("pred")
            axarr[1].set_axis_off()
            f.colorbar(im10, ax=axarr[1])

            im20 = axarr[2].imshow(
                self.resmaps[i, :, :, 0],
                cmap="inferno",
                vmin=self.vmin_resmap,
                vmax=self.vmax_resmap,
            )
            axarr[2].set_title("resmap_" + self.method)
            axarr[2].set_axis_off()
            f.colorbar(im20, ax=axarr[2])

            plt.suptitle(group.upper() + "\n" + self.filenames[i])
            if save_dir is not None:
                plot_name = utils.get_plot_name(
                    self.filenames[i], suffix="inspection"
                )  # move to this module
                f.savefig(os.path.join(save_dir, plot_name))
            plt.close(fig=f)
            # print progress bar
            time.sleep(0.1)
            printProgressBar(i + 1, l, prefix="Progress:", suffix="Complete", length=50)
        return


def calculate_resmaps(imgs_input, imgs_pred, method):
    if method == "L2":
        resmaps = resmaps_l2(imgs_input, imgs_pred)
    elif method == "SSIM":
        resmaps = resmaps_ssim(imgs_input, imgs_pred)
    elif method == "MSSIM":
        resmaps = resmaps_mssim(imgs_input, imgs_pred)
    return resmaps


def resmaps_ssim(imgs_input, imgs_pred):
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    for i in range(len(imgs_input)):
        img_input = imgs_input[i, :, :, 0]
        img_pred = imgs_pred[i, :, :, 0]
        _, resmap = ssim(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
            full=True,
        )
        resmap = np.expand_dims(resmap, axis=-1)
        resmaps[i] = 1 - resmap
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    return resmaps


def resmaps_mssim(imgs_input, imgs_pred):
    # NOT TESTED YET
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    for i in range(len(imgs_input)):
        img_input = imgs_input[i, :, :]
        img_pred = imgs_pred[i, :, :]
        _, resmap = ssim(
            img_input,
            img_pred,
            multichannel=True,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
            full=True,
        )
        resmaps[i] = 1 - resmap
    resmaps = np.clip(resmaps, a_min=-1, amax=1)
    return resmaps


def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2
    return resmaps


def resmaps_mse(imgs_input, imgs_pred):
    pass


### Image Processing Functions
