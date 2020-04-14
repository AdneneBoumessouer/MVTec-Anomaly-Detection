import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.metrics import structural_similarity as ssim


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


def calculate_resmaps(imgs_input, imgs_pred, method):
    if method == "L2":
        resmaps = resmaps_l2(imgs_input, imgs_pred)
    elif method == "MSE":
        resmaps = resmaps_l2(imgs_input, imgs_pred)
    elif method == "SSIM":
        resmaps = resmaps_ssim(imgs_input, imgs_pred)
    elif method == "MSSIM":
        resmaps = resmaps_mssim(imgs_input, imgs_pred)
    return resmaps
