import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from autoencoder import metrics
from autoencoder import losses


def get_model_info(model_path):
    dir_name = os.path.dirname(model_path)
    with open(os.path.join(dir_name, "info.json"), "r") as read_file:
        info = json.load(read_file)
    return info


def load_model_HDF5(model_path):
    """
    Loads model (HDF5 format), training setup and training history.
    """

    # load parameters
    info = get_model_info(model_path)
    loss = info["model"]["loss"]
    dynamic_range = info["preprocessing"]["dynamic_range"]

    # load autoencoder
    if loss == "mssim":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "loss": losses.mssim_loss(dynamic_range),
                "mssim": metrics.mssim_metric(dynamic_range),
            },
            compile=True,
        )

    elif loss == "ssim":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "loss": losses.ssim_loss(dynamic_range),
                "ssim": metrics.ssim_metric(dynamic_range),
            },
            compile=True,
        )

    else:
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "l2_loss": losses.l2_loss,
                "ssim": losses.ssim_loss(dynamic_range),
                "mssim": metrics.mssim_metric(dynamic_range),
            },
            compile=True,
        )

    # load training history
    dir_name = os.path.dirname(model_path)
    history = pd.read_csv(os.path.join(dir_name, "history.csv"))

    return model, info, history


def save_np(arr, save_dir, filename):
    np.save(
        file=os.path.join(save_dir, filename), arr=arr, allow_pickle=True,
    )


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def update_history(history1, history2):
    dict3 = {}
    for key in list(history1.history.keys()):
        dict3[key] = []
        dict3[key].extend(history1.history[key])
        dict3[key].extend(history2.history[key])
    history1.history = dict3
    return history1


def generate_new_name(filename, suffix):
    filename_new, ext = os.path.splitext(filename)
    filename_new = "_".join(filename_new.split("/")) + "_" + suffix + ext
    return filename_new


def save_images(save_dir, imgs, filenames, color_mode, suffix):
    filenames_new = []
    for filename in filenames:
        filename_new, ext = os.path.splitext(filename)
        filename_new = os.path.basename(filename_new)
        filename_new = filename_new + "_" + suffix + ext
        filenames_new.append(filename_new)

    if color_mode == "grayscale":
        for i in range(len(imgs)):
            img = imgs[i, :, :, 0]
            save_path = os.path.join(save_dir, filenames_new[i])
            plt.imsave(save_path, img, cmap="gray")

    if color_mode == "RGB":
        for i in range(len(imgs)):
            img = imgs[i, :, :, 0]
            save_path = os.path.join(save_dir, filenames_new[i])
            plt.imsave(save_path, img)

    print("[INFO] validation images for inspection saved at /{}".format(save_dir))


def plot_inspection_images(tensor_list, index):
    titles = ["input", "pred", "resmaps_diff", "resmap_ssim", "resmap_L2"]
    cmaps = ["gray", "gray", "inferno", "inferno", "inferno"]
    dyn_ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    cols = 2
    lines = 3

    f, axarr = plt.subplots(3, 2)
    f.set_size_inches((8, 9))
    for k, tensor in enumerate(tensor_list):
        l, c = list(divmod(k, cols))
        vmin, vmax = dyn_ranges[k]
        im = axarr[l, c].imshow(
            tensor[index, :, :, 0], cmap=cmaps[k], vmin=vmin, vmax=vmax
        )
        axarr[l, c].set_title(titles[k])
        axarr[l, c].set_axis_off()
        f.colorbar(im, ax=axarr[l, c])

    for p in range(k + 1, cols * lines):
        l, c = list(divmod(p, 2))
        axarr[l, c].set_axis_off()

    return f


def save_dataframe_as_text_file(df, save_dir, filename):
    with open(os.path.join(save_dir, filename), "w+") as f:
        f.write(df.to_string(header=True, index=True))
        print("[INFO] validation_results.txt saved at {}".format(save_dir))
