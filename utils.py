from numpy import expand_dims
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow import keras
import custom_loss_functions
import os
import csv
import pandas as pd
import json

def load_trained_model(model_path):
    """Loads model, training setup and training history"""
    # load autoencoder
    loss = model_path.split('/')[1]
    if loss == "MSSIM":
        conv_ae = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "mssim_loss": custom_loss_functions.mssim_loss,
            },  # https://stackoverflow.com/questions/55364954/keras-load-model-cant-recognize-tensorflows-activation-functions
        )

    elif loss == "SSIM":
        conv_ae = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "ssim_loss": custom_loss_functions.ssim_loss,
            },
        )       
    
    else:
        conv_ae = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
            },
        )

    # load training history
    dir_name = os.path.dirname(model_path)
    history = pd.read_csv(os.path.join(dir_name,"history.csv"))

    # load training setup
    with open(os.path.join(dir_name, "train_setup.json"), "r") as read_file:
        train_setup = json.load(read_file)

    return conv_ae, train_setup, history

# old_path = "saved_models/MSSIM/11-01-2020_14:18:20/CAE_50_flow_from_dir_datagen_0.h5"
def new_path(old_path):
    """create a new filename for pretrained model"""
    dir_name = os.path.dirname(old_path)
    suffix = os.path.basename(old_path).split('.')[1]
    old_filename = os.path.basename(old_path).split('.')[0]    
    prefix = old_filename[:-1]
    index = old_filename[-1]
    new_index = str(int(index)+1)
    new_filename = prefix + new_index + '.' + suffix
    return os.path.join(dir_name, new_filename)




def load_mvtec_data_as_tensor(dir_path, validation_split=0.1, numpy=False):
    """
    Loads training and test sets as Tensors or as numpy arrays from directory
    and returns training set, validation set (based on validation_split) and test set.
    """
    # load training data
    X_train_full = np.load(os.path.join(dir_path, "X_train.npy"))
    # TO DO : stratified sampling
    split_index = int(len(X_train_full) * (1 - validation_split))
    X_train, X_valid = X_train_full[:split_index], X_train_full[split_index:]

    # load testing data
    X_test, y_test = (
        np.load(os.path.join(dir_path, "X_test.npy")),
        np.load(os.path.join(dir_path, "y_test.npy")),
    )

    if numpy:
        X_train = X_train.numpy()
        X_valid = X_valid.numpy()
        X_test = X_test.numpy()

    return X_train, X_valid, X_test, y_test


def preprocess_tensor(tensor, loss):
    if loss == "SSIM":
        tensor = tf.image.rgb_to_grayscale(tensor)
    else:
        tensor = tf.convert_to_tensor(tensor)
    return tensor


def compare_images(img1, img2=None, model=None):
    """
    Plots img1 and img2 side by side and compute their similarity measure.
    If img2 is None and model is passed, img2 is recontructed from model.
    Model input and images' shape must be consistent! 
    """

    if (type(img2) == type(None) and model == None) or (
        type(img2) != type(None) and model != None
    ):
        raise ValueError("Pass EITHER img2 OR model to reconstruct img2 from")

    if type(img2) == type(None) and model != None:
        img2 = model.predict(tf.expand_dims(img1, 0))
        img2 = img2[0]
        img1_title = "original"
        img2_title = "reconstructed"
    else:
        img1_title = "image 1"
        img2_title = "image 2"

    _, axes = plt.subplots(1, 2)
    ax = axes.ravel()

    if img1.shape[-1] == img1.shape[-1] == 3:
        # executes when images are RGB
        ax[0].imshow(img1, vmin=0, vmax=1)
        ax[0].set_title(img1_title)
        ax[1].imshow(img2, vmin=0, vmax=1)
        ax[1].set_title(img2_title)
        mssim_value = tf.image.ssim_multiscale(
            img1=tf.expand_dims(img1, 0), img2=tf.expand_dims(img2, 0), max_val=1.0
        )
        print("multiscale_SSIM = {:.2f}".format(mssim_value.numpy()[0]))

    elif img1.shape[-1] == img1.shape[-1] == 1:
        # executes when imgages are Greyscaled

        ax[0].imshow(img1[:, :, 0], cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[0].set_title(img1_title)
        ax[1].imshow(img2[:, :, 0], cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[1].set_title(img2_title)
        ssim_value = tf.image.ssim(
            img1=tf.expand_dims(img1, 0), img2=tf.expand_dims(img2, 0), max_val=1.0
        )
        print("SSIM: {:.2f}".format(ssim_value.numpy()[0]))
    else:
        raise ValueError("image shapes are not consistent!")
    plt.show()


def residual_map_image(img1, img2):
    """
    Returns the Residual Map of two batches of tensor images
    """
    return img1 - img2

