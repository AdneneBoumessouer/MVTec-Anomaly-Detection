import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from train_mvtec import ssim_loss as ssim_loss
from train_mvtec import mssim_loss as mssim_loss
import utils
import importlib


def residual_map(img1, img2):
    pass


# LOAD MODEL
# model_path = "saved_models/l2_loss/ae_50_datagen.h5"
model_path = "saved_models/SSIM/CAE_150_datagen.h5"

if ("MSE" or "l2_loss") in model_path.split("/"):
    CAE_mse = keras.models.load_model(
        filepath=model_path,
        custom_objects={
            "LeakyReLU": keras.layers.LeakyReLU
        },  # https://stackoverflow.com/questions/55364954/keras-load-model-cant-recognize-tensorflows-activation-functions
    )

elif "SSIM" in model_path.split("/"):
    CAE_ssim = keras.models.load_model(
        filepath=model_path,
        custom_objects={
            "LeakyReLU": keras.layers.LeakyReLU,
            "ssim_loss": ssim_loss,
        },  # https://stackoverflow.com/questions/55364954/keras-load-model-cant-recognize-tensorflows-activation-functions
    )

elif "SSIM" in model_path.split("/"):
    CAE_mssim = keras.models.load_model(
        filepath=model_path,
        custom_objects={
            "LeakyReLU": keras.layers.LeakyReLU,
            "ssim_loss": mssim_loss,
        },  # https://stackoverflow.com/questions/55364954/keras-load-model-cant-recognize-tensorflows-activation-functions
    )


# LOAD TEST SET
importlib.reload(utils)
_, _, X_test, y_test = utils.load_mvtec_data(model_path, numpy=False)

# PREDICT ON TEST SET
X_reconst = CAE_ssim.predict(X_test)

# COMPUTE RESIDUAL MAP
res_map = X_reconst - X_test

# CLASSIFICATION
# find a way to compute the threshold

# SAVE RESULTS IN A DATAFRAME
