import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from train_mvtec import ssim_loss as ssim_loss


def residual_map(img1, img2):
    pass


# LOAD MODEL
conv_ae = keras.models.load_model(
    "saved_models/l2_loss/ae_50_datagen.h5",
    custom_objects={
        "LeakyReLU": keras.layers.LeakyReLU
    },  # https://stackoverflow.com/questions/55364954/keras-load-model-cant-recognize-tensorflows-activation-functions
)

conv_ae = keras.models.load_model(
    "saved_models/SSIM/CAE_150_datagen.h5",
    custom_objects={
        "LeakyReLU": keras.layers.LeakyReLU,
        "ssim_loss": ssim_loss,
    },  # https://stackoverflow.com/questions/55364954/keras-load-model-cant-recognize-tensorflows-activation-functions
)


# LOAD TEST SET
X_test, y_test = np.load("X_test.npy"), np.load("y_test.npy")

# PREDICT ON TEST SET
X_reconst = conv_ae.predict(X_test)

# COMPUTE RESIDUAL MAP
res_map = X_reconst - X_test

# CLASSIFICATION
# find a way to compute the threshold

# SAVE RESULTS IN A DATAFRAME
