import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import utils
import importlib
import custom_loss_functions

# ====================== LOAD TRAINED MODEL =======================
model_path = "saved_models/SSIM/CAE_150_datagen.h5"

conv_ae = keras.models.load_model(
    filepath=model_path,
    custom_objects={
        "LeakyReLU": keras.layers.LeakyReLU,
        "ssim_loss": custom_loss_functions.ssim_loss,
    },  # https://stackoverflow.com/questions/55364954/keras-load-model-cant-recognize-tensorflows-activation-functions
)

# ====================== LOAD DATA ===============================

X_train, X_valid, X_test, y_test = utils.load_mvtec_data(model_path, numpy=False)

# ====================== SEE RESULTS OF TRAINED AE =======================
X_train_full = np.load("X_train.npy")
importlib.reload(utils)

utils.compare_images(X_train[100], X_train[101])
utils.compare_images(X_train[100], X_train[500])
utils.compare_images(X_train[100], X_train[700])
utils.compare_images(X_train[100], X_train[1100])
utils.compare_images(X_train[1101], X_train[1100])

utils.compare_images(X_train[500], model=conv_ae)

utils.compare_images(X_train_full[100], X_train_full[101])
utils.compare_images(X_train_full[100], X_train_full[500])
utils.compare_images(X_train_full[100], X_train_full[700])
utils.compare_images(X_train_full[100], X_train_full[1100])
utils.compare_images(X_train_full[1101], X_train_full[1100])


# "[python]": {
#     "editor.defaultFormatter": "ms-python.python"
# }

