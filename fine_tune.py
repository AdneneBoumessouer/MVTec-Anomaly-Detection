import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import utils
import importlib
import custom_loss_functions
import os

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

X_train, X_valid, X_test, y_test = utils.load_mvtec_data_as_tensor(
    dir_path="datasets/tensors", loss="SSIM", numpy=False
)

# ====================== SEE RESULTS OF TRAINED AE =======================

# with flow
X_train_full = np.load("X_train.npy")
importlib.reload(utils)

utils.compare_images(X_train[100], X_train[101])
utils.compare_images(X_train[100], X_train[500])
utils.compare_images(X_train[100], X_train[700])
utils.compare_images(X_train[100], X_train[1100])
utils.compare_images(X_train[1101], X_train[1100])

utils.compare_images(X_train[1500], model=conv_ae)
import matplotlib.image as mpimg

img = mpimg.imread("datasets/data/train/good/bottle_000.png")


utils.compare_images(X_train_full[100], X_train_full[101])
utils.compare_images(X_train_full[100], X_train_full[500])
utils.compare_images(X_train_full[100], X_train_full[700])
utils.compare_images(X_train_full[100], X_train_full[1100])
utils.compare_images(X_train_full[1101], X_train_full[1100])


# with flow_from_directory: use after running train_mvtec.py !
img = validation_generator.next()[0]  # NOT SCLED [0, 255]
utils.compare_images(img[0], model=conv_ae)

img_imread = mpimg.imread("datasets/data/train/good/bottle_000.png")  # scaled [0,1]


# plot image, reconstruction and residual map
img1 = X_train[1200]
utils.compare_images(img1, model=conv_ae)
img2 = conv_ae.predict(tf.expand_dims(img1, 0))[0]
res_map = utils.residual_map_image(img1, img2)
plt.imshow(res_map[:, :, 0], cmap=plt.cm.gray, vmin=0, vmax=1)


os.path.join("datasets/tensors", "X_train.npy")

