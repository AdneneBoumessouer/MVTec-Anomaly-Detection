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


# encoder = conv_ae(autoencoder.input, autoencoder.layers[-2].output)

# decoder_input = Input(shape=(encoding_dim,))
# decoder = Model(decoder_input, autoencoder.layers[-1](decoder_input))

# encoder.summary()
# decoder.summary()

# ====================== LOAD DATA ===============================

X_train, X_valid, X_test, y_test = utils.load_mvtec_data_as_tensor(
    dir_path="datasets/tensors", validation_split=0.1, numpy=False
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


# import csv
# train_dict = {
#         "epochs": "1",
#         "batch_size": "5",
#         "loss": "MSSIM",
#         "data_augmentation": "True",
#         "flow_from_directory": "True",
#     }
# with open('test.csv', 'w') as f:
#     for key in train_dict.keys():
#         f.write("%s: %s\n"%(key,train_dict[key]))


# OLD MODEL

# conv_encoder = keras.models.Sequential(
#     [
#         keras.layers.Conv2D(
#             32,
#             kernel_size=4,
#             strides=2,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#             input_shape=[256, 256, channels],
#         ),  # CONV0 (added layer)
#         # keras.layers.MaxPool2D(pool_size=2, padding="same"),
#         keras.layers.Conv2D(
#             32,
#             kernel_size=4,
#             strides=2,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),  # CONV1
#         # keras.layers.MaxPool2D(pool_size=2, padding="same"),
#         keras.layers.Conv2D(
#             32,
#             kernel_size=4,
#             strides=2,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),  # CONV2
#         # keras.layers.MaxPool2D(pool_size=2, padding="same"),
#         keras.layers.Conv2D(
#             32,
#             kernel_size=3,
#             strides=1,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),  # CONV3
#         # keras.layers.MaxPool2D(pool_size=2, padding="same"),
#         keras.layers.Conv2D(
#             64,
#             kernel_size=4,
#             strides=2,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),  # CONV4
#         # keras.layers.MaxPool2D(pool_size=2, padding="same"),
#         keras.layers.Conv2D(
#             64,
#             kernel_size=3,
#             strides=1,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),  # CONV5
#         # keras.layers.MaxPool2D(pool_size=2, padding="same"),
#         keras.layers.Conv2D(
#             128,
#             kernel_size=4,
#             strides=2,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),  # CONV6
#         # keras.layers.MaxPool2D(pool_size=2, padding="same"),
#         keras.layers.Conv2D(
#             64,
#             kernel_size=3,
#             strides=1,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),  # CONV7
#         # keras.layers.MaxPool2D(pool_size=2, padding="same"),
#         keras.layers.Conv2D(
#             32,
#             kernel_size=3,
#             strides=1,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),  # CONV8
#         # keras.layers.MaxPool2D(pool_size=2, padding="same"),
#         keras.layers.Conv2D(
#             100, kernel_size=8, strides=1, padding="VALID", activation="relu"
#         ),  # CONV9
#     ]
# )

# conv_decoder = keras.models.Sequential(
#     [
#         keras.layers.Conv2DTranspose(
#             32,
#             kernel_size=3,
#             strides=8,
#             padding="VALID",
#             activation="relu",
#             input_shape=[1, 1, 100],
#         ),
#         keras.layers.Conv2DTranspose(
#             64,
#             kernel_size=3,
#             strides=1,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),
#         keras.layers.Conv2DTranspose(
#             128,
#             kernel_size=4,
#             strides=1,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),
#         keras.layers.Conv2DTranspose(
#             64,
#             kernel_size=3,
#             strides=2,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),
#         keras.layers.Conv2DTranspose(
#             64,
#             kernel_size=4,
#             strides=1,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),
#         keras.layers.Conv2DTranspose(
#             32,
#             kernel_size=3,
#             strides=2,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),
#         keras.layers.Conv2DTranspose(
#             32,
#             kernel_size=4,
#             strides=1,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),
#         keras.layers.Conv2DTranspose(
#             32,
#             kernel_size=4,
#             strides=2,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),
#         keras.layers.Conv2DTranspose(
#             32,
#             kernel_size=4,
#             strides=2,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),
#         keras.layers.Conv2DTranspose(
#             channels,
#             kernel_size=4,
#             strides=2,
#             padding="SAME",
#             activation=keras.layers.LeakyReLU(0.2),
#         ),
#     ]
# )

