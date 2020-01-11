import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import utils
import importlib
import custom_loss_functions
import os
from keras.preprocessing.image import ImageDataGenerator
import csv
import pandas as pd
import json

def load_trained_model(model_path):
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


# ================= LOAD TRAINED MODEL AND CORRESPONDING SETUP ===================
model_path = "saved_models/MSSIM/11-01-2020_14:18:20/CAE_50_flow_from_dir_datagen.h5"
conv_ae, train_setup, history = load_trained_model(model_path)

# ====================== LOAD DATA ===============================

# X_train, X_valid, X_test, y_test = utils.load_mvtec_data_as_tensor(
#     dir_path="datasets/tensors", validation_split=0.1, numpy=False
# )

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = validation_datagen.flow_from_directory(
                directory="datasets/data/validation",
                target_size=(256, 256),
                color_mode=train_setup["color_mode"],
                batch_size=1,
                class_mode="input",
            )


img = validation_generator.next()[0] 
img_pred = conv_ae.predict(img)
plt.imshow(img_pred[0])
utils.compare_images(img[0], model=conv_ae)

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




img_imread = mpimg.imread("datasets/data/train/good/bottle_000.png")  # scaled [0,1]


# plot image, reconstruction and residual map
img1 = X_train[1200]
utils.compare_images(img1, model=conv_ae)
img2 = conv_ae.predict(tf.expand_dims(img1, 0))[0]
res_map = utils.residual_map_image(img1, img2)
plt.imshow(res_map[:, :, 0], cmap=plt.cm.gray, vmin=0, vmax=1)




#--------------------------------------------------------------------------

# show encoder and decoder architecture

# encoder = conv_ae(autoencoder.input, autoencoder.layers[-2].output)

# decoder_input = Input(shape=(encoding_dim,))
# decoder = Model(decoder_input, autoencoder.layers[-1](decoder_input))

# encoder.summary()
# decoder.summary()


#--------------------------------------------------------------------------

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

