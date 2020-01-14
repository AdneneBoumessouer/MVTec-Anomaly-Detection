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

"""THIS SCRIPT IS MESSY AND NOT YET COMPLETE"""



# ================= LOAD TRAINED MODEL AND CORRESPONDING SETUP ===================
model_path = "saved_models/MSSIM/11-01-2020_14:18:20/CAE_e50_b4_0.h5"
model, train_setup, history = utils.load_model_HDF5(model_path)


# X_train, X_valid, X_test, y_test = utils.load_mvtec_data_as_tensor(
#     dir_path="datasets/tensors", validation_split=0.1, numpy=False
# )

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = validation_datagen.flow_from_directory(
                directory="datasets/data/validation",
                target_size=(256, 256),
                color_mode=train_setup["color_mode"],
                batch_size=train_setup["batch_size"],
                class_mode="input",
            )
# ====================== Experimentation ======================
# try to train loaded model


# # set loss function, optimizer and metric
# if train_setup["loss"] == "SSIM":
#     loss_function = custom_loss_functions.ssim_loss

#     optimizer = keras.optimizers.Adam(
#         learning_rate=2e-4, beta_1=0.9, beta_2=0.999, amsgrad=False
#     )
#     model.compile(
#         loss=loss_function,
#         optimizer=optimizer,
#         metrics=[loss_function, "mean_squared_error"],
#     )

# elif train_setup["loss"] == "MSSIM":
#     loss_function = custom_loss_functions.mssim_loss
#     optimizer = keras.optimizers.Adam(
#         learning_rate=2e-4, beta_1=0.9, beta_2=0.999, amsgrad=False
#     )
#     model.compile(
#         loss=loss_function,
#         optimizer=optimizer,
#         metrics=[loss_function, "mean_squared_error"],
#     )

# elif train_setup["loss"] == "MSE":
#     loss_function = "mean_squared_error"
#     optimizer = keras.optimizers.Adam(
#         learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False
#     )
#     model.compile(
#         loss=loss_function, optimizer=optimizer, metrics=["mean_squared_error"]
#     )


# history_test = model.fit_generator(
#         generator=validation_generator,
#         epochs=train_setup["epochs"],
#         steps_per_epoch=validation_generator.samples // train_setup["batch_size"],
#         workers=-1,
#     )
# =======================================================================
model_path = "saved_models/MSSIM/11-01-2020_14:18:20/CAE_e50_b4_0.h5"
model, train_setup, history = utils.load_model_HDF5(model_path)

!mkdir -p saved_model
model.save('saved_model/my_model')
new_model = tf.keras.models.load_model('saved_model/my_model')



# =======================================================================

img = validation_generator.next()[0] 
# img_pred = model.predict(img)
# plt.imshow(img_pred[0])
utils.compare_images(img[0], model=model)

# ====================== SEE RESULTS OF TRAINED AE =======================


model_path=  "saved_models/SSIM/CAE_150_datagen.h5"
model = keras.models.load_model(
    filepath=model_path,
    custom_objects={
        "LeakyReLU": keras.layers.LeakyReLU,
        "ssim_loss": custom_loss_functions.ssim_loss,
    },  # https://stackoverflow.com/questions/55364954/keras-load-model-cant-recognize-tensorflows-activation-functions
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = validation_datagen.flow_from_directory(
                directory="datasets/data/validation",
                target_size=(256, 256),
                color_mode="grayscale",
                batch_size=1,
                class_mode="input",
            )

img = validation_generator.next()[0] 
# img_pred = model.predict(img)
# plt.imshow(img_pred[0])
utils.compare_images(img[0], model=model)

# this does not work

img = validation_generator.next()[0] 
# img_pred = model.predict(img)
# plt.imshow(img_pred[0])
utils.compare_images(img[0], model=model)


# with flow
X_train_full = np.load("X_train.npy")
importlib.reload(utils)


utils.compare_images(X_train[1500], model=model)
import matplotlib.image as mpimg

img = mpimg.imread("datasets/data/train/good/bottle_000.png")


img_imread = mpimg.imread("datasets/data/train/good/bottle_000.png")  # scaled [0,1]


# plot image, reconstruction and residual map
img1 = X_train[1200]
utils.compare_images(img1, model=model)
img2 = model.predict(tf.expand_dims(img1, 0))[0]
res_map = utils.residual_map_image(img1, img2)
plt.imshow(res_map[:, :, 0], cmap=plt.cm.gray, vmin=0, vmax=1)


#--------------------------------------------------------------------------

# show encoder and decoder architecture

# encoder = model(autoencoder.input, autoencoder.layers[-2].output)

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

