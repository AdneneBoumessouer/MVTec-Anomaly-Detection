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

mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

for i in range(3):
    print(next(myit))

for i in range(3):
    print(next(myit))

# ========================== TRAIN ====================================
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255,
    validation_split=0.1)

train_data_dir = 'mvtec/cable/train'

train_generator = train_datagen.flow_from_directory(
    train_data_dir, #SAME
    target_size=(256, 256),
    batch_size=4,
    class_mode='input',
    shuffle=True, ######
    subset='training')

filenames_train = train_generator.filenames

validation_generator = test_datagen.flow_from_directory(
    train_data_dir, #SAME
    target_size=(256, 256),
    batch_size=1,
    class_mode='input',
    subset='validation')

filenames_valid = validation_generator.filenames

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# ========================== TEST ====================================


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
    rescale=1. / 255)

test_data_dir = 'mvtec/cable/test'

test_generator = test_datagen.flow_from_directory(
    test_data_dir, #SAME
    target_size=(256, 256),
    batch_size=1,
    class_mode='input',
    shuffle=False)

filenames_test = test_generator.filenames




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

validation_generator_1 = validation_datagen.flow_from_directory(
                directory="mvtec/pill/train/good",
                target_size=(256, 256),
                color_mode="grayscale",
                batch_size=1,
                class_mode="input",
            )

validation_generator_2 = validation_datagen.flow_from_directory(
                directory="mvtec/pill/test/crack",
                target_size=(256, 256),
                color_mode="grayscale",
                batch_size=1,
                class_mode="input",
            )

img = validation_generator_1.next()[0] 
# img_pred = model.predict(img)
# plt.imshow(img_pred[0])
utils.compare_images(img[0], model=model)

# this does not work

img = validation_generator.next()[0] 
# img_pred = model.predict(img)
# plt.imshow(img_pred[0])
utils.compare_images(img[0], model=model)

# ===========================================================================
# with flow
importlib.reload(utils)

X_train, X_valid, X_test, y_test = utils.load_mvtec_data_as_tensor('datasets/tensors')

# model_path = 'saved_models/l2_loss/ae_50_datagen.h5'
model_path = 'saved_models/SSIM/CAE_150_datagen.h5'
model = keras.models.load_model(
    filepath=model_path,
    custom_objects={
        "LeakyReLU": keras.layers.LeakyReLU,
        "ssim_loss": custom_loss_functions.ssim_loss,
    },  # https://stackoverflow.com/questions/55364954/keras-load-model-cant-recognize-tensorflows-activation-functions
)

X_train = utils.preprocess_tensor(X_train, 'SSIM')
img1 = X_train[1501]
# img1 = tf.image.rgb_to_grayscale(img1)
img2 =  model.predict(tf.expand_dims(img1, 0))[0]
utils.compare_images(img1, img2)


import matplotlib.image as mpimg
X_test = utils.preprocess_tensor(X_test, 'SSIM')
img = X_test[720] #mpimg.imread("mvtec/pill/test/crack/003.png") # scaled [0,1]
# img = tf.image.rgb_to_grayscale(img)
img3 =  model.predict(tf.expand_dims(img, 0))[0]
utils.compare_images(img, img3)



# plot image, reconstruction and residual map
np.load()
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

