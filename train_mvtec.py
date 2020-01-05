#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:46:17 2019

@author: adnene33
"""

from numpy import expand_dims
import requests
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
# from keras.preprocessing.image import ImageDataGenerator #unknown issue for this import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np


# ========================= LOAD TRAINING DATASET =======================
X_train_full = np.load("X_train.npy")

X_train, X_valid = X_train_full[:-363], X_train_full[-363:]

plt.imshow(X_train[1000])
plt.show()


# ========================= DEFINE UTILITARY FUNCTIONS =======================

# TO DO: define SSIM loss function and train model with it

# ============================ DEFINE MODEL =================================
tf.random.set_seed(42)
np.random.seed(42)

conv_encoder = keras.models.Sequential(
    [
        keras.layers.Conv2D(
            32,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
            input_shape=[256, 256, 3],
        ),  # CONV0 (added layer)
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            32,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV1
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            32,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV2
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            32,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV3
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV4
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV5
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            128,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV6
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV7
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            32,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),  # CONV8
        # keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            100, kernel_size=8, strides=1, padding="VALID", activation="relu"
        ),  # CONV9
    ]
)

conv_decoder = keras.models.Sequential(
    [
        keras.layers.Conv2DTranspose(
            32,
            kernel_size=3,
            strides=8,
            padding="VALID",
            activation="relu",
            input_shape=[1, 1, 100],
        ),
        keras.layers.Conv2DTranspose(
            64,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Conv2DTranspose(
            128,
            kernel_size=4,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Conv2DTranspose(
            64,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Conv2DTranspose(
            64,
            kernel_size=4,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Conv2DTranspose(
            32,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Conv2DTranspose(
            32,
            kernel_size=4,
            strides=1,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Conv2DTranspose(
            32,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Conv2DTranspose(
            32,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Conv2DTranspose(
            3,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),
    ]
)


conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

conv_encoder.summary()
conv_decoder.summary()
conv_ae.summary()

# =============================== TRAINING SETUP =================================
batch_size = 32
epochs = 50
data_augmentation = True
flow_from_directory = False
# num_predictions = 20
save_dir = os.path.join(os.getcwd(), "saved_models/l2_loss")
model_name = "ae_50_datagen.h5"


optimizer = keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False
)

conv_ae.compile(
    loss="mean_squared_error", optimizer=optimizer, metrics=["mean_squared_error"]
)

# history = conv_ae.fit(X_train, X_train, epochs=50,
#                       validation_data=[X_valid, X_valid], shuffle=True)
# conv_ae.save('saved_models/l2_loss/ae_50.h5')


# =============================== TRAINING =================================

if not data_augmentation:
    print("Not using data augmentation.")
    conv_ae.fit(
        X_train,
        X_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_valid, X_valid),
        shuffle=True,
    )
else:
    print("Using real-time data augmentation.")
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=90,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.2,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.2,
        shear_range=0.0,  # set range for random shear
        zoom_range=0.0,  # set range for random zoom 0.15
        channel_shift_range=0.0,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode="nearest",
        cval=0.0,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format="channels_last",
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0,
    )  # change depending on flow or flow_from_directory

    if flow_from_directory:
        pass
        train_generator = datagen.flow_from_directory(
            directory="datasets/mvtec",
            target_size=(256, 256),
            color_mode="rgb",
            # classes=classes,  # does not work because of subdirectory architecture, see https://keras.io/preprocessing/image/#flow_from_directory
            # possible solution: combine datagenerators, see https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator?version=stable
            batch_size=32,
            class_mode="binary",
        )

        conv_ae.fit_generator(
            generator=train_generator,
            epochs=epochs,
            validation_data=(X_valid, X_valid),
            workers=-1,
        )
    else:
        # Fit the model on the batches generated by datagen.flow().
        train_generator = datagen.flow(
            X_train, X_train, batch_size=batch_size, shuffle=True
        )

        conv_ae.fit_generator(
            generator=train_generator,
            epochs=epochs,
            steps_per_epoch=np.ceil(len(X_train) / batch_size),
            validation_data=(X_valid, X_valid),
            workers=-1,
        )


# ====================== SEE RESULTS OF TRAINED AE =======================

# show reconstructed image sample--------------------------------
img = X_train[100]
plt.imshow(img)
plt.show()

# expand dimension to one sample
img_tensor = expand_dims(img, 0)
img_reconstruction = conv_ae.predict(img_tensor)
img_reconstruction_3d = img_reconstruction[0]
plt.imshow(img_reconstruction_3d)
plt.show()


# ----------------------------------------------------------------


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
conv_ae.save(model_path)
print("Saved trained model at %s " % model_path)


# load saved model
conv_ae = keras.models.load_model(
    "saved_models/l2_loss/ae_50_datagen.h5",
    custom_objects={"LeakyReLU": keras.layers.LeakyReLU},
)

# Score trained model.
# scores = conv_ae.evaluate(X_test, X_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])


# =============================== TESTING =================================
# load the test set

# make predictions on test set images
test_datagen = ImageDataGenerator()
