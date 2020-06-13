import tensorflow as tf
from tensorflow import keras

# Preprocessing variables
RESCALE = None
SHAPE = (299, 299)  # check if correct
PREPROCESSING_FUNCTION = keras.applications.inception_resnet_v2.preprocess_input
PREPROCESSING = "keras.applications.inception_resnet_v2.preprocess_input"
VMIN = -1.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN

# (Doesn't work)
def build_model():
    message = "model not yet implemented"
    return message
