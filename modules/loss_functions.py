import tensorflow as tf
from tensorflow import keras
import keras.backend as K


def ssim(img_true, img_pred):
    return (1 - K.mean(tf.image.ssim(img_true, img_pred, 1.0), axis=-1)) / 2


def mssim(img_true, img_pred):
    return 1 - K.mean(tf.image.ssim_multiscale(img_true, img_pred, 1.0), axis=-1)


def l2(img_true, img_pred):
    return 2 * tf.nn.l2_loss(img_true - img_pred)


# https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss?hl=ko
