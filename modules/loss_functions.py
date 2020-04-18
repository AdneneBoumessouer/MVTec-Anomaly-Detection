import tensorflow as tf
from tensorflow import keras
import keras.backend as K


def ssim_loss(dynamic_range):
    # return (1 - K.mean(tf.image.ssim(imgs_true, imgs_pred, 1.0), axis=-1)) / 2
    def loss(imgs_true, imgs_pred):
        return -K.mean(tf.image.ssim(imgs_true, imgs_pred, dynamic_range), axis=-1)

    return loss


# def mssim_loss(imgs_true, imgs_pred):
#     # return 1 - K.mean(tf.image.ssim_multiscale(imgs_true, imgs_pred, 1.0), axis=-1)
#     # return -K.mean(tf.image.ssim_multiscale(imgs_true, imgs_pred, 1.0), axis=-1)
#     return -K.mean(tf.image.ssim_multiscale(imgs_true, imgs_pred, 2.0), axis=-1)


def mssim_loss(dynamic_range):
    # return 1 - K.mean(tf.image.ssim_multiscale(imgs_true, imgs_pred, 1.0), axis=-1)
    # return -K.mean(tf.image.ssim_multiscale(imgs_true, imgs_pred, 1.0), axis=-1)
    # return -K.mean(tf.image.ssim_multiscale(imgs_true, imgs_pred, 2.0), axis=-1)
    def loss(imgs_true, imgs_pred):
        return -K.mean(
            tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range), axis=-1
        )

    return loss


def l2_loss(imgs_true, imgs_pred):
    return 2 * tf.nn.l2_loss(imgs_true - imgs_pred)


# https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss?hl=ko
