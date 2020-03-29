import tensorflow as tf
from tensorflow import keras
import keras.backend as K


def ssim_loss(imgs_true, imgs_pred):
    """
    This is a slightly modified version of the original ssim function, 
    which is an objective function that returns values in range [-1,1].
    This modified implementation returns 1 - the mean of the SSIM values of images in batch, 
    devided by 2, to scale values in range [0,1].
    Values approaching 0 indicate similarity.
    Values approaching 1 indicate dissimilarity.
    This implementation is used as a loss function for training. 
    """
    # return (1 - K.mean(tf.image.ssim(imgs_true, imgs_pred, 1.0), axis=-1)) / 2
    return - K.mean(tf.image.ssim(imgs_true, imgs_pred, 1.0), axis=-1)


def mssim_loss(imgs_true, imgs_pred):
    """
    This is a slightly modified version of the original mssim function, 
    which is an objective function that returns values in range [0,1].
    This modified implementation returns 1 - the mean of the MSSIM values of images in batch. 
    Returned values are in range [0,1].
    Values approaching 0 indicate similarity.
    Values approaching 1 indicate dissimilarity.
    This implementation is used as a loss function for training. 
    """
    return 1 - K.mean(tf.image.ssim_multiscale(imgs_true, imgs_pred, 1.0), axis=-1)


def l2_loss(imgs_true, imgs_pred):
    """
    Returns the sum of the squares of the the pixelwise difference between two image batches.
    """
    return 2 * tf.nn.l2_loss(imgs_true - imgs_pred)


# https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss?hl=ko
