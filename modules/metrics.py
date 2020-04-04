import tensorflow as tf
from tensorflow import keras
import keras.backend as K


def ssim_metric(imgs_true, imgs_pred):
    """
    This is a slightly modified version of the original ssim function, 
    which is an objective function that returns values in range [-1,1].
    This modified implementation returns the mean of the SSIM values of images in batch, 
    plus 1 devided by 2, to scale values in range [0,1].
    Values approaching 0 indicate dissimilarity.
    Values approaching 1 indicate similarity.
    This implementation is used as a metric to monitor training. 
    """
    # return (K.mean(tf.image.ssim(imgs_true, imgs_pred, 1.0), axis=-1) + 1) / 2
    return K.mean(tf.image.ssim(imgs_true, imgs_pred, 1.0), axis=-1)


def mssim_metric(imgs_true, imgs_pred):
    """
    Returns the mean of the MSSIM values of images in batch.
    Retuend values are in range [0,1].
    Values approaching 0 indicate dissimilarity.
    Values approaching 1 indicate similarity.
    This objective function is used as a metric to monitor training. 
    """
    return K.mean(tf.image.ssim_multiscale(imgs_true, imgs_pred, 1.0), axis=-1)
