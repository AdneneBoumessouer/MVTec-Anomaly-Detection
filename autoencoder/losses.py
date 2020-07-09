import tensorflow as tf

# from tensorflow import keras
# import keras.backend as K
# from tensorflow.keras import backend as K


def ssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):
        # return -K.mean(tf.image.ssim(imgs_true, imgs_pred, dynamic_range), axis=-1)

        return (-1 * tf.image.ssim(imgs_true, imgs_pred, dynamic_range) + 1) / 2

        # return K.mean(-1 * tf.image.ssim(imgs_true, imgs_pred, dynamic_range) + 1) / 2

    return loss


def mssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):
        # return -K.mean(
        #     tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range), axis=-1
        # )
        return -1 * tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range) + 1

    return loss


def l2_loss(imgs_true, imgs_pred):
    # return 2 * tf.nn.l2_loss(imgs_true - imgs_pred)
    return tf.nn.l2_loss(imgs_true - imgs_pred)


# https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss?hl=ko
