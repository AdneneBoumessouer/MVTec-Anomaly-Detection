from numpy import expand_dims
import matplotlib.pyplot as plt
import tensorflow as tf


def show_original_and_reconstructed_img(img, model):
    fig, axes = plt.subplots(1, 2)

    ax = axes.ravel()

    ax[0].imshow(img)

    # expand dimension to one sample
    img_tensor = expand_dims(img, 0)
    img_reconstruction = model.predict(img_tensor)
    img_reconstruction = img_reconstruction[0]
    ax[1].imshow(img_reconstruction)

    # show plot
    plt.show()


def show_original_and_reconstructed_img_gray(image, model):
    fig, axes = plt.subplots(1, 2)
    ax = axes.ravel()
    # label = 'MSE: {:.2f}, SSIM: {:.2f}'

    img1 = tf.convert_to_tensor(expand_dims(image, 0))
    ax[0].imshow(img1.numpy()[0, :, :, 0], cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[0].set_title("Original image")

    img2 = tf.convert_to_tensor(model.predict(expand_dims(image, 0)))
    ax[1].imshow(img2.numpy()[0, :, :, 0], cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[1].set_title("Reconstructed image")

    ssim_value = tf.image.ssim(img1=img1, img2=img2, max_val=1.0)
    print("SSIM: {:.2f}".format(ssim_value.numpy()[0]))

    plt.show()


# --------------------------keep this as backup--------------------------------
# max_value = 1.0
# def custom_loss(max_value):
#     def loss(y_true, y_pred):
#         return tf.image.ssim(img1=y_true, img2=y_pred, max_val=max_value)

#     return loss

# loss=custom_loss(max_value),
# metrics=[custom_loss(max_value)],
# -----------------------------------------------------------------------------


# --------tested, works on specific object classes and fails on others---------
# def ssim_loss(y_true, y_pred):
#     return -1 * tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
# -----------------------------------------------------------------------------

# # img1 and img2 must be image tensors
# sim_value = tf.image.ssim(
#     img1=tf.convert_to_tensor(X_train[:4]),
#     img2=tf.convert_to_tensor(X_train[:4]),
#     max_val=1.0,
# )


# img1 = tf.convert_to_tensor(expand_dims(X_train[500], 0))
# plt.imshow(img1.numpy()[0, :, :, 0])
# img2 = tf.convert_to_tensor(conv_ae.predict(expand_dims(X_train[500], 0)))
# plt.imshow(img2.numpy()[0, :, :, 0])
# sim_value = tf.image.ssim(img1=img1, img2=img2, max_val=1.0)
# # returns <tf.Tensor: id=23418566, shape=(1,), dtype=float32, numpy=array([-0.72969997], dtype=float32)>

# # returns scalar value for similarity
# tf.image.ssim(img1=img1, img2=img2, max_val=1.0).numpy()[0]  # return -0.72969997
