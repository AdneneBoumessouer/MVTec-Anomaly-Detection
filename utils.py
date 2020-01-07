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


def show_original_and_reconstructed_img_2(image, model):
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
