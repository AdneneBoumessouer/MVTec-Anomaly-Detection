from numpy import expand_dims
import matplotlib.pyplot as plt


def show_original_and_reconstructed_img(img, model):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)

    # expand dimension to one sample
    img_tensor = expand_dims(img, 0)
    img_reconstruction = model.predict(img_tensor)
    img_reconstruction = img_reconstruction[0]
    axs[1].imshow(img_reconstruction)

    # show plot
    plt.show()
