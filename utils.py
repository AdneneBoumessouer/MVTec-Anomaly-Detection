from numpy import expand_dims
import matplotlib.pyplot as plt


def show_original_ans_reconstructed_img(img, model):
    # img = X_train[100]
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    plt.imshow(img)
    # plt.show()

    # expand dimension to one sample
    img_tensor = expand_dims(img, 0)
    img_reconstruction = model.predict(img_tensor)
    img_reconstruction = img_reconstruction[0]

    fig.add_subplot(2, 1, 2)
    plt.imshow(img_reconstruction)
    plt.show()
