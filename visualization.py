from train import AugmentSpeckle
import imageio
import numpy as np
import matplotlib.pyplot as plt


def unlog_image(im_log, vmax=10.089, vmin=-1.423):
    """

    :param im_log: Image in [0 1]
    :param vmax:
    :param vmin:
    :return:
    """
    return np.exp((vmax - vmin) * im_log + vmin)


def plot_speckle_example():
    im = imageio.imread('Examples/denoised_lely10.png')
    im = im / 255.0

    augmenters = [AugmentSpeckle(l, normalize=True) for l in range(1, 6)]

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(unlog_image(im), cmap='gray')
    plt.title('Original Image')

    for idx, augm in enumerate(augmenters):
        im_noise = augm.restore_bias_np(
            augm.add_validation_noise_np(im))

        plt.subplot(2, 3, idx + 2)
        plt.imshow(unlog_image(im_noise), cmap='gray')
        plt.title('Nb views = %d' % (idx + 1))

    plt.show()


def log_validation_plot(path, l, id_image=None):
    noiser = AugmentSpeckle(l)

    if id_image is None:
        id_image = list(range(3))

    nb_image = len(id_image)

    for idx, file_id in enumerate(id_image):

        images_descr = [str(file_id) + ' - Original Image',
                        str(file_id) + ' - Noisy Image - L = ' + str(l),
                        str(file_id) + ' - Predicted Image']

        images = [imageio.imread(path + '/img_0_val_' + str(file_id) + '_orig.png'),
                  noiser.restore_bias_np(imageio.imread(path + '/img_0_val_' + str(file_id) + '_noisy.png')),
                  noiser.restore_bias_np(imageio.imread(path + '/img_0_val_' + str(file_id) + '_pred.png'))]

        for id_im, image in enumerate(images):
            image = unlog_image(image / 255)

            plt.subplot(nb_image, 3, 3 * idx + id_im + 1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.title(images_descr[id_im])

    plt.show()


if __name__ == '__main__':
    l = 5
    log_validation_plot('results/00018-autoencoder', l, [0, 25, 50])
