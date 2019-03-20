from train import AugmentSpeckle, AugmentGaussian
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


def log_validation_plot(path, l_nb_view, id_image=None, is_log=True, noise='speckle', nb_channel=1):
    if noise == 'gaussian':
        noiser = AugmentGaussian(0.02 * 255, [0.02 * 255, 0.02 * 255])
    else:
        noiser = AugmentSpeckle(l_nb_view)

    if id_image is None:
        id_image = list(range(3))

    nb_image = len(id_image)

    plt.figure()

    for idx, file_id in enumerate(id_image):

        images_descr = [str(file_id) + ' - Original Image',
                        str(file_id) + ' - Noisy Image - L = ' + str(l_nb_view),
                        # str(file_id) + ' - Noisy Image - Gaussian ~ L = ' + str(l_nb_view),
                        str(file_id) + ' - Predicted Image']

        if noise == 'speckle':
            if is_log:
                images = [imageio.imread(path + '/img_0_val_' + str(file_id) + '_orig.png'),
                          noiser.restore_bias_np(imageio.imread(path + '/img_0_val_' + str(file_id) + '_noisy.png')),
                          noiser.restore_bias_np(imageio.imread(path + '/img_0_val_' + str(file_id) + '_pred.png'))]
            else:
                images = [imageio.imread(path + '/img_0_val_' + str(file_id) + '_orig.png'),
                          imageio.imread(path + '/img_0_val_' + str(file_id) + '_noisy.png'),
                          imageio.imread(path + '/img_0_val_' + str(file_id) + '_pred.png')]
        else:
            images = [imageio.imread(path + '/img_0_val_' + str(file_id) + '_orig.png'),
                      imageio.imread(path + '/img_0_val_' + str(file_id) + '_noisy.png'),
                      imageio.imread(path + '/img_0_val_' + str(file_id) + '_pred.png')]

        for id_im, image in enumerate(images):
            if is_log:
                image = unlog_image(image / 255)

            if nb_channel > 1 and len(image.shape) == 3:
                # image = image.mean(axis=2)
                image = image[:, :, 0]

            plt.subplot(nb_image, 3, 3 * idx + id_im + 1)
            plt.imshow(image, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.title(images_descr[id_im])

    plt.show(block=False)


def plot_noise_distr():
    speckle1 = AugmentSpeckle(l_nb_views=1, quick_noise_computation=True)
    speckle5 = AugmentSpeckle(l_nb_views=5, quick_noise_computation=True)

    gauss_eq5 = np.random.normal(speckle5.noise_sample.mean(),
                                 speckle5.noise_sample.std(),
                                 speckle5.noise_sample.shape)
    gauss_eq1 = np.random.normal(speckle1.noise_sample.mean(),
                                 speckle1.noise_sample.std(),
                                 speckle1.noise_sample.shape)

    plt.figure()
    plt.subplot(121)
    plt.hist(speckle1.noise_sample, 1000, alpha=0.5, label='Speckle Noise')
    plt.hist(gauss_eq1, 1000, alpha=0.5, label='Equivalent Gaussian')
    plt.title('Distribution of Specke Noise for 1 view (L=1)')
    plt.legend()

    plt.subplot(122)
    plt.hist(speckle5.noise_sample, 1000, alpha=0.5, label='Speckle Noise')
    plt.hist(gauss_eq5, 1000, alpha=0.5, label='Equivalent Gaussian')
    plt.title('Distribution of Specke Noise for 5 views (L=5)')
    plt.legend()

    print('5 views : mean=%.3f | std=%.3f' % (speckle5.noise_sample.mean(), speckle5.noise_sample.std()))

    plt.show()


if __name__ == '__main__':
    # plot_noise_distr()
    nb_view = 1
    # log_validation_plot('results/00047-autoencoder', nb_view, [62], is_log=False, noise='speckle')
    # log_validation_plot('results/00049-autoencoder', nb_view, [62], is_log=False, noise='gaussian')
    # log_validation_plot('results/00053-autoencoder', nb_view, [62], is_log=False, noise='speckle', nb_channel=3)
    log_validation_plot('results/00055-autoencoder', nb_view, [62], is_log=False, noise='speckle', nb_channel=3)
    plt.show()
