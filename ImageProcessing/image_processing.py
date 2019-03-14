import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt


def normalize(im, vmax=None, vmin=None):
    """ Normalise an image

    :param im: NP image
    :param vmax:
    :param vmin:
    :return im_norm :
    """

    if vmax is None:
        vmax = im.max()
    if vmin is None:
        vmin = im.min()

    return (im - vmin) / (vmax - vmin)


def plot_hist(im1, im2, bins):
    hist_png = np.histogram(im1, bins)
    hist_npy = np.histogram(im2, bins)

    plt.figure()
    plt.subplot(121)
    plt.bar(hist_png[1][0:len(bins) - 1], hist_png[0])

    plt.subplot(122)
    plt.bar(hist_npy[1][0:len(bins) - 1], hist_npy[0])
    plt.show(block=False)


# Loading the images
im_png = np.array(imageio.imread('ImageProcessing/OriginalImages/denoised_lely.png'))
im_npy = np.load('ImageProcessing/OriginalImages/denoised_lely.npy')

# Computing histograms
bins_im = np.linspace(0, 255, 2 * 256)

plot_hist(im_png, im_npy, bins_im)

# Log images
log_png = np.log(im_png)
log_npy = np.log(im_npy)

# Normalize images
max_log_png = log_png.max()
min_log_png = log_png.min()

# log_png = normalize(log_png, max_log_png, min_log_png) * 255.0
log_npy = normalize(log_npy, max_log_png, min_log_png) * 255.0

# Saving the images
cv2.imwrite('ImageProcessing/ProcessedImages/log_lely.png', log_npy)
np.save('ImageProcessing/ProcessedImages/log_lely.npy', log_npy)

log_png = np.array(imageio.imread('ImageProcessing/ProcessedImages/log_lely.png'))
log_npy = np.load('ImageProcessing/ProcessedImages/log_lely.npy')

# Histograms
bins_log = np.linspace(0, 255, 255 * 4 + 1)

plot_hist(log_png, log_npy, bins_log)

# Samples
samp_png = log_png[0:256, 0:256]
samp_npy = log_npy[0:256, 0:256]

print("Sample shape : ")
print(samp_npy.shape)

plot_hist(samp_png, samp_npy, bins_log)

plt.show()
