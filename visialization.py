from train import AugmentSpeckle
import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('Examples/denoised_lely10.png')
im = im.mean(axis=2)
im = im /255.0

augmenters = [AugmentSpeckle(l=n, normalize=True) for n in range(1, 6)]

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(im, cmap='gray')
plt.title('Original Image')

for idx, augm in enumerate(augmenters):
    im_noise = augm.add_validation_noise_np(im)

    plt.subplot(2, 3, idx + 2)
    plt.imshow(im_noise, cmap='gray')
    plt.title('Nb views = %d' % (idx+1))

plt.show()
