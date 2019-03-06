#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAPER REFERENCE: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1000333
"Statistical properties of logaritmically transformed speckle"

Author: Emmanuele Dalsasso
"""

import numpy as np


def injectspeckle_amplitude(img, L):
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    s = np.zeros((rows, columns))
    for _ in range(0, L):
        gamma = np.abs(np.random.randn(rows, columns) + np.random.randn(rows, columns) * 1j) ** 2 / 2
        s = s + gamma
    s_amplitude = np.sqrt(s / L)
    ima_speckle_amplitude = np.multiply(img, s_amplitude)
    return ima_speckle_amplitude


data = np.load('clean_images/denoised_marais1.npy')
noisy_data = injectspeckle_amplitude(data, 1)
residual_noise_intensity = (noisy_data / data) ** 2
print("Var=%f Mean=%f" % (np.var(residual_noise_intensity), np.mean(residual_noise_intensity)))
# Mean must be equal to 1 and Var 1/L
