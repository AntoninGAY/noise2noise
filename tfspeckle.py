"""
Author: Emanuele Dalsasso <emanuele.dalsasso@telecom-paristech.fr>
"""

import tensorflow as tf


def additive_speckle_noise_tf(im_log, l, norm_max=1., norm_min=0.):
    s = tf.zeros(shape=tf.shape(im_log))
    for k in range(0, l):
        gamma = (tf.abs(tf.complex(tf.random_normal(shape=tf.shape(im_log), stddev=1),
                                   tf.random_normal(shape=tf.shape(im_log), stddev=1))) ** 2) / 2
        s = s + gamma
    s_amplitude = tf.sqrt(s / l)
    log_speckle = tf.log(s_amplitude)

    # comment this line if you don't normalize the images
    log_norm_speckle = log_speckle / (norm_max - norm_min)

    x = im_log + log_norm_speckle
    return x
