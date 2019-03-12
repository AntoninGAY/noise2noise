"""
Author: Emanuele Dalsasso <emanuele.dalsasso@telecom-paristech.fr>
"""

import tensorflow as tf


def additiveSpeckleNoiseTF(im_log, L, norm_max=1., norm_min=0.):
    s = tf.zeros(shape=tf.shape(im_log))
    for k in range(0, L):  
        gamma = (tf.abs(tf.complex(tf.random_normal(shape=tf.shape(im_log), stddev=1),
                                   tf.random_normal(shape=tf.shape(im_log), stddev=1))) ** 2) / 2
        s = s + gamma
    s_amplitude = tf.sqrt(s / L)
    log_speckle = tf.log(s_amplitude)

    # comment this line if you don't normalize the images
    log_norm_speckle = log_speckle / (norm_max - norm_min)

    X = im_log + log_norm_speckle
    return X
