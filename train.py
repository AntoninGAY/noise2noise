# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import tensorflow as tf
import numpy as np

import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
import dnnlib.tflib.tfutil as tfutil
import dnnlib.util as util

import config

from util import save_image, save_snapshot
from validation import ValidationSet
from dataset import create_dataset

from scipy import special


class AugmentGaussian:
    """ Class for Gaussian noise

    """
    def __init__(self, validation_stddev, train_stddev_rng_range):
        self.validation_stddev = validation_stddev
        self.train_stddev_range = train_stddev_rng_range

    def add_train_noise_tf(self, x: tf.Tensor):
        """ Adding gaussian noise to training images, in TF format

        :param x:
        :return:
        """
        (minval, maxval) = self.train_stddev_range
        shape = tf.shape(x)
        rng_stddev = tf.random_uniform(shape=[1, 1, 1], minval=minval / 255.0, maxval=maxval / 255.0)
        y = x + tf.random_normal(shape) * rng_stddev
        return y

    def add_validation_noise_np(self, x: np.ndarray):
        """ Adding gaussian noise to the validation images, in NP format

        :param x:
        :return:
        """
        return x + np.random.normal(size=x.shape) * (self.validation_stddev / 255.0)


class AugmentPoisson:
    def __init__(self, lam_max):
        self.lam_max = lam_max

    def add_train_noise_tf(self, x):
        chi_rng = tf.random_uniform(shape=[1, 1, 1], minval=0.001, maxval=self.lam_max)
        return tf.random_poisson(chi_rng * (x + 0.5), shape=[]) / chi_rng - 0.5

    @staticmethod
    def add_validation_noise_np(x):
        chi = 30.0
        return np.random.poisson(chi * (x + 0.5)) / chi - 0.5


class AugmentSpeckle:
    """ This class add speckle noise to an image which is already on its log form

    For a reason of optimization, the validation noise is not computed when the method is called.
    In fact, this method was too long. Thus, a list of additive realizations of the speckle noise is computed and saved
    at the creation of the object, and on method call, a random one of them is added
    """

    def __init__(self, l_nb_views, norm_max=10.089, norm_min=-1.423, quick_noise_computation=False):
        self.L = l_nb_views
        self.norm_max = norm_max
        self.norm_min = norm_min

        # Recommanded if L > 1
        self.quick_noise = quick_noise_computation

        if self.quick_noise:
            self.noise_sample = self.generate_validation_noise_np(shape=(512 * 512))

    def add_train_noise_tf(self, im_log):
        """ Add noise to training dataset.
        Author: Emanuele Dalsasso <emanuele.dalsasso@telecom-paristech.fr>

        PAPER REFERENCE: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1000333
        "Statistical properties of logaritmically transformed speckle"

        :param im_log: log of RGB image of size [3,256,256] TF
        :return:
        """

        # We compute the Speckle noise
        s = tf.zeros(shape=tf.shape(im_log))
        if config.get_nb_channels() == 3:
            s = s[0]

        for k in range(0, self.L):
            gamma = (tf.abs(tf.complex(tf.random_normal(shape=tf.shape(s), stddev=1),
                                       tf.random_normal(shape=tf.shape(s), stddev=1))) ** 2) / 2
            s = s + gamma
        s_amplitude = tf.sqrt(s / self.L)

        # We get the log, remove the bias, and normalize the image
        log_speckle = tf.log(s_amplitude)
        log_speckle = self.add_bias_tf(log_speckle)
        log_speckle_norm = log_speckle / (self.norm_max - self.norm_min)

        # Depending on the number of channels, we add the noise one to three times
        if config.get_nb_channels() == 3:
            log_speckle_norm_3ch = tf.concat([[log_speckle_norm], [log_speckle_norm], [log_speckle_norm]], axis=0)
            im_noisy = im_log + log_speckle_norm_3ch
        else:
            im_noisy = im_log + log_speckle_norm

        return im_noisy

    def generate_validation_noise_np(self, shape):
        """ Generate a array of speckle noise

        :param shape:
        :return:
        """

        # Computes the Speckle noise
        s = np.zeros(shape=shape)
        for k in range(0, self.L):
            gamma = (np.abs(np.random.normal(size=shape, scale=1) +
                            1j * np.random.normal(size=shape, scale=1)) ** 2) / 2
            s = s + gamma
        s_amplitude = np.sqrt(s / self.L)

        # Get the log of the SPeckle noise, remove the bias and normalize it
        log_speckle = np.log(s_amplitude)
        log_speckle = self.add_bias_np(log_speckle)
        log_speckle_norm = log_speckle / (self.norm_max - self.norm_min)

        # We unbiased the image
        return log_speckle_norm

    def add_validation_noise_np(self, im_log: np.ndarray):
        """ Add noise to training dataset.
        Author: Emanuele Dalsasso <emanuele.dalsasso@telecom-paristech.fr>

        PAPER REFERENCE: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1000333
        "Statistical properties of logaritmically transformed speckle"

        :param im_log: log of RGB image of size [3,256,256] NP
        :return:
        """

        # We compute the shape of the noise array to compute
        if config.get_nb_channels() == 3:
            shape = im_log[0].shape
        else:
            shape = im_log.shape

        # If quick noise computation, we pick the noise from pre-computed list
        if self.quick_noise:
            log_speckle_norm = np.random.choice(self.noise_sample, size=shape)
        else:
            # Otherwise, we compute the Speckle noise
            s = np.zeros(shape=shape)
            for k in range(0, self.L):
                gamma = (np.abs(np.random.normal(size=shape, scale=1) +
                                1j * np.random.normal(size=shape, scale=1)) ** 2) / 2
                s = s + gamma
            s_amplitude = np.sqrt(s / self.L)

            # Get the log of the SPeckle noise, remove the bias and normalize it
            log_speckle = np.log(s_amplitude)
            log_speckle = self.add_bias_np(log_speckle)
            log_speckle_norm = log_speckle / (self.norm_max - self.norm_min)

        # If we have 3 channels, we add the same noise to all the channels.
        if config.get_nb_channels() == 3:
            log_speckle_norm_3ch = np.array([log_speckle_norm, log_speckle_norm, log_speckle_norm])
            im_noisy = im_log + log_speckle_norm_3ch
        else:
            im_noisy = im_log + log_speckle_norm

        return im_noisy

    def add_bias_tf(self, x, bias=None):
        """ Adds a bias to an image. If none given, automatically computed

        :param x:
        :param bias:
        """
        if bias is None:
            bias = (1 / 2) * (special.psi(self.L) - tf.log(self.L))
            bias = - bias

        return x + bias

    def add_bias_np(self, x, bias=None):
        """ Adds a bias to an image. If none given, automatically computed

        :param x:
        :param bias:
        """
        if bias is None:
            bias = (1 / 2) * (special.psi(self.L) - np.log(self.L))
            bias = - bias

        return x + bias

    def restore_bias_np(self, x, bias=None):
        """ Adds a bias to an image. If none given, automatically computed

        :param x:
        :param bias:
        """
        if bias is None:
            bias = (1 / 2) * (special.psi(self.L) - np.log(self.L))

        return x + bias


def compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate):
    ramp_down_start_iter = iteration_count * (1 - ramp_down_perc)
    if i >= ramp_down_start_iter:
        t = ((i - ramp_down_start_iter) / ramp_down_perc) / iteration_count
        smooth = (0.5 + np.cos(t * np.pi) / 2) ** 2
        return learning_rate * smooth
    return learning_rate


def train(
        submit_config: dnnlib.SubmitConfig,
        iteration_count: int,
        eval_interval: int,
        minibatch_size: int,
        learning_rate: float,
        ramp_down_perc: float,
        noise: dict,
        validation_config: dict,
        train_tfrecords: str,
        noise2noise: bool):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**validation_config)

    # Create a run context (hides low level details, exposes simple API to manage the run)
    # noinspection PyTypeChecker
    ctx = dnnlib.RunContext(submit_config, config)

    # Initialize TensorFlow graph and session using good default settings
    tfutil.init_tf(config.tf_config)

    dataset_iter = create_dataset(train_tfrecords, minibatch_size, noise_augmenter.add_train_noise_tf)

    # Construct the network using the Network helper class and a function defined in config.net_config
    with tf.device("/gpu:0"):
        net = tflib.Network(**config.net_config)

    # Optionally print layer information
    net.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device("/cpu:0"):
        lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])

        noisy_input, noisy_target, clean_target = dataset_iter.get_next()
        noisy_input_split = tf.split(noisy_input, submit_config.num_gpus)
        noisy_target_split = tf.split(noisy_target, submit_config.num_gpus)
        clean_target_split = tf.split(clean_target, submit_config.num_gpus)

    # Define the loss function using the Optimizer helper class, this will take care of multi GPU
    opt = tflib.Optimizer(learning_rate=lrate_in, **config.optimizer_config)

    for gpu in range(submit_config.num_gpus):
        with tf.device("/gpu:%d" % gpu):
            net_gpu = net if gpu == 0 else net.clone()

            denoised = net_gpu.get_output_for(noisy_input_split[gpu])

            if noise2noise:
                meansq_error = tf.reduce_mean(tf.square(noisy_target_split[gpu] - denoised))
            else:
                meansq_error = tf.reduce_mean(tf.square(clean_target_split[gpu] - denoised))
            # Create an autosummary that will average over all GPUs
            with tf.control_dependencies([autosummary("Loss", meansq_error)]):
                opt.register_gradients(meansq_error, net_gpu.trainables)

    train_step = opt.apply_updates()

    # Create a log file for Tensorboard
    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    summary_log.add_graph(tf.get_default_graph())

    print('Training...')
    time_maintenance = ctx.get_time_since_last_update()
    ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=0, max_epoch=iteration_count)

    # ***********************************
    # The actual training loop
    for i in range(iteration_count):
        # Whether to stop the training or not should be asked from the context
        if ctx.should_stop():
            break

        # Dump training status
        if i % eval_interval == 0:
            time_train = ctx.get_time_since_last_update()
            time_total = ctx.get_time_since_start()

            # Evaluate 'x' to draw a batch of inputs
            [source_mb, target_mb] = tfutil.run([noisy_input, clean_target])
            denoised = net.run(source_mb)
            save_image(submit_config, denoised[0], "img_{0}_y_pred.png".format(i))
            save_image(submit_config, target_mb[0], "img_{0}_y.png".format(i))
            save_image(submit_config, source_mb[0], "img_{0}_x_aug.png".format(i))

            validation_set.evaluate(net, i, noise_augmenter.add_validation_noise_np)

            print('iter %-10d time %-12s eta %-12s sec/eval %-7.1f sec/iter %-7.2f maintenance %-6.1f' % (
                autosummary('Timing/iter', i),
                dnnlib.util.format_time(autosummary('Timing/total_sec', time_total)),
                dnnlib.util.format_time(
                    autosummary('Timing/total_sec', (time_train / eval_interval) * (iteration_count - i))),
                autosummary('Timing/sec_per_eval', time_train),
                autosummary('Timing/sec_per_iter', time_train / eval_interval),
                autosummary('Timing/maintenance_sec', time_maintenance)))

            dnnlib.tflib.autosummary.save_summaries(summary_log, i)
            ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=i, max_epoch=iteration_count)
            time_maintenance = ctx.get_last_update_interval() - time_train

        # Training epoch
        lrate = compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate)
        tfutil.run([train_step], {lrate_in: lrate})

    # End of training
    print("Elapsed time: {0}".format(util.format_time(ctx.get_time_since_start())))
    save_snapshot(submit_config, net, 'final')

    # Summary log and context should be closed at the end
    summary_log.close()
    ctx.close()
