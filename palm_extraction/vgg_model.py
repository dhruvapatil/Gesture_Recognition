
from collections import namedtuple

import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams',
                     'batch_size, image_shape, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class VGG(object):
    """ResNet model."""

    def __init__(self, hps, mode):
        """ResNet constructor.
        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self._images = tf.placeholder(tf.float32, [None, hps.image_shape[0],hps.image_shape[1], hps.image_shape[2]])
        self.labels = tf.placeholder(tf.float32, [hps.batch_size, int(round(hps.image_shape[0]/8.0)),int(round(hps.image_shape[1]/8.0)), hps.num_classes])
        self.mode = mode

        self._extra_train_ops = []

    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.merge_all_summaries()

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        """Build the core model within the graph."""
        with tf.variable_scope('layer1'):
            x = self._images
            print x
            x = self._conv('conv1_1', x, 3, self.hps.image_shape[2], 64, self._stride_arr(1))
            x = self._conv('conv1_2', x, 3, 64, 64, self._stride_arr(1))
            x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=self._stride_arr(2), padding='SAME')

        with tf.variable_scope('layer2'):
            x = self._conv('conv2_1', x, 3, 64, 128, self._stride_arr(1))
            x = self._conv('conv2_2', x, 3, 128, 128, self._stride_arr(1))
            x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=self._stride_arr(2), padding='SAME')

        with tf.variable_scope('layer3'):

            x = self._conv('conv3_1', x, 3, 128, 256, self._stride_arr(1))
            x = self._conv('conv3_2', x, 3, 256, 256, self._stride_arr(1))
            x = self._conv('conv3_3', x, 3, 256, 256, self._stride_arr(1))
            x = self._conv('conv3_4', x, 3, 256, 256, self._stride_arr(1))
            x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=self._stride_arr(2), padding='SAME')

        with tf.variable_scope('layer4'):

            x = self._conv('conv4_1', x, 3, 256, 512, self._stride_arr(1))
            x = self._conv('conv4_2', x, 3, 512, 512, self._stride_arr(1))
            x = self._conv('conv4_3', x, 3, 512, 2, self._stride_arr(1), relu=False)
            print x

            self.predictions = x

        with tf.variable_scope('costs'):
            xent = tf.nn.l2_loss(x-self.labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

            tf.scalar_summary('cost', self.cost)

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.scalar_summary('learning rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
	elif self.hps.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lrn_rate)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                tf.histogram_summary(mean.op.name, mean)
                tf.histogram_summary(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.histogram_summary(var.op.name, var)

        return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, relu=True):
        #print "*",[filter_size, filter_size, in_filters, out_filters]
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            if relu:
                return tf.nn.relu(tf.nn.conv2d(x, kernel, strides, padding='SAME'))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
