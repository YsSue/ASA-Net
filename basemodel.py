#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: basemodel.py

import tensorflow as tf
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.models import (
    Conv2D, MaxPooling,FixedUnPooling, BatchNorm, BNReLU,layer_register)
import numpy as np

def image_preprocess(image, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        image = image * (1.0 / 255)

        mean = [0.485, 0.456, 0.406]    # rgb
        std = [0.229, 0.224, 0.225]
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_std
        return image


def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name: BatchNorm('bn', x)

@layer_register(log_shape=True)
def GroupNorm(x, group=32, gamma_initializer=tf.constant_initializer(1.)):
    """
    More code that reproduces the paper can be found at https://github.com/ppwwyyxx/GroupNorm-reproduce/.
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')

def resnet_shortcut(l, n_out, stride, nl=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        if stride == 2:
            l = l[:, :, :-1, :-1]
            return Conv2D('convshortcut', l, n_out, 1,
                          stride=stride, padding='VALID', nl=nl)
        else:
            return Conv2D('convshortcut', l, n_out, 1,
                          stride=stride, nl=nl)
    else:
        return l


def resnet_bottleneck(l, ch_out, stride):
    l, shortcut = l, l
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    if stride == 2:
        l = tf.pad(l, [[0, 0], [0, 0], [0, 1], [0, 1]])
        l = Conv2D('conv2', l, ch_out, 3, stride=2, nl=BNReLU, padding='VALID')
    else:
        l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))


def resnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features,
                               stride if i == 0 else 1)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l

@auto_reuse_variable_scope
def pretrained_resnet_conv4(image, num_blocks):
    assert len(num_blocks) == 3
    with argscope([Conv2D, MaxPooling, BatchNorm], data_format='NCHW'), \
            argscope(Conv2D, nl=tf.identity, use_bias=False), \
            argscope(BatchNorm, use_local_stat=False):
        l = tf.pad(image, [[0, 0], [0, 0], [2, 3], [2, 3]])
        b_2 = Conv2D('conv0', l, 64, 7, stride=2, nl=BNReLU, padding='VALID')
        l = tf.pad(b_2, [[0, 0], [0, 0], [0, 1], [0, 1]])
        l = MaxPooling('pool0', l, shape=3, stride=2, padding='VALID')
        b_4 = resnet_group(l, 'group0', resnet_bottleneck, 64, num_blocks[0], 1)
        # TODO replace var by const to enable folding
        b_4 = tf.stop_gradient(b_4)
        b_8 = resnet_group(b_4, 'group1', resnet_bottleneck, 128, num_blocks[1], 2)
        b_16 = resnet_group(b_8, 'group2', resnet_bottleneck, 256, num_blocks[2], 2)
    # 16x downsampling up to now
    return b_2,b_4,b_8,b_16


@auto_reuse_variable_scope
def resnet_conv5(image, num_block):
    with argscope([Conv2D, BatchNorm], data_format='NCHW'), \
            argscope(Conv2D, nl=tf.identity, use_bias=False), \
            argscope(BatchNorm, use_local_stat=False):
        # 14x14:
        l = resnet_group(image, 'group3', resnet_bottleneck, 512, num_block, stride=2)
        return l

@layer_register(log_shape=True)
def fpn_model(features,num_channel):
    """
    Args:
        features ([tf.Tensor]): ResNet features c2-c5
    Returns:
        [tf.Tensor]: FPN features p2-p6
    """
    assert len(features) == 4, features

    def upsample2x(name, x):
        try:
            resize = tf.compat.v2.image.resize_images
            with tf.name_scope(name):
                shp2d = tf.shape(x)[2:]
                x = tf.transpose(x, [0, 2, 3, 1])
                x = resize(x, shp2d * 2, 'nearest')
                x = tf.transpose(x, [0, 3, 1, 2])
                return x
        except AttributeError:
            return FixedUnPooling(
                name, x, 2, unpool_mat=np.ones((2, 2), dtype='float32'),
                data_format='channels_first')

    with argscope(Conv2D, data_format='channels_first',
                  activation=tf.identity, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(scale=1.)):
        lat_2345 = [Conv2D('lateral_1x1_c{}'.format(i + 2), c, num_channel, 1)
                    for i, c in enumerate(features)]
        lat_2345 = [GroupNorm('gn_c{}'.format(i + 2), c) for i, c in enumerate(lat_2345)]
        lat_sum_5432 = []
        for idx, lat in enumerate(lat_2345[::-1]):
            if idx == 0:
                lat_sum_5432.append(lat)
            else:
                lat = lat + upsample2x('upsample_lat{}'.format(6 - idx), lat_sum_5432[-1])
                lat_sum_5432.append(lat)
        p2345 = [Conv2D('posthoc_3x3_p{}'.format(i + 2), c, num_channel, 3)
                 for i, c in enumerate(lat_sum_5432[::-1])]
        p6 = MaxPooling('maxpool_p6', p2345[-1], pool_size=1, strides=2, data_format='channels_first', padding='VALID')
        return p2345 + [p6]