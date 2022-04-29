#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: model.py

import tensorflow as tf
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.models import (
    Conv2D, FullyConnected, GlobalAvgPooling, layer_register, Deconv2D,BNReLU)
import config
import numpy as np
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

@layer_register(log_shape=True)
def CaffeBilinearUpSample(x, shape):
    """
    Deterministic bilinearly-upsample the input images.
    It is implemented by deconvolution with "BilinearFiller" in Caffe.
    It is aimed to mimic caffe behavior.
    Args:
        x (tf.Tensor): a NHWC tensor
        shape (int): the upsample factor
    Returns:
        tf.Tensor: a NHWC tensor.
    """
    inp_shape = x.shape.as_list()
    ch = inp_shape[3]
    assert ch is not None

    shape = int(shape)
    filter_shape = 2 * shape

    def bilinear_conv_filler(s):
        """
        s: width, height of the conv filter
        https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/filler.hpp#L219-L268
        """
        f = np.ceil(float(s) / 2)
        c = float(2 * f - 1 - f % 2) / (2 * f)
        ret = np.zeros((s, s), dtype='float32')
        for x in range(s):
            for y in range(s):
                ret[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return ret
    w = bilinear_conv_filler(filter_shape)
    w = np.repeat(w, ch * ch).reshape((filter_shape, filter_shape, ch, ch))

    weight_var = tf.constant(w, tf.float32,
                             shape=(filter_shape, filter_shape, ch, ch),
                             name='bilinear_upsample_filter')
    x = tf.pad(x, [[0, 0], [shape - 1, shape - 1], [shape - 1, shape - 1], [0, 0]], mode='SYMMETRIC')
    out_shape = tf.shape(x) * tf.constant([1, shape, shape, 1], tf.int32)
    deconv = tf.nn.conv2d_transpose(x, weight_var, out_shape,
                                    [1, shape, shape, 1], 'SAME')
    edge = shape * (shape - 1)
    deconv = deconv[:, edge:-edge, edge:-edge, :]

    if inp_shape[1]:
        inp_shape[1] *= shape
    if inp_shape[2]:
        inp_shape[2] *= shape
    deconv.set_shape(inp_shape)
    return deconv

@layer_register(log_shape=True)
def seg_head(featuremap,stride):
    """
    Args:
        featuremap (NxCxHxW):
        stride(int): 224 / H
        attribute_feature (NxNUM_ATT)
        num_att(int): num_att
    Returns:
        mask_logits (N x C  x 224 x 224):
    """
    num_output=1

    with argscope([Conv2D, Deconv2D], data_format='NCHW',
                  W_init=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_in', distribution='normal')):
        #l= Conv2D('conv1', featuremap, config.FPN_NUM_CHANNEL, 3, activation=tf.nn.relu)
        #l= Conv2D('conv2', l, config.FPN_NUM_CHANNEL, 3, activation=tf.nn.relu)
        l= Conv2D('seg', featuremap, num_output, 1)
    l = tf.transpose(l, [0, 2, 3, 1])
    while stride!= 1:
        l = CaffeBilinearUpSample('upsample{}'.format(stride), l, 2) #input/output NHWC tensor
        stride= stride/ 2
    l = tf.transpose(l, [0, 3, 1, 2])
    return l

@layer_register(log_shape=True)
def seg_head_with_category(featuremap,stride,num_cls):
    """
    Args:
        featuremap (NxCxHxW):
        stride(int): 224 / H
        attribute_feature (NxNUM_ATT)
        num_att(int): num_att
    Returns:
        mask_logits (N x C  x 224 x 224):
    """
    num_output=num_cls

    with argscope([Conv2D, Deconv2D], data_format='NCHW',
                  W_init=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_in', distribution='normal')):
        l= Conv2D('seg', featuremap, num_output, 1)
    l = tf.transpose(l, [0, 2, 3, 1])
    while stride!= 1:
        l = CaffeBilinearUpSample('upsample{}'.format(stride), l, 2) #input/output NHWC tensor
        stride= stride/ 2
    l = tf.transpose(l, [0, 3, 1, 2])
    return l


@under_name_scope()
def seg_loss(mask_logits, fg_target_masks):
    """
    Args:
        mask_logits: N x1x224 x224 
        fg_target_masks: Nx224 x224,
    """
    mask_logits = tf.squeeze(mask_logits, 1)  # Nx224x224
    mask_probs = tf.sigmoid(mask_logits)

    pred_label = mask_probs > 0.5
    truth_label = fg_target_masks > 0.5

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fg_target_masks, logits=mask_logits)
    loss = tf.reduce_mean(loss, name='maskrcnn_loss')

    accuracy = tf.reduce_mean(
        tf.to_float(tf.equal(pred_label, truth_label)),
        name='accuracy')
    pos_accuracy = tf.logical_and(
        tf.equal(pred_label, truth_label),
        tf.equal(truth_label, True))
    pos_accuracy = tf.reduce_mean(tf.to_float(pos_accuracy), name='pos_accuracy')
    fg_pixel_ratio = tf.reduce_mean(tf.to_float(truth_label), name='fg_pixel_ratio')

    # add some training visualizations to tensorboard
    with tf.name_scope('mask_viz'):
        viz = tf.concat([fg_target_masks, mask_probs], axis=1)
        viz = tf.expand_dims(viz, 3)
        viz = tf.cast(viz * 255, tf.uint8, name='viz')
        tf.summary.image('mask_truth|pred', viz, max_outputs=10)

    add_moving_summary(loss, accuracy, fg_pixel_ratio, pos_accuracy)
    return loss

@under_name_scope()
def seg_loss_with_category(mask_logits, fg_target_masks,gt_labels):
    """
    Args:
        mask_logits: N xCx224 x224 
        fg_target_masks: Nx224 x224,
        gt_labels: N,

    """
    num_fg = tf.size(gt_labels)
    indices = tf.stack([tf.range(num_fg), tf.to_int32(gt_labels)], axis=1)  # #Nx2
    mask_logits = tf.gather_nd(mask_logits, indices)  # Nx224x224
    mask_probs = tf.sigmoid(mask_logits)

    pred_label = mask_probs > 0.5
    truth_label = fg_target_masks > 0.5

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fg_target_masks, logits=mask_logits)
    loss = tf.reduce_mean(loss, name='maskrcnn_loss')

    accuracy = tf.reduce_mean(
        tf.to_float(tf.equal(pred_label, truth_label)),
        name='accuracy')
    pos_accuracy = tf.logical_and(
        tf.equal(pred_label, truth_label),
        tf.equal(truth_label, True))
    pos_accuracy = tf.reduce_mean(tf.to_float(pos_accuracy), name='pos_accuracy')
    fg_pixel_ratio = tf.reduce_mean(tf.to_float(truth_label), name='fg_pixel_ratio')

    # add some training visualizations to tensorboard
    with tf.name_scope('mask_viz'):
        viz = tf.concat([fg_target_masks, mask_probs], axis=1)
        viz = tf.expand_dims(viz, 3)
        viz = tf.cast(viz * 255, tf.uint8, name='viz')
        tf.summary.image('mask_truth|pred', viz, max_outputs=10)

    add_moving_summary(loss, accuracy, fg_pixel_ratio, pos_accuracy)
    return loss

