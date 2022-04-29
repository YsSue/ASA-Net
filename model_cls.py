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


@layer_register(log_shape=True)
def category_head(feature, num_classes):
    """
    Args:
        feature (NxCx7x7):
        num_classes(int): num_category
    Returns:
        cls_logits (Nxnum_category)
    """
    feature = GlobalAvgPooling('gap', feature, data_format='NCHW')
    classification = FullyConnected(
        'class', feature, num_classes,
        W_init=tf.random_normal_initializer(stddev=0.01))
    return classification

@under_name_scope()
def category_loss(labels, label_logits,label_smoothing=None):
    """
    Args:
        labels: n,
        label_logits: nxC
    """
    if label_smoothing==None:
        label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=label_logits)

    #label smoothing
    else:
        onehot_labels=tf.one_hot(labels,config.NUM_CLASS)
        label_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=label_logits,weights=1.0,label_smoothing=0.1)

    label_loss = tf.reduce_mean(label_loss, name='label_loss')
    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
        correct = tf.to_float(tf.equal(prediction, labels))  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name='accuracy')
    add_moving_summary(label_loss, accuracy)
    return label_loss

