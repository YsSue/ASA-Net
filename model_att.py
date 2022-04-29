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
def attribute_head(feature, dim_att):
    """
    Args:
        feature (NxCx7x7):
        dim_att(int): dim_att
    Returns:
        att_logits (Nxdim_att)
    """
    feature = GlobalAvgPooling('gap', feature, data_format='NCHW')
    attribute = FullyConnected(
        'attribute', feature, dim_att,
        W_init=tf.random_normal_initializer(stddev=0.01))
    return attribute


@under_name_scope()
def attribute_loss(labels, label_logits):
    """
    Args:
        labels: #fgxNUM_ATT, int
        label_logits: #fg x NUM_ATT
    """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=label_logits)
    attribute_label_loss = tf.reduce_mean(loss, name='attribute_loss')
    attribute_probs = tf.sigmoid(label_logits)

    pred_label = attribute_probs > 0.5
    truth_label = labels > 0.5
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(pred_label, truth_label)),name='att_accuracy')
    add_moving_summary(attribute_label_loss, accuracy)

    return attribute_label_loss