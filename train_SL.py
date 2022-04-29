#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import os
import argparse
import cv2

import numpy as np
import tensorflow as tf

from tensorpack import *
from tensorpack.callbacks import Callback
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
import tensorpack.utils.viz as tpviz
from tensorpack.utils.gpu import get_nr_gpu


from basemodel import (
    image_preprocess, pretrained_resnet_conv4, resnet_conv5)
from model_cls import (
    category_head, category_loss)
from model_att import (
    attribute_head, attribute_loss)
from model_seg import (
    seg_head,seg_head_with_category,
    seg_loss,seg_loss_with_category)
from data import (
    get_train_dataflow, get_eval_dataflow)
from common import print_config
from eval import (
    eval_on_dataflow, detect_one_image, DetectionResult,test_on_dataflow)
import config

def get_batch_factor():
    nr_gpu = get_nr_gpu()
    assert nr_gpu in [1, 2, 4, 8], nr_gpu
    return 8 // nr_gpu


def get_model_output_names():
    ret = ['final_labels', 'final_attribute','final_masks']
    return ret

class Model(ModelDesc):
    def inputs(self):
        ret = [
            tf.TensorSpec([None,config.INPUT_SIZE,config.INPUT_SIZE,3], tf.float32, 'image'),
            tf.TensorSpec([None,None],tf.int64, 'gt_labels'),  # all > 0
            tf.TensorSpec([None,None, config.NUM_ATT],tf.float32,'gt_attribute'),   # NR_GT x NUM_ATT
            tf.TensorSpec([None,None, config.INPUT_SIZE, config.INPUT_SIZE],tf.uint8, 'gt_masks')]# NR_GT x height x width
        return ret

    def preprocess(self, image):
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def build_graph(self, image, gt_labels, gt_attribute, gt_masks):
        gt_labels = tf.squeeze(gt_labels, 1)
        gt_masks = tf.squeeze(gt_masks, 1)
        gt_attribute = tf.squeeze(gt_attribute, 1)

        image = self.preprocess(image)     # 1CHW
        image_shape2d = tf.shape(image)[2:]

        c1,c2,c3,c4=pretrained_resnet_conv4(image, config.RESNET_NUM_BLOCKS[:3])
        c5=resnet_conv5(c4, config.RESNET_NUM_BLOCKS[-1])    # nxcx7x7

        stride=32 # for conv5 7x7x1->224x224x1

        category_label_logits = category_head('category', c5, config.NUM_CLASS)
        attribute_label_logits = attribute_head('attribute', c5,config.NUM_ATT) # nx1xNUM_ATT

        if config.MODE_SEMANTIC:
            mask_logits = seg_head_with_category('segmentation', c5,stride,config.NUM_CLASS)# n x C x 224x224
        else:
            mask_logits = seg_head('segmentation', c5,stride)   # n x 1 x 224x224

        with tf.name_scope('fg_sample_patch_viz'):
            fg_sampled_patches = image
            fg_sampled_patches = tf.transpose(fg_sampled_patches, [0, 2, 3, 1])
            fg_sampled_patches = tf.reverse(fg_sampled_patches, axis=[-1])  # BGR->RGB
            tf.summary.image('viz', fg_sampled_patches, max_outputs=32)

        # category loss
        category_label_loss = category_loss(gt_labels, category_label_logits,config.LABEL_SMOOTH)
        # mask loss
        target_masks=tf.cast(gt_masks,dtype=tf.float32)
        if config.MODE_SEMANTIC:
            mask_loss = seg_loss_with_category(mask_logits, target_masks,gt_labels)
        else:
            mask_loss = seg_loss(mask_logits, target_masks)

        # attribute loss
        attribute_label_loss = attribute_loss(gt_attribute, attribute_label_logits)

        if self.training:
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(config.WEIGHT_DECAY), name='wd_cost')
            cost_list=[]
            cost_list.append(wd_cost)
            if config.MODE_CLS:
                cost_list.append(config.MODE_CLS*category_label_loss)
            if config.MODE_SEG:
                cost_list.append(config.MODE_SEG*mask_loss)
            if config.MODE_ATT:
                cost_list.append(config.MODE_ATT*attribute_label_loss)
            total_cost = tf.add_n(cost_list, 'total_cost')
            add_moving_summary(total_cost, wd_cost)
            return total_cost

        else:
            final_labels = tf.argmax(category_label_logits, axis=1, name='final_labels')
            final_attribute=tf.sigmoid(attribute_label_logits, name='final_attribute')
            if config.MODE_SEMANTIC:
                num_fg = tf.size(final_labels)
                indices = tf.stack([tf.range(num_fg), tf.to_int32(final_labels)], axis=1)  # #Nx2
                mask_logits = tf.gather_nd(mask_logits, indices)  # Nx224x224
                final_masks = tf.sigmoid(mask_logits,name='final_masks')
            else:
                final_masks = tf.sigmoid(tf.squeeze(mask_logits, 1),name='final_masks')   # nx224x224
            return []

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate', lr)

        factor = get_batch_factor()
        if factor != 1:
            lr = lr / float(factor)
            opt = tf.train.MomentumOptimizer(lr, 0.9)
            opt = optimizer.AccumGradOptimizer(opt, factor)
        else:
            opt = tf.train.MomentumOptimizer(lr, 0.9)

        return opt

def offline_evaluate(pred_func, output_file):
    os.mkdir(output_file) 
    df = get_eval_dataflow()
    all_results,all_GT,scores = test_on_dataflow(
        df, lambda img: detect_one_image(img, pred_func),output_file)

def predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    img =cv2.resize(img, (config.INPUT_SIZE,config.INPUT_SIZE))
    img = np.expand_dims(img, 0)
    results = detect_one_image(img, pred_func)
    for r in results:
        mask=cv2.resize(r.mask, (config.INPUT_SIZE,config.INPUT_SIZE)) 
        mask=(mask*255.0)
        cv2.imwrite('result.jpg',mask)

class EvalCallback(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'],
            get_model_output_names())
        self.df = get_eval_dataflow()

    def _before_train(self):
        interval = self.trainer.max_epoch // (config.EVAL_TIMES + 1)
        self.epochs_to_eval = set([interval * k for k in range(1,config.EVAL_TIMES)])
        self.epochs_to_eval.add(self.trainer.max_epoch)

    def _eval(self):
        scores = eval_on_dataflow(self.df, lambda img: detect_one_image(img, self.pred))
        for k, v in scores.items():
            self.trainer.monitors.put_scalar(k, v)

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            logger.info("Running evaluation ...")
            self._eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logdir', help='logdir', default='train_log/ST-ATT')
    parser.add_argument('--datadir', help='override config.BASEDIR')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evaluate', help='path to the output json eval file')
    parser.add_argument('--predict', help='path to the input image file')
    args = parser.parse_args()

    if args.datadir:
        config.BASEDIR = args.datadir

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.visualize or args.evaluate or args.predict:
        # autotune is too slow for inference
        # os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

        assert args.load
        print_config()
        if args.visualize:
            visualize(args.load)
        else:
            pred = OfflinePredictor(PredictConfig(
                model=Model(),
                session_init=get_model_loader(args.load),
                input_names=['image'],
                output_names=get_model_output_names()))
            if args.evaluate:
                offline_evaluate(pred, args.evaluate)
            elif args.predict:
                predict(pred, args.predict)
    else:
        logger.set_logger_dir(args.logdir)
        train_ds=get_train_dataflow()
        print_config()

        # Compute the training schedule from the number of GPUs ...
        factor = get_batch_factor()
        stepnum = config.STEPS_PER_EPOCH
        # warmup is step based, lr is epoch based
        init_lr = config.WARMUP_INIT_LR * min(factor, 1.)
        warmup_schedule = [(0, init_lr), (config.WARMUP, config.BASE_LR)]
        warmup_end_epoch = config.WARMUP * 1. / stepnum
        lr_schedule = [(int(warmup_end_epoch + 0.5), config.BASE_LR)]

        for idx, steps in enumerate(config.LR_SCHEDULE[:-1]):
            mult = 0.1 ** (idx + 1)
            lr_schedule.append(
                (steps * factor // stepnum, config.BASE_LR * mult))
        logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
        logger.info("LR Schedule (epochs, value): " + str(lr_schedule))

        # This is what's commonly referred to as "epochs"
        total_passes = config.LR_SCHEDULE[-1] * 8 * config.BATCH_SIZE_ONE_GPU/config.TRAIN_NUM
        logger.info("Total passes of the training set is: {:.5g}".format(total_passes))

        cfg = TrainConfig(
            model=Model(),
            data=QueueInput(get_train_dataflow()),
            callbacks=[
                ModelSaver(max_to_keep=1),
                # step decay
                # linear warmup
                ScheduledHyperParamSetter(
                    'learning_rate', warmup_schedule, interp='linear', step_based=True),
                ScheduledHyperParamSetter('learning_rate', lr_schedule),
                EstimatedTimeLeft(median=True),
                #EvalCallback(),
                GPUUtilizationTracker(),
            ],
            steps_per_epoch=stepnum,
            max_epoch=config.LR_SCHEDULE[-1] * factor // stepnum,
            session_init=get_model_loader(args.load) if args.load else None,
            #session_init=SaverRestore(args.logdir+'/checkpoint'),
        )
        trainer = SyncMultiGPUTrainerReplicated(get_nr_gpu(), average=False)
        launch_train_with_config(cfg, trainer)
