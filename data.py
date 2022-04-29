#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: data.py

import cv2
import numpy as np
import copy
from tensorpack.utils.argtools import memoized, log_once
from tensorpack.dataflow import (
    imgaug, TestDataSpeed, PrefetchDataZMQ, MultiProcessMapData,
    MapDataComponent, DataFromList,BatchData)
# import tensorpack.utils.viz as tpviz

from CUB_ZSL import CUBDetection as ObjDetection
from common import (
    DataFromListOfDict, CustomResize,
    box_to_point8, point8_to_box)
import config
class MalformedData(BaseException):
    pass


def get_train_dataflow():
    """
    Return a training dataflow. Each datapoint is:
    image, fm_labels, fm_boxes, gt_boxes, gt_class , masks
    """
    imgs = ObjDetection.load_many(config.BASEDIR, config.TRAIN_DATASET,is_train=True)
    # Valid training images should have at least one fg box.
    # But this filter shall not be applied for testing.
    imgs = list(filter(lambda img: len(img['boxes']) > 0, imgs))    # log invalid training
    config.TRAIN_NUM=len(imgs)

    aug = imgaug.AugmentorList(
        [imgaug.Rotation(max_deg=30),
         #imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.8, 1.0), target_shape=config.INPUT_SIZE),
         imgaug.Resize((config.INPUT_SIZE,config.INPUT_SIZE)),
         imgaug.Flip(horiz=True)])


    ds = DataFromList(imgs, shuffle=True)

    def preprocess(img):
        im, fname, boxes, klass, seg,attribute = img['image'], img['file_name'], img['boxes'], img['class'], img['seg'], img['attribute']

        if config.BBOX:
            # assume floatbox as input
            assert boxes.dtype == np.float32
            x=int(boxes[0][0])
            y=int(boxes[0][1])
            x2=int(boxes[0][2])
            y2=int(boxes[0][3])
            im = im[y:y2,x:x2]
        im = cv2.resize(im, (config.TRAIN_IMAGE_SIZE,config.TRAIN_IMAGE_SIZE)) 
        im = im.astype('float32')

        # augmentation:
        tfm=aug.get_transform(im)
        im=tfm.apply_image(im)

        # one image-sized binary mask per box
        if config.BBOX:
            seg = seg[y:y2,x:x2]
        seg = cv2.resize(seg, (config.TRAIN_IMAGE_SIZE,config.TRAIN_IMAGE_SIZE))   #cv2.resize(src,(width,height))
        seg=tfm.apply_image(seg)
        r,segmentation=cv2.threshold(seg,127,1,cv2.THRESH_BINARY) 
        #r,att=cv2.threshold(attribute,0.5,1,cv2.THRESH_BINARY) 
        masks = [] 
        masks.append(segmentation)       
        masks = np.asarray(masks, dtype='uint8')    # values in {0, 1}
        ret = [im, klass,attribute,masks]
        return ret

    ds = MultiProcessMapData(ds, 3, preprocess)
    ds = BatchData(ds, config.BATCH_SIZE_ONE_GPU)
    return ds

def get_eval_dataflow():
    imgs = ObjDetection.load_many(config.BASEDIR, config.VAL_DATASET,is_train=False)
    config.VAL_NUM=len(imgs)
    ds = DataFromList(imgs, shuffle=True)
    def preprocess(img):
        id, im, fname, boxes, klass, seg,attribute = img['id'],img['image'], img['file_name'], img['boxes'], img['class'], img['seg'], img['attribute']
        if config.BBOX:
            # assume floatbox as input
            assert boxes.dtype == np.float32
            x=int(boxes[0][0])
            y=int(boxes[0][1])
            x2=int(boxes[0][2])
            y2=int(boxes[0][3])
            im = im[y:y2,x:x2]
        im = cv2.resize(im, (config.INPUT_SIZE,config.INPUT_SIZE)) 
        im = im.astype('float32')
        ims=[]
        ims.append(im)
        ims=np.asarray(ims)

        # one image-sized binary mask per box
        if config.BBOX:
            seg = seg[y:y2,x:x2]
        r,seg=cv2.threshold(seg,127,1,cv2.THRESH_BINARY) 
        masks = [] 
        masks.append(seg)       
        masks = np.asarray(masks, dtype='uint8')    # values in {0, 1}
        ret = [id, fname,ims, klass,attribute,masks,boxes]

        return ret

    ds = MultiProcessMapData(ds, 3, preprocess)
    ds = PrefetchDataZMQ(ds, 1)
    #ds = BatchData(ds,1)
    return ds

if __name__ == '__main__':
    config.BASEDIR = 'dataset/CUB'
    config.VAL_DATASET = ['zsl_test_1']
    from tensorpack.dataflow import PrintData
    ds = get_train_dataflow()
    ds = PrintData(ds, 5)
    TestDataSpeed(ds, 10).start()
    ds.reset_state()
    for k in ds.get_data():
        print (k[0])
        import pdb
        pdb.set_trace()