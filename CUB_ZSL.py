#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: CUB_ZSL.py

import numpy as np
import os
import scipy.io
import cv2
from termcolor import colored
from tabulate import tabulate
import json
import operator
from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation
from tensorpack.utils.argtools import log_once
import config

__all__ = ['CUBDetection', 'CUBMeta']

config.NUM_CATEGORY_TEST = 50
config.NUM_CATEGORY_TRAIN = 150
config.NUM_ATT = 312

def load_file(examples_list_file):
    lines = np.genfromtxt(examples_list_file, dtype='i8')
    examples = []
    for example in lines:
        examples.append(example)
    #print np.asarray(examples), len(lines)
    return np.asarray(examples), len(lines)

def knn(trainData, testData, labels, k):
    trainData=np.array(trainData)
    rowSize = trainData.shape[0]

    #consine distance
    distances=[]
    vec1=testData
    for i in range(rowSize):
        vec2=trainData[i]
        diff= np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        distances.append(1.0/(diff+0.000001))
    '''
    #Euclidean distance
    diff = np.tile(testData,(rowSize, 1))-trainData
    sqrDiff = diff ** 2
    sqrDiffSum = sqrDiff.sum(axis=1)
    distances = sqrDiffSum ** 0.5

    #Manhattan distance
    diff = np.tile(testData,(rowSize, 1))-trainData
    absDiff = np.abs(diff)
    distances=absDiff.sum(axis=1)
    '''

    distances=np.array(distances)
    sortDistance = distances.argsort()
    count = {}
    for i in range(k):
        vote = labels[sortDistance[i]]
        count[vote] = count.get(vote,0) + 1
    sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)

    return sortCount[0][0]

class _CUBMeta(object):
    INSTANCE_TO_BASEDIR = {
        'zsl_train_1': 'train',
        'zsl_train_2': 'train',
        'zsl_train_3': 'train',
        'zsl_train_4': 'train',
        'zsl_train_5': 'train',
        'zsl_test_1': 'val',
        'zsl_test_2': 'val',
        'zsl_test_3': 'val',
        'zsl_test_4': 'val',
        'zsl_test_5': 'val'
    }

    def valid(self):
        return hasattr(self, 'cat_ids')

    def create(self, cat_ids, att_data):
        """
        cat_ids: list of ids
        att_data: list of attributes
        """
        self.cat_ids = cat_ids
        self.att_data = att_data
        self.test_att_data =[att_data[i-1] for i in cat_ids]

        self.category_id_to_class_id = {
            v: i for i, v in enumerate(cat_ids)}
        self.class_id_to_category_id = {
            v: k for k, v in self.category_id_to_class_id.items()}

    def attribute_to_category_id(self, att):
        #KNN
        category = knn(self.test_att_data, att, self.cat_ids, 1)
        return category 

CUBMeta = _CUBMeta()


class CUBDetection(object):
    def __init__(self, basedir, name):
        assert name in CUBMeta.INSTANCE_TO_BASEDIR.keys(), name
        self.name = name
        self._imgdir = os.path.join(basedir, 'images/')
        self._segdir = os.path.join(basedir, 'segmentations/')
        self._matdir = os.path.join(basedir, 'mat/')
        assert os.path.isdir(self._matdir), self._matdir
        assert os.path.isdir(self._imgdir), self._imgdir
        assert os.path.isdir(self._segdir), self._segdir

        annotation_file = '{}/split/{}.txt'.format(basedir, name)
        assert os.path.isfile(annotation_file), annotation_file
        self._examples, self._examples_num = load_file(annotation_file)

        imageAttributes_file=os.path.join(basedir, 'zero-shot/image_attribute_labels.mat')
        imageAttributes=scipy.io.loadmat(imageAttributes_file)['imageAttributes']
        self._imageAttributes=np.transpose(imageAttributes)

        classAttributes_file=os.path.join(basedir, 'zero-shot/class_attribute_labels_continuous.mat')
        classAttributes=scipy.io.loadmat(classAttributes_file)['classAttributes']
        config.AttData = classAttributes* (1. / 100)

        class_file = '{}/split/{}_cls.txt'.format(basedir, name)
        assert os.path.isfile(class_file), class_file
        cat_data = np.genfromtxt(class_file, dtype=None)
        cat_ids=[]
        for i in range(len(cat_data)):
            cat_ids.append(cat_data[i])

        if not CUBMeta.valid():
            CUBMeta.create(cat_ids, config.AttData)
        else:
            assert CUBMeta.cat_ids == cat_ids

        logger.info("Instances loaded from {}.".format(annotation_file))

    def load(self, is_train=True):
        """
        Args:
            add_gt: whether for train(True) or for val(False)
        Returns:
            a list of dict, each has keys including:
                height, width, id, file_name,
                and (if add_gt is True) boxes, class, is_crowd
        """

        with timed_operation('Load Groundtruth Boxes for {}'.format(self.name)):
            imgs=[]
            img_ids = self._examples

            # list of dict, each has keys: height,width,id,file_name
            for i in img_ids:
                id=i
                example='{}/{}.mat'.format(self._matdir, i)
                mat = scipy.io.loadmat(example)
                fname,fextension=os.path.splitext(mat['GT'][0]['imagePath'][0][0][0][0][:])

                file_name= self._imgdir+fname+'.jpg'
                assert os.path.isfile(file_name), file_name
                im = cv2.imread(file_name, cv2.IMREAD_COLOR)
                assert im is not None, file_name

                #obj
                #class
                category=mat['GT'][0]['cls'][0][0][0]
                cls=[]
                cls.append(CUBMeta.category_id_to_class_id[category])
                cls=np.array(cls,dtype=np.int32)

                #attribute
                att=[]
                #attribute=self._imageAttributes[i-1]
                attribute=config.AttData[category-1]
                att.append(attribute)
                att=np.array(att,dtype=np.float32)

                #box location
                bbox=mat['GT'][0]['box'][0][0]
                x1=float(bbox[0])
                y1=float(bbox[1])
                w=float(bbox[2])
                h=float(bbox[3])
                # bbox is originally in float
                # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
                # But we do assume that (0.0, 0.0) is upper-left corner of the first pixel
                boxes=[]
                boxes.append([x1, y1, x1 + w, y1 + h])
                boxes=np.array(boxes,dtype=np.float32)

                #mask
                seg_name = self._segdir+fname+'.png'
                assert os.path.isfile(seg_name), seg_name
                seg = cv2.imread(seg_name,0)
                assert seg is not None, seg_name

                img={'id':id,'file_name':file_name,'image':im, 'class':cls,'attribute':att,'boxes':boxes,'seg':seg,'seg_name':seg_name}
                imgs.append(img)             
            return imgs

    @staticmethod
    def load_many(basedir, names, is_train=True):
        """
        Load and merges several instance files together.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            cub = CUBDetection(basedir, n)
            ret.extend(cub.load(is_train))
        return ret

    def print_class_histogram(self, imgs):
        nr_class = len(CUBMeta.cat_ids)
        hist_bins = np.arange(nr_class + 1)

        # Histogram of ground-truth objects
        gt_hist = np.zeros((nr_class,), dtype=np.int)
        for entry in imgs:
            # filter crowd?
            gt_inds = np.where(
                (entry['class'] > 0) & (entry['is_crowd'] == 0))[0]
            gt_classes = entry['class'][gt_inds]
            gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
        data = [[CUBMeta.cat_ids[i], v] for i, v in enumerate(gt_hist)]
        data.append(['total', sum([x[1] for x in data])])
        table = tabulate(data, headers=['class', '#box'], tablefmt='pipe')
        logger.info("Ground-Truth Boxes:\n" + colored(table, 'cyan'))

if __name__ == '__main__':
    c = CUBDetection('dataset/CUB', 'zsl_test_1')
    gt_boxes = c.load(is_train=True)
    print("#Images:", len(gt_boxes))
    print(gt_boxes[0])
