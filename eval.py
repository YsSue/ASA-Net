#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple
import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
from tensorpack.utils.utils import get_tqdm_kwargs
from sklearn.metrics import confusion_matrix
from contextlib import ExitStack
from CUB_ZSL import CUBMeta as DataMeta
from common import CustomResize, clip_boxes
import config
import ACC

import json

DetectionResult = namedtuple(
    'DetectionResult',
    ['class_id','attribute','mask'])

"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""


def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.
    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])
    Returns:
        [DetectionResult]
    """
    labels, attribute, masks= model_func(img)
    results = [DetectionResult(*args) for args in zip(labels, attribute, masks)]
    return results


def eval_on_dataflow(df, detect_func):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]
    Returns:b
        list of dict, to be dumped to COCO json format
    """
    allMask=[]
    allMaskGT=[]
    allAtt=[]
    allAttGT=[]
    allCls=[]
    allClsGT=[]
    allBox=[]
    allBoxGT=[]
    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for image_id,image,cls,att,masks,boxes in df.get_data():
            results = detect_func(image)
            for r in results:
                #ground truth
                bbox = boxes[0]
                allClsGT.append(DataMeta.class_id_to_category_id[cls[0]])
                allMaskGT.append(masks[0])
                allAttGT.append(att[0])
                allBoxGT.append(bbox)

                #result
                box = boxes[0]
                width=masks[0].shape[1]
                height=masks[0].shape[0]
                mask=cv2.resize(r.mask, (width,height)) 
                if config.IS_ZSL:                    
                    cat_id = DataMeta.attribute_to_category_id(r.attribute)
                else:
                    cat_id = DataMeta.class_id_to_category_id(r.class_id)    
                allCls.append(cat_id)
                allMask.append(mask)
                allAtt.append(r.attribute)
                allBox.append(box)

            pbar.update(1)
    print("Evaluate annotation...")
    #calculate accuracy
    score={}
    #attribute
    score['mAUC(att)'],score['AUC_list(att)'] =ACC.cal_mAUC(allAttGT,allAtt,len(allAtt),config.NUM_ATT,0.5)

    #class label
    score['Acc(class)']=ACC.accuracy(np.array(allCls), np.array(allClsGT))
    score['Acc_list(class)'],score['MCA(class)'] = ACC.cal_MCA(allClsGT,allCls,config.NUM_CATEGORY_TEST)

    #seg
    score['aIoU(seg)'],score['precision(seg)']=ACC.seg_IOU(allMask,allMaskGT,len(allMask))
    #seg&cls
    score['iou_list(segClass)'],score['mMSO(segClass)'],score['precision(segClass)']=ACC.cal_MSO(allClsGT,allCls,config.NUM_CATEGORY_TEST,allMask,allMaskGT)
    gt_sseg,pre_sseg=ACC.trans_semantic(allCls,allClsGT,allMaskGT,allMask)
    score['mIoU(segClass)']=ACC.compute_mIoU(gt_sseg,pre_sseg,config.NUM_CATEGORY+1)

    return score

def test_on_dataflow(df, detect_func,outputfold):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]
    Returns:b
        list of dict, to be dumped to COCO json format
    """
    df.reset_state()
    all_results = []
    all_GT=[]
    allMask=[]
    allMaskGT=[]
    allAtt=[]
    allAttGT=[]
    allCls=[]
    allClsGT=[]
    allBox=[]
    allBoxGT=[]
    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for image_id,fname,image,cls,att,masks,boxes in df.get_data():
            results = detect_func(image)
            for r in results:
                #ground truth
                bbox = boxes[0]
                #bbox[2] -= bbox[0]
                #bbox[3] -= bbox[1]
                contoursGT,hierarchy=cv2.findContours(255*masks[0],cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
                contoursGT_list=[]
                for countour in contoursGT:
                    contoursGT_list.append(countour.tolist())

                gt = {
                    'image_id': int(image_id),
                    'image_file': str(fname),
                    'category_id': int(DataMeta.class_id_to_category_id[cls[0]]),
                    'attribute':att[0].tolist(),
                    'segmentation':contoursGT_list,
                    'bbox': list(map(lambda x: float(round(x, 1)), bbox)),
                }
                allClsGT.append(DataMeta.class_id_to_category_id[cls[0]])
                allMaskGT.append(masks[0])
                allAttGT.append(att[0])
                allBoxGT.append(bbox)
                all_GT.append(gt)

                #result
                box = boxes[0]
                if config.IS_ZSL:                    
                    cat_id = DataMeta.attribute_to_category_id(r.attribute)
                else:
                    cat_id = DataMeta.class_id_to_category_id(r.class_id)    
                width=masks[0].shape[1]
                height=masks[0].shape[0]
                mask=cv2.resize(r.mask, (width,height)) 
                ret,mask_bin=cv2.threshold(mask,0.5,1,cv2.THRESH_BINARY)
                mask_bin=np.asarray(mask_bin, dtype='uint8')    # values in {0, 1}
                contours,hierarchy=cv2.findContours(255*mask_bin,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
                contours_list=[]
                for countour in contours:
                    contours_list.append(countour.tolist())

                res = {
                    'image_id': int(image_id),
                    'category_id': int(cat_id),
                    'attribute':r.attribute.tolist(),
                    'segmentation':contours_list,
                    'bbox': list(map(lambda x: float(round(x, 1)), box)),
                }

                allCls.append(cat_id)
                allMask.append(mask)
                allAtt.append(r.attribute)
                allBox.append(box)
                all_results.append(res)
            pbar.update(1)

    #save json
    gt_path = '{}/valGT.json'.format(outputfold)
    json.dump(all_GT, open(gt_path, 'w'))
    res_path = '{}/valRes.json'.format(outputfold)
    json.dump(all_results, open(res_path, 'w'))

    #evaluate
    save_file='{}/Evaluate.txt'.format(outputfold)
    with open(save_file,'wt') as f: 
        print("Evaluate annotation...",file=f)
        score={}

        #class label
        print ("Category:",file=f)
        score['Acc(class)']=ACC.accuracy(np.array(allCls), np.array(allClsGT))
        print ("Acc of class:", score['Acc(class)'],file=f)
        score['Acc_list(class)'],score['MCA(class)'] = ACC.cal_MCA(allClsGT,allCls,config.NUM_CATEGORY_TEST)
        print ("AP list:",file=f)
        print (score['Acc_list(class)'],file=f)                  
        print ("category MCA:" ,score['MCA(class)'],file=f)
        C=confusion_matrix(allClsGT,allCls)
        print (C,file=f)

        #attribute
        print ("Attribute:",file=f)
        score['mAUC(att)'],score['AUC_list(att)'] =ACC.cal_mAUC(allAttGT,allAtt,len(allAtt),config.NUM_ATT,0.5)
        print ("attribute mAUC:", score['mAUC(att)'],file=f)
        print ("AUC of attribute:",file=f)
        print (score['AUC_list(att)'],file=f) 

        #seg
        print ("Segmentation:",file=f)
        score['aIoU(seg)'],score['precision(seg)']=ACC.seg_IOU(allMask,allMaskGT,len(allMask))
        print (score['precision(seg)'],file=f)
        #seg&cls
        print ("Segmentation & Category:",file=f)
        score['iou_list(segClass)'],score['mMSO(segClass)'],score['precision(segClass)']=ACC.cal_MSO(allClsGT,allCls,config.NUM_CATEGORY_TEST,allMask,allMaskGT)
        print ("MSO list:",file=f)
        print (score['iou_list(segClass)'],file=f)     
        print (score['precision(segClass)'],file=f)  
        gt_sseg,pre_sseg=ACC.trans_semantic(allCls,allClsGT,allMaskGT,allMask)
        score['mIoU(segClass)']=ACC.compute_mIoU(gt_sseg,pre_sseg,config.NUM_CATEGORY+1)  
        print ("semantic segmentation mIoU:",file=f)
        print (score['mIoU(segClass)'],file=f)

    return all_results,all_GT,score
