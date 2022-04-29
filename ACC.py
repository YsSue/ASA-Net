import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from math import isnan,exp
import cv2
import os
import scipy.io

import config
from CUB_ZSL import CUBDetection
from CUB_ZSL import CUBMeta
import json

#ATT
def cal_auc(label, predict):  
    fpr, tpr, thresholds = roc_curve(label, predict, pos_label=1)
    auc_val=auc(fpr, tpr)
    return auc_val

def cal_mAUC(gt,pred,data_num,dim,threshold):
    gt=np.array(gt)
    pred=np.array(pred)
    #print gt
    for m in range(data_num):
        for n in range(dim):
             gt[m,n]= 1 if gt[m,n]>threshold else 0
    #print gt
    mAUC=0
    c_dim=dim
    AUC_list=[]
    for i in range(dim):
        label=gt[:,i]
        predict=pred[:,i]
        auc=cal_auc(label, predict)
        AUC_list.append(auc)
        if isnan(auc):
             c_dim=c_dim-1
        else:
             mAUC=mAUC+auc
    if (c_dim):
        mAUC=mAUC/c_dim
    else:
        mAUC=0
    return mAUC,AUC_list

#Cls
def cal_MCA(gt_cls,pred_cls,num_class):
    gt_cnt = {}
    for gt_cl in gt_cls:
        gt_cnt[gt_cl] = gt_cnt.get(gt_cl,0)+1
    correct = {k:0 for k in gt_cnt.keys()}
    for i in range(len(gt_cls)):
        if (gt_cls[i]==pred_cls[i]):
            correct[gt_cls[i]]+=1
    rate={}
    for k in gt_cnt.keys():
        rate[k]=correct[k]*1.0/gt_cnt[k]
    MCA=sum(rate.values())*1.0/len(rate)

    return rate,MCA

def accuracy(predictions, labels):
    #acc=100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
    acc=np.sum(predictions == labels) / predictions.shape[0]
    return acc

#seg 
def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U

def cal_iou(scores,mask):
    # Accuracy
    cum_I, cum_U = 0, 0
    seg_correct = 0
    seg_total = 0
    score_thresh = 0.5

    labels= mask[0]
    labels= (labels> 0.5)
    labels = np.squeeze(labels)
    scores_val = scores[0]
    scores_val = np.squeeze(scores_val)
    # Evaluate the segmentation performance
    predicts  = (scores_val >= score_thresh).astype(np.float32)
    I, U = compute_mask_IU(predicts, labels)
    this_IoU = I*1.0/U
    return this_IoU

def seg_IOU(scores,mask,batch_size):
    # Accuracy
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    score_thresh = 0.5

    num_im = batch_size
    for n_im in range(num_im):
        #print('testing image %d / %d' % (n_im, num_im))
        labels= mask[n_im]
        labels= (labels> score_thresh)
        labels = np.squeeze(labels)
        scores_val = scores[n_im]
        scores_val = np.squeeze(scores_val)

        # Evaluate the segmentation performance
        predicts  = (scores_val >= score_thresh).astype(np.float32)
        I, U = compute_mask_IU(predicts, labels)
        cum_I += I
        cum_U += U
        this_IoU = I*1.0/U
        for n_eval_iou in range(len(eval_seg_iou_list)):
            eval_seg_iou = eval_seg_iou_list[n_eval_iou]
            seg_correct[n_eval_iou] += (this_IoU >= eval_seg_iou)
        seg_total += 1

    # Print results
    #print('Final results on the whole test set')
    result_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        result_str += 'precision@%s = %f\n' % \
           (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou]*1.0/seg_total)

    result_str += 'overall IoU = %f\n' % (cum_I*1.0/cum_U)
    return cum_I*1.0/cum_U,result_str

#cla & seg	
def cal_MSO(gt_cls,pred_cls,num_class,scores,mask):
    gt_cnt = {}
    for gt_cl in gt_cls:
        gt_cnt[gt_cl] = gt_cnt.get(gt_cl,0)+1
    IoU = {k:0 for k in gt_cnt.keys()}
    for i in range(len(gt_cls)):
        iou=cal_iou([scores[i]],[mask[i]])
        IoU[gt_cls[i]]+=iou
    rate={}
    for k in gt_cnt.keys():
        rate[k]=IoU[k]*1.0/gt_cnt[k]

    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    out=0
    result_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        result = mso(gt_cls,pred_cls,num_class,scores,mask,eval_seg_iou_list[n_eval_iou])
        result_str += 'MSO@%s = %f\n' % (str(eval_seg_iou_list[n_eval_iou]),result)
        out=out+result

    result_str += 'average MSO = %f\n' % (out*1.0/5)
    return rate,out*1.0/5,result_str

def mso(gt_cls,pred_cls,num_class,scores,mask,threshold):
    gt_cnt = {}
    for gt_cl in gt_cls:
        gt_cnt[gt_cl] = gt_cnt.get(gt_cl,0)+1
    correct = {k:0 for k in gt_cnt.keys()}
    score_thresh = 0.5
    for i in range(len(gt_cls)):
        if (gt_cls[i]==pred_cls[i]):
            l= mask[i]
            l= (l> score_thresh)
            l = np.squeeze(l)
            scores_val = scores[i]
            scores_val = np.squeeze(scores_val)
            # Evaluate the segmentation performance
            predicts  = (scores_val >= score_thresh).astype(np.float32)
            I, U = compute_mask_IU(predicts, l)
            this_IoU = I*1.0/U
            if (this_IoU>threshold):
                correct[gt_cls[i]]+=1

    rate={}
    for k in gt_cnt.keys():
        rate[k]=correct[k]*1.0/gt_cnt[k]
    MSO=sum(rate.values())*1.0/len(rate)

    return MSO	

def trans_semantic(gt_cls,pred_cls,gtmask,premask):
    score_thresh = 0.5
    gt_sseg=[]
    pre_sseg=[]
    for i in range(len(gt_cls)):
        pre=  np.squeeze(premask[i]>=score_thresh)
        gt=  np.squeeze(gtmask[i]>=score_thresh)
        pre=pre.astype(int)*pred_cls[i]
        gt=gt.astype(int)*gt_cls[i]
        pre_sseg.append(pre)
        gt_sseg.append(gt)

    return gt_sseg,pre_sseg	

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def compute_mIoU(gt_masks,masks,num_class):
    hist = np.zeros((num_class, num_class))#hist initialize
    for ind in range(len(gt_masks)):
        label=gt_masks[ind]
        pred=masks[ind]
        hist += fast_hist(label.flatten(), pred.flatten(), num_class)#ClassxClass
    mIoUs = per_class_iu(hist)#mIoU for each class
    return mIoUs

def load_file(examples_list_file):
    lines = np.genfromtxt(examples_list_file, dtype='i8')
    examples = []
    for example in lines:
        examples.append(example)
    return np.asarray(examples), len(lines)
	
def test():
    c = CUBDetection('dataset/CUB', 'zsl_test_1')
    gt_boxes = c.load(is_train=True)
    
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

    for img in gt_boxes:
        id, fname, boxes, cls, is_crowd, att = img['id'],img['file_name'], img['boxes'], img['class'], img['is_crowd'], img['attribute']
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        im = im.astype('float32')

        im = cv2.resize(im, (config.INPUT_SIZE,config.INPUT_SIZE)) 
        ims=[]
        ims.append(im)
        ims=np.asarray(ims)

        # one image-sized binary mask per box
        seg_name=img['seg_name']
        seg = cv2.imread(seg_name,0)
        r,seg=cv2.threshold(seg,127,1,cv2.THRESH_BINARY) 
        masks = [] 
        masks.append(seg)       
        masks = np.asarray(masks, dtype='uint8')    # values in {0, 1}

        bbox = boxes[0]
        #bbox[2] -= bbox[0]
        #bbox[3] -= bbox[1]

        contoursGT,hierarchy=cv2.findContours(255*masks[0],cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        contoursGT_list=[]
        for countour in contoursGT:
            contoursGT_list.append(countour.tolist())
        gt = {
            'image_id': int(id),
            'image_file': str(fname),
            'category_id': int(CUBMeta.class_id_to_category_id[cls[0]]),
            'attribute':att[0].tolist(),
            'segmentation':contoursGT_list,
            'bbox': list(map(lambda x: float(round(x, 1)), bbox)),
            }
        allClsGT.append(CUBMeta.class_id_to_category_id[cls[0]])
        allMaskGT.append(masks[0])
        allAttGT.append(att[0])
        allBoxGT.append(bbox)
        all_GT.append(gt)


        #result
        box = boxes[0]
        cat_id = CUBMeta.attribute_to_category_id(att[0])
        width=masks[0].shape[1]
        height=masks[0].shape[0]
        mask=cv2.resize(masks[0], (width,height)) 
        r,mask_bin=cv2.threshold(mask,0.5,1,cv2.THRESH_BINARY)
        contours,hierarchy=cv2.findContours(255*mask_bin,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        contours_list=[]
        for countour in contours:
            contours_list.append(countour.tolist())

        res = {
             'image_id': int(id),
             'category_id': int(cat_id),
             'attribute':att[0].tolist(),
             'segmentation':contours_list,
             'bbox': list(map(lambda x: float(round(x, 1)), box)),
            }

        allCls.append(cat_id)
        allMask.append(mask)
        allAtt.append(att[0])
        allBox.append(box)
        all_results.append(res)


    #save json
    gt_path = 'valGT.json'
    json.dump(all_GT, open(gt_path, 'w'))
    res_path = 'valRes.json'
    json.dump(all_results, open(res_path, 'w'))

    #calculate accuracy
    score={}
    #attribute
    score['mAUC(att)'],score['AUC_list(att)'] =cal_mAUC(allAttGT,allAtt,len(allAtt),config.NUM_ATT,0.5)

    #class label
    score['Acc(class)']=accuracy(np.array(allCls), np.array(allClsGT))
    score['Acc_list(class)'],score['MCA(class)'] = cal_MCA(allClsGT,allCls,config.NUM_CATEGORY_TEST)

    #seg
    score['aIoU(seg)'],score['precision(seg)']=seg_IOU(allMask,allMaskGT,len(allMask))
    #seg&cls
    score['iou_list(segClass)'],score['mMSO(segClass)'],score['precision(segClass)']=cal_MSO(allClsGT,allCls,config.NUM_CATEGORY_TEST,allMask,allMaskGT)
    gt_sseg,pre_sseg=trans_semantic(allCls,allClsGT,allMaskGT,allMask)
    score['mIoU(segClass)']=compute_mIoU(gt_sseg,pre_sseg,config.NUM_CATEGORY+1)

    return all_results,all_GT,score

def print_evaluation_scores(gt_json_file,res_json_file,outputfold):
    with open(gt_json_file,'r') as f:
        all_GT=json.load(f)
    with open(res_json_file,'r') as f:
        all_results=json.load(f)
    assert len(all_GT)==len(all_results), "not match"

    os.mkdir(outputfold) 
    allMask=[]
    allMaskGT=[]
    allAtt=[]
    allAttGT=[]
    allCls=[]
    allClsGT=[]
    
    for i in range(len(all_GT)):
        gt=all_GT[i]
        res=all_results[i]
        assert gt['image_id']==res['image_id'], "not match"

        allClsGT.append(gt['category_id'])
        allCls.append(res['category_id'])

        allAttGT.append(gt['attribute'])
        allAtt.append(res['attribute'])

        contoursGT_list=gt['segmentation']
        contours_list=res['segmentation']

        contours=[]
        for c in range(len(contours_list)):
            countour=np.array(contours_list[c],dtype=np.int32)
            contours.append(countour)
        contoursGT=[]
        for c in range(len(contoursGT_list)):
            countour=np.array(contoursGT_list[c],dtype=np.int32)
            contoursGT.append(countour)

        #save segmentation
        im = cv2.imread(gt['image_file'], cv2.IMREAD_COLOR)
        assert im is not None, gt['image_file']
   
        mask_gt=np.zeros_like(im)
        mask=np.zeros_like(im)  
        cv2.drawContours(mask_gt, contoursGT, -1, (255, 255, 255), cv2.FILLED)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)
        mask_gt = cv2.cvtColor(mask_gt,cv2.COLOR_BGR2GRAY)
        ret,mask_gt = cv2.threshold(mask_gt,127,1,cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        ret,mask = cv2.threshold(mask,127,1,cv2.THRESH_BINARY)        
        allMaskGT.append(mask_gt)
        allMask.append(mask)

        im.flags.writeable = True
        for h in range(mask.shape[0]):
            for w in range(mask.shape[1]): 
                if mask[h,w]>0.5:
                    im[h,w][1]=im[h,w][1]/4
                    #im[h,w][2]=im[h,w][2]/4#blue
                    im[h,w][0]=im[h,w][0]/4#red

        this_iou=cal_iou([mask],[mask_gt])
        seg_file='{}/{}_{}.jpg'.format(outputfold,gt['image_file'],this_iou)
        cv2.imwrite(seg_file,im)
        print(seg_file)

    #calculate accuracy
    score={}
    #attribute
    score['mAUC(att)'],score['AUC_list(att)'] =cal_mAUC(allAttGT,allAtt,len(allAtt),config.NUM_ATT,0.5)

    #class label
    score['Acc(class)']=accuracy(np.array(allCls), np.array(allClsGT))
    score['Acc_list(class)'],score['MCA(class)'] = cal_MCA(allClsGT,allCls,config.NUM_CATEGORY_TEST)

    #seg
    score['aIoU(seg)'],score['precision(seg)']=seg_IOU(allMask,allMaskGT,len(allMask))
    #seg&cls
    score['iou_list(segClass)'],score['mMSO(segClass)'],score['precision(segClass)']=cal_MSO(allClsGT,allCls,config.NUM_CATEGORY_TEST,allMask,allMaskGT)
    gt_sseg,pre_sseg=trans_semantic(allCls,allClsGT,allMaskGT,allMask)
    score['mIoU(segClass)']=compute_mIoU(gt_sseg,pre_sseg,config.NUM_CATEGORY+1)

    return score
if __name__ == '__main__':
    #test()
    score=print_evaluation_scores('valGT.json','valRes.json','test')
    print (score)
