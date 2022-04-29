import numpy as np
import os
import random


TRAIN_CLS_NUM=150
TEST_CLS_NUM=50


class ZSL_DATA(object):
    def __init__(self, basedir):
        class_file = os.path.join(basedir, 'classes.txt')
        assert os.path.isfile(class_file), class_file
        cat_data = np.genfromtxt(class_file, dtype='i8,S40')
        image_class_file = os.path.join(basedir, 'image_class_labels.txt')
        assert os.path.isfile(image_class_file), image_class_file
        image_data = np.genfromtxt(image_class_file, dtype='i8')

        self._cls_dic={}
        self._cls_list=[]
        for i in range(len(cat_data)):
            cat_id=cat_data[i][0]
            self._cls_list.append(cat_id)
            self._cls_dic[cat_id]=cat_data[i][1]

        self._image_dic={}
        self._image_list=[]
        for i in range(len(image_data)):
            image_id=image_data[i][0]
            self._image_list.append(image_id)
            self._image_dic[image_id]=image_data[i][1]


    def select_sample(self,random_id,train_num,test_num):
        random.shuffle(self._cls_list)
        assert (train_num+test_num)==len(self._cls_list), "wrong class number"

        self._train_cls_list=self._cls_list[:train_num]
        self._test_cls_list=self._cls_list[train_num:]

        save_train_cls_file='zsl_train_{}_cls.txt'.format(random_id)
        save_test_cls_file='zsl_test_{}_cls.txt'.format(random_id)

        save_train_file='zsl_train_{}.txt'.format(random_id)
        save_test_file='zsl_test_{}.txt'.format(random_id)

        with open(save_train_cls_file,'wt') as f: 
            for id in self._train_cls_list:
                print(id,file=f)

        with open(save_test_cls_file,'wt') as f: 
            for id in self._test_cls_list:
                print(id,file=f)

        with open(save_train_file,'wt') as ftrain, open(save_test_file,'wt') as ftest:
            for id in self._image_list:
                if self._image_dic[id] in self._train_cls_list:
                    print(id,file=ftrain)
                else:                
                    print(id,file=ftest)
                


if __name__ == '__main__':

    basedir='dataset/CUB'
    CUBzsl=ZSL_DATA(basedir)
    CUBzsl.select_sample(1,TRAIN_CLS_NUM,TEST_CLS_NUM)
    CUBzsl.select_sample(2,TRAIN_CLS_NUM,TEST_CLS_NUM)
    CUBzsl.select_sample(3,TRAIN_CLS_NUM,TEST_CLS_NUM)
    CUBzsl.select_sample(4,TRAIN_CLS_NUM,TEST_CLS_NUM)
    CUBzsl.select_sample(5,TRAIN_CLS_NUM,TEST_CLS_NUM)