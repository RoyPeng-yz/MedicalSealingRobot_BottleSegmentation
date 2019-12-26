# -*- coding:utf-8 -*-
 
import csv
import os
import glob
import sys
import random

train_rate=0.7
val_rate=0.3
 
class PascalVOC2CSV(object):
    def __init__(self,xml=[], train_path='./keras-retinanet-master/keras_retinanet/train.csv', val_path='./keras-retinanet-master/keras_retinanet/val.csv',class_path='./keras-retinanet-master/keras_retinanet/class.csv'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param ann_path: ann_path
        :param classes_path: classes_path
        '''
        self.xml = xml
        self.train_path = train_path
        self.val_path = val_path
        self.class_path=class_path
        self.label=[]
        self.train=[]
        self.val=[]

        print('>> Create CSV File-------------')
        self.data_transfer()
        self.write_file()
 
 
    def data_transfer(self):
        train_len=int(train_rate*len(self.xml))
        list=range(len(self.xml))
        train_num=random.sample(list,train_len)
        # print('train_num='+str(train_num))
        for num, xml_file in enumerate(self.xml):
            try:
                # print(xml_file)
                # 进度输出
                sys.stdout.write('\r>> Converting image %d/%d' % (
                    num + 1, len(self.xml)))
                sys.stdout.flush()
 
                with open(xml_file, 'r') as fp:
                    for p in fp:
                        if '<filename>' in p:
                            self.filen_ame = p.split('>')[1].split('<')[0]
 
                        if '<object>' in p:
                            # 类别
                            d = [next(fp).split('>')[1].split('<')[0] for _ in range(9)]
                            self.supercategory = d[0]
                            if self.supercategory not in self.label:
                                self.label.append(self.supercategory)
 
                            # 边界框
                            x1 = int(d[-4]);
                            y1 = int(d[-3]);
                            x2 = int(d[-2]);
                            y2 = int(d[-1])
                            if num in train_num:
                                # print('train'+str(num))
                                self.train.append([os.path.join('JPEGImages',self.filen_ame),x1,y1,x2,y2,self.supercategory])
                            else:
                                # print('val'+str(num))
                                self.val.append([os.path.join('JPEGImages', self.filen_ame), x1, y1, x2, y2, self.supercategory])
            except:
                continue
 
        sys.stdout.write('\n')
        sys.stdout.flush()
 
    def write_file(self,):
        with open(self.train_path, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(self.train)

        with open(self.val_path, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(self.val)

        class_name=sorted(self.label)
        class_=[]
        for num,name in enumerate(class_name):
            class_.append([name,num])
        with open(self.class_path, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(class_)
 
 
# xml_file = glob.glob('./Annotations/*.xml')
# print(xml_file)
#
# PascalVOC2CSV(xml_file)