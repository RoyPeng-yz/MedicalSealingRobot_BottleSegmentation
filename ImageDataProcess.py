# -*- coding: utf-8 -*-
"""
Created on 2019.07.05

@author: 30753
"""
from DataEnhancement import DataEnhancement
from ImageNameChange import BatchRename
from pascal2csv import PascalVOC2CSV
import cv2
import glob

imgdir = 'F:/Programming/Git_Repository/MedicalSealingRobot_BottleSegmentation/JPEGImages/'
xmldir = 'F:/Programming/Git_Repository/MedicalSealingRobot_BottleSegmentation/Annotations/'

DEE = DataEnhancement(imgdir, xmldir)
imageFilePath_list = DEE.get_filePathList(imgdir, '.bmp')
xmlFilePath_list = DEE.get_filePathList(xmldir, '.xml')

#改图片大小，原先2592*1944，缩小3倍，864，648
# for imagefilepath in imageFilePath_list:
#     DEE.Resize(imagefilepath,864,648)

# #更改图片名字为：000000格式
# demo = BatchRename(imgdir, '.bmp', '.bmp')
# demo.rename()

# 生成csv文件
xml_file = glob.glob('./Annotations/*.xml')
PascalVOC2CSV(xml_file)
