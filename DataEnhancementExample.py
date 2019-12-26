# -*- coding: utf-8 -*-
"""
Created on 2019.07.05

@author: 30753
"""
from DataEnhancement import DataEnhancement
import cv2
imgdir = 'C:/Users/30753/Desktop/data/JPEGImages/'
xmldir = 'C:/Users/30753/Desktop/data/Annotations/'

DEE = DataEnhancement(imgdir, xmldir)
imageFilePath_list=DEE.get_filePathList(imgdir, '.jpg')
i=10
for imageFilePath in imageFilePath_list:
    # DEE.reset_imgfilesuffix(imageFilePath,'.png')#修改图片后缀名，并生成新的图片
    DEE.reset_xml(imageFilePath)## 输入img路径，根据img的名字修改其对应的xml的名字、路径存储内容
    DEE.Horizontal_Mirror(imageFilePath,DEE.num_to_str_6(i))#图片+xml水平镜像
    i=i+1
    DEE.Vertical_Mirror(imageFilePath,DEE.num_to_str_6(i))#图片+xml垂直镜像
    i=i+1
    DEE.Horizontal_Vertical_Mirror(imageFilePath,DEE.num_to_str_6(i))#图片+xml水平垂直镜像
    i=i+1
    DEE.GammaTranform(1,2,imageFilePath,DEE.num_to_str_6(i))#gamme滤波
    i=i+1
    DEE.BlurFilter(imageFilePath,DEE.num_to_str_6(i))
    i=i+1
    DEE.EdgeEnhance(imageFilePath,DEE.num_to_str_6(i))
    i=i+1
    DEE.xmlResetClass(imageFilePath,['11111','22222'],['safeguard','cola'])#更改标签class
    # DEE.Resize(imageFilePath,640,360)#修改图片及方框大小


