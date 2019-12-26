import os
import xml.dom.minidom
import xml.etree.ElementTree
import cv2
import numpy as np
import math
from PIL import Image
from PIL import ImageFilter

class DataEnhancement():
    def __init__(self, imgdir, xmldir):
        self.imgdir = imgdir
        self.xmldir = xmldir
        self._imgsuffix = '.jpg'
    # 获得对应文件夹下的特定后缀文件的list路径格式
    # dirPath路径
    # partOfFileName后缀名

    def get_filePathList(self, dirPath, partOfFileName=''):
        allFileName_list = list(os.walk(dirPath))[0][2]
        fileName_list = [k for k in allFileName_list if partOfFileName in k]
        filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
        return filePath_list

    # 生成000001格式的字符串
    def num_to_str_6(self, num):
        s = str(num)
        n = len(s)
        for i in range(6-n):
            s = '0'+s
        return s

    # 修改图片后缀名变量
    def reset_imgsuffix(self, imgsuffix):
        self._imgsuffix = imgsuffix

    # 修改图片后缀名
    def reset_imgfilesuffix(self, imageFilePath, imgfilesuffix):
        shotname = os.path.split(imageFilePath)[-1].split('.')[0]
        newimageFilePath = self.imgdir+shotname+imgfilesuffix
        img = cv2.imread(imageFilePath)
        cv2.imwrite(newimageFilePath, img)

    # 输入img路径，根据img的名字修改其对应的xml的名字、路径存储内容
    def reset_xml(self, imageFilePath):
        shotname = os.path.split(imageFilePath)[-1].split('.')[0]
        xmlFilePath = self.xmldir+shotname+'.xml'
        if os.path.exists(xmlFilePath):
            # 读取xml文件
            dom = xml.dom.minidom.parse(xmlFilePath)
            root = dom.documentElement
            root.getElementsByTagName('filename')[
                0].firstChild.data = shotname+self._imgsuffix
            root.getElementsByTagName(
                'path')[0].firstChild.data = self.imgdir+shotname+self._imgsuffix
            # 保存xml文件
            with open(xmlFilePath, 'w') as fh:
                dom.writexml(fh)

# ----------------------------------私有方法----------------------------- #
    # 图像—————水平镜像翻转，返回变换后的图片
    def __img_shuipingjingxiang(self, imageFilePath):
        img = cv2.imread(imageFilePath)
        imgHeight, imgWidth, imgMode = img.shape
        shuiping = np.zeros((imgHeight, imgWidth, imgMode), np.uint8)
        for i in range(imgHeight):
            for j in range(imgWidth):
                shuiping[i, j] = img[i, imgWidth-j-1]
        return shuiping

    # 图像—————垂直镜像翻转，返回变换后的图片
    def __img_chuizhijingxiang(self, imageFilePath):
        img = cv2.imread(imageFilePath)
        imgHeight, imgWidth, imgMode = img.shape
        chuizhi = np.zeros((imgHeight, imgWidth, imgMode), np.uint8)
        for i in range(imgHeight):
            for j in range(imgWidth):
                chuizhi[i, j] = img[imgHeight-i-1, j]
        return chuizhi

    # 图像—————水平及垂直镜像翻转，返回变换后的图片
    def __img_shuipingchuizhijingxiang(self, imageFilePath):
        img = cv2.imread(imageFilePath)
        imgHeight, imgWidth, imgMode = img.shape
        chuizzhi_shuiping = np.zeros((imgHeight, imgWidth, imgMode), np.uint8)
        for i in range(imgHeight):
            for j in range(imgWidth):
                chuizzhi_shuiping[i, j] = img[imgHeight-i-1, imgWidth-j-1]
        return chuizzhi_shuiping

    # xml——————水平镜像函数，root为根节点，xmin、xmax为需要改变的标签框尺寸，width为图片尺寸
    def __xml_shuipingjingxiang(self, root, xmin, xmax, width):
        for xi,xa in zip(root.getElementsByTagName(xmin),root.getElementsByTagName(xmax)):
            xi.firstChild.data = width-int(xi.firstChild.data)-1
            xa.firstChild.data = width-int(xa.firstChild.data)-1
            if int(xi.firstChild.data) > int(xa.firstChild.data):
                z = int(xi.firstChild.data)
                xi.firstChild.data = int(xa.firstChild.data)
                xa.firstChild.data = z

    # xml——————垂直镜像函数，root为根节点，ymin、ymax为需要改变的标签框尺寸，height为图片尺寸
    def __xml_chuizhijingxiang(self, root, ymin, ymax, height):
        for yi,ya in zip(root.getElementsByTagName(ymin),root.getElementsByTagName(ymax)):
            yi.firstChild.data = height-int(yi.firstChild.data)-1
            ya.firstChild.data = height-int(ya.firstChild.data)-1
            if int(yi.firstChild.data) > int(ya.firstChild.data):
                z = int(yi.firstChild.data)
                yi.firstChild.data = int(ya.firstChild.data)
                ya.firstChild.data = z            

    # xml——————水平+垂直镜像函数，root为根节点，xmin、xmax、ymin、ymax为需要改变的标签框尺寸，height为图片尺寸
    def __xml_shuiping_chuizhijingxiang(self, root, xmin, xmax, ymin, ymax, width, height):
        for xi,xa in zip(root.getElementsByTagName(xmin),root.getElementsByTagName(xmax)):
            xi.firstChild.data = width-int(xi.firstChild.data)-1
            xa.firstChild.data = width-int(xa.firstChild.data)-1
            if int(xi.firstChild.data) > int(xa.firstChild.data):
                z = int(xi.firstChild.data)
                xi.firstChild.data = int(xa.firstChild.data)
                xa.firstChild.data = z
        for yi,ya in zip(root.getElementsByTagName(ymin),root.getElementsByTagName(ymax)):
            yi.firstChild.data = height-int(yi.firstChild.data)-1
            ya.firstChild.data = height-int(ya.firstChild.data)-1
            if int(yi.firstChild.data) > int(ya.firstChild.data):
                z = int(yi.firstChild.data)
                yi.firstChild.data = int(ya.firstChild.data)
                ya.firstChild.data = z 

    # 滤波——————Gamma变换调节明暗程度
    def __gammaTranform(self,c, gamma, image):
        h, w, d = image.shape[0], image.shape[1], image.shape[2]
        new_img = np.zeros((h, w, d), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                new_img[i, j, 0]=c*math.pow(image[i, j, 0], gamma)
                new_img[i, j, 1]=c*math.pow(image[i, j, 1], gamma)
                new_img[i, j, 2]=c*math.pow(image[i, j, 2], gamma)
        cv2.normalize(new_img,new_img,0,225,cv2.NORM_MINMAX)
        new_img = cv2.convertScaleAbs(new_img)
        return new_img
    #xml更改框大小,n倍缩小或放大
    def __xml_resize(self,root,nw,nh,nwflag=0,nhflag=0):
        if nwflag==0:
            for xi in root.getElementsByTagName('xmin'):
                xi.firstChild.data=int(int(xi.firstChild.data)//nw)
            for xa in root.getElementsByTagName('xmax'):
                xa.firstChild.data=int(int(xa.firstChild.data)//nw)
            if nhflag==0:
                for yi in root.getElementsByTagName('ymin'):
                    yi.firstChild.data=int(int(yi.firstChild.data)//nh)
                for ya in root.getElementsByTagName('ymax'):
                    ya.firstChild.data=int(int(ya.firstChild.data)//nh)
            else:
                for yi in root.getElementsByTagName('ymin'):
                    yi.firstChild.data=int(int(yi.firstChild.data)*nh)
                for ya in root.getElementsByTagName('ymax'):
                    ya.firstChild.data=int(int(ya.firstChild.data)*nh)                
        elif nwflag==1:
            for xi in root.getElementsByTagName('xmin'):
                xi.firstChild.data=int(int(xi.firstChild.data)*nw)
            for xa in root.getElementsByTagName('xmax'):
                xa.firstChild.data=int(int(xa.firstChild.data)*nw)
            if nhflag==0:
                for yi in root.getElementsByTagName('ymin'):
                    yi.firstChild.data=int(int(yi.firstChild.data)//nh)
                for ya in root.getElementsByTagName('ymax'):
                    ya.firstChild.data=int(int(ya.firstChild.data)//nh)
            else:
                for yi in root.getElementsByTagName('ymin'):
                    yi.firstChild.data=int(int(yi.firstChild.data)*nh)
                for ya in root.getElementsByTagName('ymax'):
                    ya.firstChild.data=int(int(ya.firstChild.data)*nh)       

# ------------------------------------------------------------------------ #

    # 图像+xml水平镜像并保存
    def Horizontal_Mirror(self, imageFilePath, newname):
        shotname=os.path.split(imageFilePath)[-1].split('.')[0]
        xmlFilePath=self.xmldir+shotname+'.xml'
        newimageFilePath=self.imgdir+newname+self._imgsuffix
        newxmlFilePath=self.xmldir+newname+'.xml'
        if os.path.exists(xmlFilePath):
            # 读取xml文件
            dom=xml.dom.minidom.parse(xmlFilePath)
            root=dom.documentElement
            width=int(root.getElementsByTagName('width')[0].firstChild.data)
            # 1.图片进行水平镜像，并保存
            imgshuiping=self.__img_shuipingjingxiang(imageFilePath)
            cv2.imwrite(newimageFilePath, imgshuiping)
            # 2.xml进行水平镜像，并保存
            self.__xml_shuipingjingxiang(root, 'xmin', 'xmax', width)
            root.getElementsByTagName('filename')[
                0].firstChild.data= newname + self._imgsuffix
            root.getElementsByTagName(
                'path')[0].firstChild.data=newimageFilePath
            # 保存xml文件
            with open(newxmlFilePath, 'w') as fh:
                dom.writexml(fh)

    # 图像+xml垂直镜像并保存
    def Vertical_Mirror(self, imageFilePath, newname):
        shotname=os.path.split(imageFilePath)[-1].split('.')[0]
        xmlFilePath=self.xmldir+shotname+'.xml'
        newimageFilePath=self.imgdir+newname+self._imgsuffix
        newxmlFilePath=self.xmldir+newname+'.xml'
        if os.path.exists(xmlFilePath):
            # 读取xml文件
            dom=xml.dom.minidom.parse(xmlFilePath)
            root=dom.documentElement
            height=int(root.getElementsByTagName('height')[0].firstChild.data)
            # 1.图片进行水平镜像，并保存
            imgchuizhi=self.__img_chuizhijingxiang(imageFilePath)
            cv2.imwrite(newimageFilePath, imgchuizhi)
            # 2.xml进行水平镜像，并保存
            self.__xml_chuizhijingxiang(root, 'ymin', 'ymax', height)
            root.getElementsByTagName('filename')[
                0].firstChild.data= newname + self._imgsuffix
            root.getElementsByTagName(
                'path')[0].firstChild.data=newimageFilePath
            # 保存xml文件
            with open(newxmlFilePath, 'w') as fh:
                dom.writexml(fh)

    # 图像+xml水平垂直镜像并保存
    def Horizontal_Vertical_Mirror(self, imageFilePath, newname):
        shotname=os.path.split(imageFilePath)[-1].split('.')[0]
        xmlFilePath=self.xmldir+shotname+'.xml'
        newimageFilePath=self.imgdir+newname+self._imgsuffix
        newxmlFilePath=self.xmldir+newname+'.xml'
        if os.path.exists(xmlFilePath):
            # 读取xml文件
            dom=xml.dom.minidom.parse(xmlFilePath)
            root=dom.documentElement
            width=int(root.getElementsByTagName('width')[0].firstChild.data)
            height=int(root.getElementsByTagName('height')[0].firstChild.data)
            # 1.图片进行水平镜像，并保存
            imgshuipingchuizhi=self.__img_shuipingchuizhijingxiang(
                imageFilePath)
            cv2.imwrite(newimageFilePath, imgshuipingchuizhi)
            # 2.xml进行水平镜像，并保存
            self.__xml_shuiping_chuizhijingxiang(
                root, 'xmin', 'xmax', 'ymin', 'ymax', width, height)
            root.getElementsByTagName('filename')[
                0].firstChild.data= newname + self._imgsuffix
            root.getElementsByTagName(
                'path')[0].firstChild.data=newimageFilePath
            # 保存xml文件
            with open(newxmlFilePath, 'w') as fh:
                dom.writexml(fh)

    # 图像Gamma变换调节明暗程度
    def GammaTranform(self, c, gamma, imageFilePath,newname):
        shotname=os.path.split(imageFilePath)[-1].split('.')[0]
        xmlFilePath=self.xmldir+shotname+'.xml'
        newimageFilePath=self.imgdir+newname+self._imgsuffix
        newxmlFilePath=self.xmldir+newname+'.xml'
        if os.path.exists(xmlFilePath):
            dom=xml.dom.minidom.parse(xmlFilePath)
            root=dom.documentElement
            img = cv2.imread(imageFilePath)
            new_img = self.__gammaTranform(c,gamma,img)#gamma
            cv2.imwrite(newimageFilePath,new_img)
            root.getElementsByTagName('filename')[
                0].firstChild.data= newname + self._imgsuffix
            root.getElementsByTagName(
                'path')[0].firstChild.data=newimageFilePath            
            with open(newxmlFilePath, 'w') as fh:
                dom.writexml(fh)

    # 图像模糊滤波
    def BlurFilter(self,imageFilePath,newname):
        shotname=os.path.split(imageFilePath)[-1].split('.')[0]
        xmlFilePath=self.xmldir+shotname+'.xml'
        newimageFilePath=self.imgdir+newname+self._imgsuffix
        newxmlFilePath=self.xmldir+newname+'.xml'
        if os.path.exists(xmlFilePath):
            dom=xml.dom.minidom.parse(xmlFilePath)
            root=dom.documentElement
            img=Image.open(imageFilePath)
            im_blur = img.filter(ImageFilter.BLUR)#模糊滤波
            im_blur.save(newimageFilePath)
            root.getElementsByTagName('filename')[
                0].firstChild.data= newname + self._imgsuffix
            root.getElementsByTagName(
                'path')[0].firstChild.data=newimageFilePath            
            with open(newxmlFilePath, 'w') as fh:
                dom.writexml(fh)

    # 图像边界增强
    def EdgeEnhance(self,imageFilePath,newname):
        shotname=os.path.split(imageFilePath)[-1].split('.')[0]
        xmlFilePath=self.xmldir+shotname+'.xml'
        newimageFilePath=self.imgdir+newname+self._imgsuffix
        newxmlFilePath=self.xmldir+newname+'.xml'
        if os.path.exists(xmlFilePath):
            dom=xml.dom.minidom.parse(xmlFilePath)
            root=dom.documentElement
            img=Image.open(imageFilePath)
            im_edge = img.filter(ImageFilter.EDGE_ENHANCE)#边界增强
            im_edge.save(newimageFilePath)
            root.getElementsByTagName('filename')[
                0].firstChild.data= newname + self._imgsuffix
            root.getElementsByTagName(
                'path')[0].firstChild.data=newimageFilePath            
            with open(newxmlFilePath, 'w') as fh:
                dom.writexml(fh)

    # 改xml中的class标签
    def xmlResetClass(self,imageFilePath,originalClass=[],newClass=[]):
        shotname=os.path.split(imageFilePath)[-1].split('.')[0]
        xmlFilePath=self.xmldir+shotname+'.xml'
        if os.path.exists(xmlFilePath):
            dom=xml.dom.minidom.parse(xmlFilePath)
            root=dom.documentElement
            for index,value in enumerate(originalClass):
                for name in root.getElementsByTagName('name'):
                    if name.firstChild.data == value:
                        name.firstChild.data=newClass[index]
            with open(xmlFilePath, 'w') as fh:
                dom.writexml(fh) 
    
    # 改变一张图片+xml的size，路径为图片绝对路径
    def Resize(self,imageFilePath,newwidth,newheight):
        shotname=os.path.split(imageFilePath)[-1].split('.')[0]
        xmlFilePath=self.xmldir+shotname+'.xml'
        newimageFilePath=imageFilePath
        newxmlFilePath=xmlFilePath
        if os.path.exists(xmlFilePath):
            dom=xml.dom.minidom.parse(xmlFilePath)
            root=dom.documentElement
            img=cv2.imread(imageFilePath)
            imgHeight, imgWidth, imgMode = img.shape
            #调整图片大小             
            img=cv2.resize(img, (newwidth, newheight))
            cv2.imwrite(newimageFilePath,img)
            #调整xml方框大小
            if newwidth<imgWidth:
                nw=imgWidth/newwidth
                nwflag=0
            else:
                nw=newwidth/imgWidth
                nwflag=1
            if newheight<imgHeight:           
                nh=imgHeight/newheight
                nhflag=0
            else:
                nh=newheight/imgHeight
                nhflag=1
            self.__xml_resize(root,nw,nh,nwflag,nhflag)   
            root.getElementsByTagName('width')[
                0].firstChild.data= newwidth
            root.getElementsByTagName(
                'height')[0].firstChild.data=newheight            
            with open(newxmlFilePath, 'w') as fh:
                dom.writexml(fh)        



