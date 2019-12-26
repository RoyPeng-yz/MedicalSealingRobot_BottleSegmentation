from PIL import Image
import os
import cv2
import time
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# 获取文件夹中的文件路径
import os
def get_filePathList(dirPath, partOfFileName=''):
    allFileName_list = list(os.walk(dirPath))[0][2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list
    
# 检测单张图片，返回画框后的图片
def get_detectedImage(retinanet_model, image):
    labels_to_names = {0:'Groove', 1:'Sealed', 2:'Unsealed'}
    startTime = time.time()
    new_size = (864, 648)
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    boxes, scores, labels = retinanet_model.predict_on_batch(np.expand_dims(image, axis=0))
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.9:
            break
        # if label == 0:
        #     break
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    usedTime = time.time() - startTime
    print("检测这张图片用时%.2f秒"  %usedTime)
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
#    cv2.namedWindow("result", 0)
#    cv2.resizeWindow("result", 500, 747)
#    cv2.imshow('result', draw)
    return draw


# 解析运行代码文件时传入的参数
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirPath', type=str, help='directory path', default='./keras_retinanet/JPEGImages')
    parser.add_argument('-s', '--suffix', type=str, default='000046.bmp')
#    parser.add_argument('-s', '--suffix', type=str, default='.png')
    parser.add_argument('-m', '--modelFilePath', type=str, default='./bottle_Segmentation2.h5')
    parser.add_argument('-o', '--out_mp4FilePath', type=str, default='fish_output.avi')
    argument_namespace = parser.parse_args()
    return argument_namespace  


import math
import numpy as np
# gamme滤波
def gammaTranform(c, gamma, image):
    h, w, d = image.shape[0], image.shape[1], image.shape[2]
    new_img = np.zeros((h, w, d), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            new_img[i, j, 0] = c * math.pow(image[i, j, 0], gamma)
            new_img[i, j, 1] = c * math.pow(image[i, j, 1], gamma)
            new_img[i, j, 2] = c * math.pow(image[i, j, 2], gamma)
    cv2.normalize(new_img, new_img, 0, 225, cv2.NORM_MINMAX)
    new_img = cv2.convertScaleAbs(new_img)
    return new_img

# 主函数 
argument_namespace = parse_args()
dirPath = argument_namespace.dirPath
suffix = argument_namespace.suffix
modelFilePath = argument_namespace.modelFilePath
retinanet_model = models.load_model(modelFilePath, backbone_name='resnet50')

#单张
imageFilePath = os.path.join(dirPath, suffix)
image = read_image_bgr(imageFilePath)
# new_image=gammaTranform(1,1.5,image)
result = get_detectedImage(retinanet_model,image)
cv2.imshow('result', result)
while cv2.waitKey(25)!=27:
    continue

# #多张图片
# imgdir = './keras_retinanet/JPEGImages/'
# imageFilePath_list = get_filePathList(imgdir, '.bmp')
# for imagefilepath in imageFilePath_list:
#     image = read_image_bgr(imagefilepath)
#     result=get_detectedImage(retinanet_model,image)
#     cv2.imshow('result',result)
#     c=cv2.waitKey(25)
#     if c == 27:
#         break

# #视频
# cap = cv2.VideoCapture('sw1.mp4')#打开相机
#  #创建VideoWriter类对象
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps =cap.get(cv2.CAP_PROP_FPS)
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# out = cv2.VideoWriter('result.mp4',fourcc, 20.0, (1920,1080))
# print('000')
# while(True):
#     ret,frame = cap.read()#捕获一帧图像
#     if ret:
#         print('111')
#         result=get_detectedImage(retinanet_model,frame)
#         cv2.imshow('result',result)
#         out.write(result)#保存帧
#         c=cv2.waitKey(25)
#         if c == 27:
#             break
#     else:
#         break
# print('222')
# cap.release()#关闭相机
# out.release()
# cv2.destroyAllWindows()#关闭窗口