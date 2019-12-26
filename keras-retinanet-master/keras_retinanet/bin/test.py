#!/usr/bin/env python
import argparse
import os
import sys
import warnings

import keras
import keras.preprocessing.image
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.kitti import KittiGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator

def get_session():
    """ Construct a modified tf session.
        动态申请显存
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model

def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model

def create_generators(args, preprocess_image):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
        'shuffle_groups': False,
    }

    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(flip_x_chance=0.5)


    if args.dataset_type == 'csv':
        if args.test_annotations:
            test_generator = CSVGenerator(
                args.test_annotations,
                args.classes,
                **common_args
            )
        else:
            validation_generator = None
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return test_generator

class Arg():
    def __init__(self):
        self.backbone='resnet50'
        self.gpu=0
        self.dataset_type='csv'
        self.batch_size=1
        self.config= None
        self.image_min_side=648
        self.image_max_side=864
        self.test_annotations='F:/Programming/Git_Repository/MedicalSealingRobot_BottleSegmentation/keras-retinanet-master/keras_retinanet/val.csv'
        self.classes='F:/Programming/Git_Repository/MedicalSealingRobot_BottleSegmentation/keras-retinanet-master/keras_retinanet/class.csv'
        self.weights='F:/Programming/Git_Repository/MedicalSealingRobot_BottleSegmentation/bottle_Segmentation2.h5'
        self.lr=1e-5
        self.multi_gpu=0
        self.steps=200
        self.max_queue_size=10
        self.workers=0

def test_funtion():
    # args={
    #     'backbone':'resnet50',
    #     'gpu':0,
    #     'dataset_type':'csv',
    #     'batch_size': 1,
    #     'config': None,
    #     'image_min_side': 648,
    #     'image_max_side': 864,
    #     'test_annotations':'F:/Programming/Git_Repository/MedicalSealingRobot_BottleSegmentation/keras-retinanet-master/keras_retinanet/val.csv',
    #     'classes':3,
    #     'weights':'F:/Programming/Git_Repository/MedicalSealingRobot_BottleSegmentation/bottle_Segmentation2.h5',
    #     'lr':1e-5,
    #     'multi_gpu':0,
    #     'steps':200,
    #     'max_queue_size':10,
    #     'workers':1,
    # }
    args=Arg()

    # create object that stores backbone information

    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU指定特定GPU设备
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())# 设置cpu使用动态申请显存（需要多少申请多少）

    # optionally load config parameters配置参数.ini文件的路径。
    if args.config:
        args.config = read_config_file(args.config)

    # create the generators构造生成器
    test_generator = create_generators(args, backbone.preprocess_image)#根据csv等类型文件得到指定随机变换的图片数据生成器

    # create the model可以从指定文件权值初始化模型
    weights = args.weights

    print('Creating model, this may take a second...')#初始化模型
    model, training_model, prediction_model = create_models(
        backbone_retinanet=backbone.retinanet,
        num_classes=test_generator.num_classes(),
        weights=weights,
        multi_gpu=args.multi_gpu,
        lr=args.lr,
        config=args.config
    )

    # print model summary打印模型摘要
    print(model.summary())

    # # create the callbacks回调
    # callbacks = create_callbacks(
    #     model,
    #     training_model,
    #     prediction_model,
    #     validation_generator,
    #     args,
    # )

    # Use multiprocessing if workers > 0
    if args.workers > 0:
        use_multiprocessing = True
    else:
        use_multiprocessing = False

    # start training开始测试
    return prediction_model.predict_generator(
        generator=test_generator,
        verbose=1,
        workers=args.workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=args.max_queue_size,
    ),len(test_generator),test_generator


# if __name__ == '__main__':
#     test_funtion()