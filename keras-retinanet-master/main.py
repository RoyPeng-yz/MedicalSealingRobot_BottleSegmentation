#!/usr/bin/env python

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
from keras_retinanet.bin.test import test_funtion

result,len,test_generator=test_funtion()
