# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:46:52 2023

@author: Francesco Chiti Tegli
"""

import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
from random import randrange
from random import shuffle
from random import choice

def loadFiles(path):
  trainImages = []
  for r, d, f in os.walk(path):
     for file in f:
        filepath = os.path.join(r, file)
        img = cv2.imread(filepath)
        trainImages.append(img)

  return trainImages

cutout = np.array(loadFiles('C:/Users/fchit/Downloads/cutoutActualCandPS')).astype('float64')
cutout /= 255.
cutout_mono_pre = cutout[...,1]
cutout_mono = []
for i in np.arange(cutout_mono_pre.shape[0]):
    cutout_mono.append(cv2.resize(cutout_mono_pre[i], (128,128) ))
cutout_mono = np.array(cutout_mono)
cutout_mono = np.reshape(cutout_mono, (cutout_mono.shape[0], cutout_mono.shape[1], cutout_mono.shape[2], 1))

#LENSQUAS6623
#LENSQUASfits9823
ready_model = tf.keras.models.load_model("C:/Users/fchit/Downloads/LENSQUASfits9823")

#%%
ready_predict = np.ndarray.flatten(ready_model.predict(cutout_mono))
ready_predict_bi = np.zeros(len(ready_predict))
for i in np.arange(len(ready_predict)):
    if ready_predict[i]<0.5:
        ready_predict_bi[i]=0
    else:
        ready_predict_bi[i]=1
#%%
plt.figure(1)
plt.suptitle('Raw Results READY', fontsize=14)
plt.hist(ready_predict, 50)
plt.show()
#%%
counter_ready = 0

good = []
good_score = []
good_index = []

bad = []
bad_score = []
bad_index = []

for i in np.arange(len(ready_predict_bi)):
    counter_ready += abs(ready_predict_bi[i] - 0)
    if (ready_predict_bi[i] == 0):
        bad.append(cutout_mono[i])
        bad_score.append(ready_predict[i])
        bad_index.append(i)
    else:
        good.append(cutout_mono[i])
        good_score.append(ready_predict[i])
        good_index.append(i)

good = np.array(good)
good_score = np.array(good_score)
good_index = np.array(good_index)
bad = np.array(bad)
bad_score = np.array(bad_score)
bad_index = np.array(bad_index)

counter_ready = counter_ready / len(ready_predict_bi)
print('counter_ready = ' + str(counter_ready))
print( )

for i in np.arange(len(ready_predict)):
    if ((ready_predict[i] < 0.7) and (ready_predict[i] > 0.3)):
        print(i)
        print(ready_predict[i])
        print( )

print('---------------------------------------------------- good')
perfect = 0
perfect_array = []

for i in np.arange(len(good)):
    if (good_score[i] < 0.8):
        print(i)
        print(good_score[i])
        print( )
    if (good_score[i] == 1.):
        perfect = perfect + 1
        perfect_array.append(good[i])

perfect_array = np.array(perfect_array)
print('perfect = ' + str(perfect))
print( )
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ bad')
awful = 0

for i in np.arange(len(bad)):
    if (bad_score[i] > 0.3):
        print(i)
        print(bad_score[i])
        print( )
    if (bad_score[i] == 0.):
        awful = awful + 1

print('awful = ' + str(awful))
print( )