# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:15:31 2023

@author: Francesco Chiti Tegli
"""

import astropy.io
import numpy as np
from matplotlib import pyplot as plt
import os
from astropy.io import fits
import random
from random import randrange
from random import shuffle
from random import choice
import cv2

def loadFITS(path):
  trainImages = []
  for r, d, f in os.walk(path):
     for file in f:
        filepath = os.path.join(r, file)
        fit = fits.open(filepath)
        img = np.flipud(np.array(fit[0].data))
        img = np.reshape(img, (img.shape[0],img.shape[1], 1))
        trainImages.append(img)

  return trainImages

keep_y = np.array(loadFITS('C:/Users/fchit/Downloads/cutout/fits/good')).astype('float64')
discard_y = np.array(loadFITS('C:/Users/fchit/Downloads/cutout/fits/bad')).astype('float64')
single_y = np.array(loadFITS('C:/Users/fchit/Downloads/cutout/fits/apparent_single_source')).astype('float64')
discard_y = np.concatenate((discard_y, single_y))

keep_i = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/i/good')).astype('float64')
discard_i = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/i/bad')).astype('float64')
single_i = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/i/apparent_ss')).astype('float64')
discard_i = np.concatenate((discard_i, single_i))

keep_r = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/r/good')).astype('float64')
discard_r = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/r/bad')).astype('float64')
single_r = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/r/apparent_ss')).astype('float64')
discard_r = np.concatenate((discard_r, single_r))

images_y = np.concatenate((keep_y,discard_y))
images_i = np.concatenate((keep_i,discard_i))
images_r = np.concatenate((keep_r,discard_r))
labels = np.concatenate((np.ones(keep_y.shape[0]),np.zeros(discard_y.shape[0])))

plt.figure(1)
plt.title('Keep i-band')
plt.imshow(keep_i[randrange(keep_i.shape[0])])
plt.show()

plt.figure(2)
plt.title('Discard i-band')
plt.imshow(discard_i[randrange(discard_i.shape[0])])
plt.show()

#%% (1) Correcting the nan errors / Deleting the blank pictures

def cleaningCORR(images):
   counternan = 0
   for i in np.arange(images.shape[0]):
       if (np.isnan(images[i].min()) == True):
           counternan = counternan + 1
           for j in np.arange(images.shape[1]):
               for k in np.arange(images.shape[2]):
                   if (np.isnan(images[i, j, k]) == True):
                       images[i, j, k] = max( images[i, 10, 10])
   print('numbers of nan (saturated) : ' + str(counternan))

   counternan = 0
   images_clean = []
   for i in np.arange(images.shape[0]):
       if (np.isnan(images[i].min()) == False):
           images_clean.append(images[i])
       else:
           counternan = counternan + 1
   images_clean = np.array(images_clean)
   print('numbers of blanks: ' + str(counternan))
   return images_clean

print('y-band:')
images_y_clean = cleaningCORR(images_y)
print('i-band:')
images_i_clean = cleaningCORR(images_i)
print('r-band:')
images_r_clean = cleaningCORR(images_r)

#There are usually no blanks

#%% (2) We resize normalize

def standardizeFITS(img):
    sig = (np.percentile(img, 50 + 34.1) - np.percentile(img[i], 50 - 34.1)) / 2.
    norm_img = (img - np.median(img)) / sig
    return norm_img

def normalizeFITS(img):
    xmax, xmin = img.max(), img.min()
    img_nor = (img - xmin)/(xmax - xmin)
    return img_nor

def resizeANDnormalizeFITS(img):
    fimg = cv2.resize(img, (128,128) )
    fimg = normalizeFITS(fimg)
    fimg = np.reshape(fimg, (fimg.shape[0],fimg.shape[1], 1))
    return fimg

images_y_nor = []
images_i_nor = []
images_r_nor = []

for i in np.arange(images_y_clean.shape[0]):
    images_y_nor.append( resizeANDnormalizeFITS(images_y_clean[i]) )
    images_i_nor.append( resizeANDnormalizeFITS(images_i_clean[i]) )
    images_r_nor.append( resizeANDnormalizeFITS(images_r_clean[i]) )
    
images_y_nor = np.array(images_y_nor)
images_i_nor = np.array(images_i_nor)
images_r_nor = np.array(images_r_nor)

#%% (3) We create the faux colours images

images_nor_pre = np.zeros((images_y_nor.shape[0], images_y_nor.shape[1], images_y_nor.shape[2], 3))
images_nor_pre[..., 0] = np.reshape(images_y_nor, (699,128,128))
images_nor_pre[..., 1] = np.reshape(images_i_nor, (699,128,128))
images_nor_pre[..., 2] = np.reshape(images_r_nor, (699,128,128))

#%% (4) Augmenting the dataset

def gaussian_noise(picture, mean=0, variat=0.03):
    noisy = picture.copy()
    
    noise = (np.random.normal(mean, variat, picture.shape)).astype('float64')
    mask_overflow_upper = noisy + noise >= 15.0
    mask_overflow_lower = noisy + noise < 0
    noise[mask_overflow_upper] = 15.0
    noise[mask_overflow_lower] = 0
    noisy = noisy + noise
    return noisy

def colour_noise(img, channel=random.choice(['r','g','b']), ratio=random.choice([0.3,0.4,0.5,0.6])):
    assert channel in 'rgb', "Value for channel: r|g|b"
    img = img.copy()
    ci = 'rgb'.index(channel)
    img[:, :, ci] *= ratio
    return img

def translate(img, shift=3, direction='right', roll=True):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice
    return img

def augmentFITS(img):
    rot_decision = randrange(8)
    tras_decision = randrange(2)
    tras_decision_2 = randrange(2)
    noise_decision = randrange(10)
    colour_decision = randrange(10)
    colour_decision_2 = randrange(10)

    if (rot_decision == 0):
        img = img
    elif (rot_decision == 1):
        img = np.flipud(img)
    elif (rot_decision == 2):
        img = np.rot90(img)
    elif (rot_decision == 3):
        img = np.rot90(img)
        img = np.flipud(img)
    elif (rot_decision == 4):
        img = np.rot90(img, k=2)
    elif (rot_decision == 5):
        img = np.rot90(img, k=2)
        img = np.flipud(img)
    elif (rot_decision == 6):
        img = np.rot90(img, k=3)
    elif (rot_decision == 7):
        img = np.rot90(img, k=3)
        img = np.flipud(img)
    
    if (tras_decision == 0):
        traslist = ['right', 'left', 'down', 'up']
        tras_1 = random.choice(traslist)
        shift_1 = random.choice([2,4,6,8])
        img = translate(img, direction = tras_1, shift = shift_1)
        
        if (tras_decision_2 == 0):
            traslist.remove(tras_1)
            tras_2 = random.choice(traslist)
            shift_2 = random.choice([2,4,6,8])
            img = translate(img, direction = tras_2, shift = shift_2)
    
    '''
    if (colour_decision == 0):
        img = colour_noise(img)
        if (colour_decision_2 == 0):
            img  = colour_noise(img)
    '''

    if (noise_decision == 0):
        img = gaussian_noise(img)
    
    return img

mass_production = 10100
images_aug = []
labels_nor = []
for i in np.arange(mass_production):
    j = randrange(images_nor_pre.shape[0])
    images_aug.append(augmentFITS( images_nor_pre[j] ))
    labels_nor.append( labels[j] )
images_aug = np.array(images_aug)
labels_nor = np.array(labels_nor)

#%% (5) We normalize again!

def normalizeFITS(img):
    xmax, xmin = img.max(), img.min()
    img_nor = (img - xmin)/(xmax - xmin)
    return img_nor

def normalizeCHANNELFITS(img):
    img_nor_chan = np.zeros_like(img)
    for i in np.arange(img.shape[2]):
        img_nor_chan[...,i] = normalizeFITS(img[...,i])
    return img_nor_chan

def normalizeARRAY(array):
    array_nor = []
    for i in np.arange(array.shape[0]):
        array_nor.append(normalizeCHANNELFITS(array[i]))
    return np.array(array_nor)

images_nor = normalizeARRAY(images_aug)

#%% (6) Cutting the test set for the quality augmentation

images_test = images_nor[:100]
labels_test = labels_nor[:100]
images_nor = images_nor[100:]
labels_nor = labels_nor[100:]

#%% (7) We create clear the session

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import LearningRateScheduler

tf.keras.backend.clear_session()

#%% (8a) We create the CNN model (GALCLAS)

def step_decay(epoch):
 initial_lrate = 0.1
 drop = 0.5
 epochs_drop = 10.0
 lrate = initial_lrate * np.pow(drop, np.floor((1+epoch)/epochs_drop))
 return lrate

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(images_nor.shape[1],images_nor.shape[2],images_nor.shape[3]), kernel_regularizer = tf.keras.regularizers.L2(l=0.01), bias_regularizer = tf.keras.regularizers.L2(0.01)))
model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Conv2D(16, (3,3), 1, activation='relu', kernel_regularizer = tf.keras.regularizers.L2(l=0.01), bias_regularizer = tf.keras.regularizers.L2(0.01)))
model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
#lrate = LearningRateScheduler(step_decay)
#callbacks_list = [lrate]

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
model.summary()

#%% (8b) We create the CNN model (LENSbuona)

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(images_nor.shape[1],images_nor.shape[2],images_nor.shape[3])))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
model.summary()

#%% (9) We train the CNN model

hist = model.fit(images_nor, labels_nor, epochs=15, validation_split = 0.3, shuffle=True)

#%% (10) We evaluate the trend of the loss functions

plt.figure(3)
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
plt.suptitle('Loss Function', fontsize=14)
plt.legend(loc='upper left')
plt.xlabel('Epochs')
plt.show()

plt.figure(4)
plt.plot(hist.history['binary_accuracy'], color='teal', label='acc')
plt.plot(hist.history['val_binary_accuracy'], color='orange', label='val_acc')
plt.plot(hist.history['precision'], color='black', label='prec')
plt.plot(hist.history['recall'], color='blue', label='recall')
plt.suptitle('Trends', fontsize=14)
plt.legend(loc='upper left')
plt.xlabel('Epochs')
plt.show()

#%% (11) We test in on the augmented test set

labels_pred = np.array(model.predict(images_test))
labels_pred_bi = np.zeros(len(labels_pred))
for i in np.arange(len(labels_pred)):
    if labels_pred[i]<0.5:
        labels_pred_bi[i]=0
    else:
        labels_pred_bi[i]=1

plt.figure(5)
plt.suptitle('Test Results Predict', fontsize=14)
plt.hist(labels_pred, 50)
plt.show()

counter_test = 0
counter_good = 0
good = []
good_score = []
bad = []
bad_score = []

for i in np.arange(len(labels_pred_bi)):
    counter_test += abs(labels_pred_bi[i] - labels_test[i])
    if (abs(labels_pred_bi[i] - labels_test[i]) != 0):
        bad.append(images_test[i])
        bad_score.append(labels_pred[i])
    else:
        counter_good = counter_good + 1
        good.append(images_test[i])
        good_score.append(labels_pred[i])
good = np.array(good)
good_score = np.array(good_score)
bad = np.array(bad)
bad_score = np.array(bad_score)
print('numbers of errors (binary) : ' + str(counter_test))
print('numbers of matches (binary) : ' + str(counter_good))

#%% (12) Matrice di Confusione

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(labels_test,labels_pred_bi)

plt.figure(6)
plt.suptitle('Confusion Matrix', fontsize=14)
plt.imshow(confusion, cmap='hot', interpolation='nearest')
plt.xticks([0,1], ['discard','keep'])
plt.yticks([0,1], ['discard','keep'])
plt.xlabel('PREDICTED')
plt.ylabel('TRUE')
plt.text(0, 0, '%.4f' % confusion[0,0], horizontalalignment='center', verticalalignment='center', color='k')
plt.text(1, 0, '%.4f' % confusion[0,1], horizontalalignment='center', verticalalignment='center', color='w')
plt.text(0, 1, '%.4f' % confusion[1,0], horizontalalignment='center', verticalalignment='center', color='w')
plt.text(1, 1, '%.4f' % confusion[1,1], horizontalalignment='center', verticalalignment='center', color='k')
plt.show()

#%% (13) Let's save the model!

model.save("LENSQUASfitsTRI31823")