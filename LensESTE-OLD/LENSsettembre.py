# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:47:54 2024

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
import pathlib

plt.rcParams['figure.dpi'] = 1000

def loadFITS(path):
  trainImages = []
  for r, d, f in os.walk(path):
     for file in f:
        filepath = os.path.join(r, file)
        if (pathlib.Path(filepath).suffix == '.fits'):
            fit = fits.open(filepath)
            img = np.flipud(np.array(fit[0].data))
            img = np.reshape(img, (img.shape[0],img.shape[1], 1))
            trainImages.append(img)
  return trainImages

def loadImages(pathk,pathd):
    keep = np.array(loadFITS(pathk)).astype('float64')
    discard = np.array(loadFITS(pathd)).astype('float64')
    images = np.concatenate((keep,discard))
    labels = np.concatenate((np.ones(keep.shape[0]),np.zeros(discard.shape[0])))
    return images, labels

images_y = loadImages('C:/Users/fchit/Downloads/KnownLensesAndDoubles/y', 'C:/Users/fchit/Downloads/cutout/fits/bad')[0]
images_i = loadImages('C:/Users/fchit/Downloads/KnownLensesAndDoubles/i', 'C:/Users/fchit/Downloads/cutout_i_r/i/bad')[0]
images_r = loadImages('C:/Users/fchit/Downloads/KnownLensesAndDoubles/r', 'C:/Users/fchit/Downloads/cutout_i_r/r/bad')[0]

labels = loadImages('C:/Users/fchit/Downloads/KnownLensesAndDoubles/y', 'C:/Users/fchit/Downloads/cutout/fits/bad')[1]
counterk = 0
counterd = 0
for i in np.arange(labels.shape[0]):
    if (labels[i] == 1):
        counterk = counterk + 1
    else:
        counterd = counterd + 1

#%% (1) Correcting the nan errors

def cleaningCORR(Images, Labels):
   counternan = 0
   keepnan = 0
   for i in np.arange(Images.shape[0]):
       if (np.isnan(Images[i].min()) == True):
           counternan = counternan + 1
           if (i <= (counterk -1)):
               keepnan = keepnan + 1
               print(i)
           for j in np.arange(Images.shape[1]):
               for k in np.arange(Images.shape[2]):
                   if (np.isnan(Images[i, j, k]) == True):
                       Images[i, j, k] = np.nanmax(Images[i])
   print('number of nan (saturated) : ' + str(counternan))
   print('number of keep with nan (keep saturated) : ' + str(keepnan))

   #Seems like that the 294 y-band is completely blank

   counternan = 0
   Images_clean = []
   Labels_clean = []
   print('indexes of blanks: ')
   for i in np.arange(Images.shape[0]):
       if ((np.isnan(Images[i].min()) == False) and (i != 294)):
           Images_clean.append(Images[i])
           Labels_clean.append(Labels[i])
       else:
           counternan = counternan + 1
           print(i)
   Images_clean = np.array(Images_clean)
   Labels_clean = np.array(Labels_clean)
   print('number of blanks: ' + str(counternan))
   print()
   return Images_clean, Labels_clean

print('y-band:')
images_y_clean = cleaningCORR(images_y,labels)[0]
print('i-band:')
images_i_clean = cleaningCORR(images_i,labels)[0]
print('r-band:')
images_r_clean = cleaningCORR(images_r,labels)[0]

labels_clean = cleaningCORR(images_y,labels)[1]

del images_y, images_i, images_r, labels

#%% (2) We resize normalize

def standardizeFITS(img):
    sig = (np.percentile(img, 50 + 34.1) - np.percentile(img[i], 50 - 34.1)) / 2.
    norm_img = (img - np.median(img)) / sig
    return norm_img

def normalizeFITS(img):
    xmax, xmin = img.max(), img.min()
    img_nor = (img - xmin)/(xmax - xmin)
    return img_nor

def normalizeFITS_3chan(img0, img1, img2):
    xmax = max(img0.max(), img1.max(), img2.max())
    xmin = min(img0.min(), img1.min(), img2.min())
    img_nor0 = (img0 - xmin)/(xmax - xmin)
    img_nor1 = (img1 - xmin)/(xmax - xmin)
    img_nor2 = (img2 - xmin)/(xmax - xmin)
    return img_nor0, img_nor1, img_nor2

def resizeANDnormalizeFITS(img):
    fimg = cv2.resize(img, (128,128) )
    fimg = normalizeFITS(fimg)
    fimg = np.reshape(fimg, (fimg.shape[0],fimg.shape[1], 1))
    return fimg

def resizeANDnormalizeFITS_3chan(img0, img1, img2):
    fimg0 = cv2.resize(img0, (128,128) )
    fimg1 = cv2.resize(img1, (128,128) )
    fimg2 = cv2.resize(img2, (128,128) )
    fimg0, fimg1, fimg2 = normalizeFITS_3chan(fimg0,fimg1, fimg2)
    fimg0 = np.reshape(fimg0, (fimg0.shape[0],fimg0.shape[1], 1))
    fimg1 = np.reshape(fimg1, (fimg1.shape[0],fimg1.shape[1], 1))
    fimg2 = np.reshape(fimg2, (fimg2.shape[0],fimg2.shape[1], 1))
    return fimg0, fimg1, fimg2

def DEresizeANDnormalizeFITS_3chan(img0, img1, img2):
    fimg0 = cv2.resize(img0, (48,48) )
    fimg1 = cv2.resize(img1, (48,48) )
    fimg2 = cv2.resize(img2, (48,48) )
    fimg0, fimg1, fimg2 = normalizeFITS_3chan(fimg0,fimg1, fimg2)
    fimg0 = np.reshape(fimg0, (fimg0.shape[0],fimg0.shape[1], 1))
    fimg1 = np.reshape(fimg1, (fimg1.shape[0],fimg1.shape[1], 1))
    fimg2 = np.reshape(fimg2, (fimg2.shape[0],fimg2.shape[1], 1))
    return fimg0, fimg1, fimg2

images_y_nor = []
images_i_nor = []
images_r_nor = []

for i in np.arange(images_y_clean.shape[0]):
    images_y_nor.append( resizeANDnormalizeFITS_3chan(images_y_clean[i],images_i_clean[i],images_r_clean[i])[0] )
    images_i_nor.append( resizeANDnormalizeFITS_3chan(images_y_clean[i],images_i_clean[i],images_r_clean[i])[1] )
    images_r_nor.append( resizeANDnormalizeFITS_3chan(images_y_clean[i],images_i_clean[i],images_r_clean[i])[2] )
    
images_y_nor = np.array(images_y_nor)
images_i_nor = np.array(images_i_nor)
images_r_nor = np.array(images_r_nor)

del images_y_clean, images_i_clean, images_r_clean

#%% (3) We create the faux colours images

images_nor_pre = np.zeros((images_y_nor.shape[0], images_y_nor.shape[1], images_y_nor.shape[2], 3))
images_nor_pre[..., 0] = np.reshape(images_y_nor, (images_y_nor.shape[0],images_y_nor.shape[1],images_y_nor.shape[2]))
images_nor_pre[..., 1] = np.reshape(images_i_nor, (images_y_nor.shape[0],images_y_nor.shape[1],images_y_nor.shape[2]))
images_nor_pre[..., 2] = np.reshape(images_r_nor, (images_y_nor.shape[0],images_y_nor.shape[1],images_y_nor.shape[2]))

del images_y_nor, images_i_nor, images_r_nor

#%% (4) Augmenting the dataset

def noisy(image, noise_typ=random.choice(['gauss','poisson'])):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = random.choice([0.0005,0.00075,0.001,0.00125])
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      noise = np.random.poisson(image)
      noisy = image + 0.05* noise 
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss * 0.1
      return noisy

def gaussian_noise(picture):
    row, col, ch= picture.shape
    mean=0
    variat=random.choice([0.1,0.2,0.3,0.4,0.5])
    sigma = variat**0.5
    noise = np.random.normal(mean,sigma,(row,col,ch))
    noise = noise.reshape(row,col,ch)
    noisy = picture + noise
    return noisy

def poisson(picture):
    vals = len(np.unique(picture))
    vals = 2 ** np.ceil(np.log2(picture))
    noisy = np.random.poisson(picture * vals) / float(vals)
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
    noise_decision = randrange(3)
    #colour_decision = randrange(10)
    #colour_decision_2 = randrange(10)

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
        shift_1 = random.choice([1,2,3,4])
        img = translate(img, direction = tras_1, shift = shift_1)
        
        if (tras_decision_2 == 0):
            traslist.remove(tras_1)
            tras_2 = random.choice(traslist)
            shift_2 = random.choice([1,2,3,4])
            img = translate(img, direction = tras_2, shift = shift_2)
    
    '''
    if (colour_decision == 0):
        img = colour_noise(img)
        if (colour_decision_2 == 0):
            img  = colour_noise(img)
    '''

    if (noise_decision == 0):
        img = noisy(img)
    
    return img

mass_production = 9000
images_aug = []
labels_nor = []
for i in np.arange(mass_production):
    j = randrange(images_nor_pre.shape[0])
    images_aug.append(augmentFITS( images_nor_pre[j] ))
    labels_nor.append( labels_clean[j] )
images_aug = np.array(images_aug)
labels_nor = np.array(labels_nor)

#%% (5) We normalize again!

def normalizeCHANNELFITS(img):
    img_nor_chan = np.zeros_like(img)
    for i in np.arange(img.shape[2]):
        img_nor_chan[...,i] = normalizeFITS(img[...,i])
    return img_nor_chan

def normalize3_CHANNELFITS(img):
    img_nor_chan = np.zeros_likr(img)
    xmax = max(img[...,0].max(), img[...,1].max(), img[...,2].max())
    xmin = min(img[...,0].min(), img[...,1].min(), img[...,2].min())
    for i in np.arange(img.shape[2]):
        img_nor_chan[...,i] = (img[...,i] - xmin)/(xmax - xmin)
    return img_nor_chan

def normalizeARRAY(array):
    array_nor = []
    for i in np.arange(array.shape[0]):
        array_nor.append(normalizeCHANNELFITS(array[i]))
    return np.array(array_nor)

def normalize3_CHANNELARRAY(array):
    array_nor = []
    for i in np.arange(array.shape[0]):
        array_nor.append(normalizeCHANNELFITS(array[i]))
    return np.array(array_nor)

images_nor = normalize3_CHANNELARRAY(images_aug)

del images_aug

#%% (6) Cutting the test set for the quality augmentation
from sklearn.model_selection import train_test_split
images_test = images_nor[:1000]
labels_test = labels_nor[:1000]
images_nor = images_nor[1000:]
labels_nor = labels_nor[1000:]

images_nor, images_val, labels_nor, labels_val = train_test_split(images_nor, labels_nor,test_size=0.25, random_state=42)

#%% (7) We create and clear the session

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

tf.keras.backend.clear_session()

#%% (8a) We create the CNN model (GALCLAS)


def LearningRate(epoch):
    if (epoch < 5):
        return 0.001
    if ((epoch >= 5) and (epoch < 10)):
        return 0.0002
    if (epoch >= 10) and (epoch < 15):
        return 0.00004
    if (epoch >= 15) and (epoch < 20):
        return 0.000008
    if (epoch >= 20) and (epoch < 30):
        return 0.000004
    if (epoch >= 30):
        return 0.000004
    
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3), padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
tf.keras.layers.BatchNormalization()
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu', padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
tf.keras.layers.BatchNormalization()
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu', padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
tf.keras.layers.BatchNormalization()
model.add(Dropout(0.2))
model.add(Conv2D(128, (3,3), activation='relu', padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
tf.keras.layers.BatchNormalization()
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(8, activation='relu'))
tf.keras.layers.BatchNormalization()
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
tf.keras.layers.BatchNormalization()
model.add(Dense(1, activation='sigmoid'))
tf.keras.layers.BatchNormalization()

model.compile(optimizer=Adam(), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

Learning_scheduler = tf.keras.callbacks.LearningRateScheduler(LearningRate)

#%% (9)
hist = model.fit(images_nor, labels_nor, epochs=120, validation_data=(images_val, labels_val), shuffle=True, callbacks=[Learning_scheduler])

#%% (10) We evaluate the trend of the loss functions

plt.figure(3)
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
plt.suptitle('Loss Function', fontsize=14)
plt.legend()
plt.xlabel('Epochs')
plt.show()

plt.figure(4)
plt.plot(hist.history['accuracy'], color='teal', label='acc')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_acc')
plt.suptitle('Accuracy', fontsize=14)
plt.legend()
plt.xlabel('Epochs')
plt.show()

#historygraph = [hist.history['loss'],hist.history['val_loss'],hist.history['accuracy'],hist.history['val_accuracy']]
#np.savetxt('C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/LENSseptember120history', historygraph, delimiter=",", newline = "\n")

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

#%% (12) Confusion matrix

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

#model.save('C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/LENSseptember120')