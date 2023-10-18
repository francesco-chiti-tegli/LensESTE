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

def loadBOOST(boostlist = (open("C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/boost.txt", "r").read()).split(",")):
    boostImages_y = []
    boostImages_i = []
    boostImages_r = []
    for i in np.arange(len(boostlist)):
        boostpath_y = "C:/Users/fchit/Downloads/candidates/candidates/y/" + str(boostlist[i])
        boostpath_i = "C:/Users/fchit/Downloads/candidates/candidates/i/" + str(boostlist[i])
        boostpath_r = "C:/Users/fchit/Downloads/candidates/candidates/r/" + str(boostlist[i])
        boostimg_y = np.flipud(np.array((fits.open(boostpath_y))[0].data))
        boostimg_y = np.reshape(boostimg_y, (boostimg_y.shape[0],boostimg_y.shape[1], 1))
        boostimg_i = np.flipud(np.array((fits.open(boostpath_i))[0].data))
        boostimg_i = np.reshape(boostimg_i, (boostimg_i.shape[0],boostimg_i.shape[1], 1))
        boostimg_r = np.flipud(np.array((fits.open(boostpath_r))[0].data))
        boostimg_r = np.reshape(boostimg_r, (boostimg_r.shape[0],boostimg_r.shape[1], 1))
        boostImages_y.append(boostimg_y)
        boostImages_i.append(boostimg_i)
        boostImages_r.append(boostimg_r)
    return boostImages_y, boostImages_i, boostImages_r

def confrontDIRfits(path1, path2):
    names1 = []
    for r, d, f in os.walk(path1):
       for file in f:
          names1.append(file)
    
    names2 = []
    for r, d, f in os.walk(path2):
       for file in f:
          names2.append(file)
    
    nonexistent = []
    if (len(names1) >= len(names2)):
       for i in np.arange(len(names1)):
           exists = names1[i] in names2
           if (exists == False):
               nonexistent.append(names1[i])
    else:
        for i in np.arange(len(names2)):
            exists = names2[i] in names1
            if (exists == False):
                nonexistent.append(names2[i])
    

    return nonexistent

plt.rcParams['figure.dpi'] = 1000

index_single_keep = [0,1,2,13,14,15,16,17,34,35,50,51,100,101,102,
                     103,104,105,114,115,124,125,126,127,132,133]
index_single_discard = [3,4,5,6,11,12,18,19,20,21,22,23,24,25,26,
                        27,32,33,36,37,38,39,40,41,42,43,44,45,
                        46,47,48,49,52,53,56,57,58,59,60,61,66,67,68,
                        69,70,71,74,75,78,79,82,83,90,91,92,93,94,95,
                        96,97,106,107,112,113,116,117,122,123,130,131,
                        134,135,136,137,138,139,140,141,142,143]
index_single_indec = [7,8,9,10,28,29,30,31,54,55,62,63,64,65,72,73,76,
                           77,80,81,84,85,86,87,88,89,98,99,108,109,110,111,
                           118,119,120,121,128,129,144,145]

def single_choice(keep_array, discard_array, single_array, idx_keep = index_single_keep, idx_discard = index_single_discard, idx_indec = index_single_indec, dec = True, boost = True, band = 'y'):
    single_keep = []
    single_discard = []
    for i in np.arange(single_array.shape[0]):
        if ((i in idx_keep) == True):
            single_keep.append(single_array[i])
        elif ((i in idx_discard) == True):
            single_discard.append(single_array[i])
        elif (((i in idx_indec) == True) and (dec == True)):
            single_keep.append(single_array[i])
    
    if (boost == True):
        if (band == 'y'):
            boostlist = loadBOOST()[0]
        if (band == 'i'):
            boostlist = loadBOOST()[1]
        if (band == 'r'):
            boostlist = loadBOOST()[2]
        boostlist = np.array(boostlist)
        for i in np.arange(boostlist.shape[0]):
            single_keep.append(boostlist[i])
        
    single_keep = np.array(single_keep)
    single_discard = np.array(single_discard)
    
    keep_array = np.concatenate((keep_array, single_keep))
    discard_array = np.concatenate((discard_array, single_discard))
    
    return keep_array, discard_array

def removeINDEX(keep_array, index):
    keep_new_array = []
    for i in np.arange(keep_array.shape[0]):
        if ((i != index) and i != (index + 1)):
            keep_new_array.append(keep_array[i])
    keep_new_array = np.array(keep_new_array)
    return keep_new_array

#The keep 168/169 is a very faint lens which outliers

keep_y = np.array(loadFITS('C:/Users/fchit/Downloads/cutout/fits/good')).astype('float64')
discard_y = np.array(loadFITS('C:/Users/fchit/Downloads/cutout/fits/bad')).astype('float64')
single_y = np.array(loadFITS('C:/Users/fchit/Downloads/cutout/fits/apparent_single_source')).astype('float64')
keep_y = removeINDEX(keep_y,168)
keep_y, discard_y = single_choice(keep_y, discard_y, single_y, dec=True, boost=True, band='y')

keep_i = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/i/good')).astype('float64')
discard_i = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/i/bad')).astype('float64')
single_i = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/i/apparent_ss')).astype('float64')
keep_i = removeINDEX(keep_i,168)
keep_i, discard_i = single_choice(keep_i, discard_i, single_i, dec=True, boost=True, band='i')

keep_r = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/r/good')).astype('float64')
discard_r = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/r/bad')).astype('float64')
single_r = np.array(loadFITS('C:/Users/fchit/Downloads/cutout_i_r/r/apparent_ss')).astype('float64')
keep_r = removeINDEX(keep_r,168)
keep_r, discard_r = single_choice(keep_r, discard_r, single_r, dec=True, boost=True, band='r')

images_y = np.concatenate((keep_y,discard_y))
images_i = np.concatenate((keep_i,discard_i))
images_r = np.concatenate((keep_r,discard_r))
labels = np.concatenate((np.ones(keep_y.shape[0]),np.zeros(discard_y.shape[0])))

plt.figure(1)
W = randrange(keep_i.shape[0])
f, axarr = plt.subplots(1,3)
axarr[0].imshow(keep_y[W])
axarr[0].set_title('Keep y-band')
axarr[0].axis('off')
axarr[1].imshow(keep_i[W])
axarr[1].set_title('Keep r-band')
axarr[1].axis('off')
axarr[2].imshow(keep_r[W])
axarr[2].set_title('Keep r-band')
axarr[2].axis('off')
plt.show()

plt.figure(2)
Z = randrange(discard_i.shape[0])
f, axarr = plt.subplots(1,3)
axarr[0].imshow(discard_y[Z])
axarr[0].set_title('Discard y-band')
axarr[0].axis('off')
axarr[1].imshow(discard_i[Z])
axarr[1].set_title('Discard i-band')
axarr[1].axis('off')
axarr[2].imshow(discard_r[Z])
axarr[2].set_title('Discard r-band')
axarr[2].axis('off')
plt.show()

#%% (1) Correcting the nan errors / Deleting the blank pictures

def cleaningCORR(images):
   counternan = 0
   keepnan = 0
   for i in np.arange(images.shape[0]):
       if (np.isnan(images[i].min()) == True):
           counternan = counternan + 1
           if (i <= (keep_y.shape[0] -1)):
               keepnan = keepnan + 1
               print(i)
           for j in np.arange(images.shape[1]):
               for k in np.arange(images.shape[2]):
                   if (np.isnan(images[i, j, k]) == True):
                       images[i, j, k] = np.nanmax(images[i])
   print('number of nan (saturated) : ' + str(counternan))
   print('number of keep with nan (keep saturated) : ' + str(keepnan))

   counternan = 0
   images_clean = []
   print('indexes of blanks: ')
   for i in np.arange(images.shape[0]):
       if (np.isnan(images[i].min()) == False):
           images_clean.append(images[i])
       else:
           counternan = counternan + 1
           print(i)
   images_clean = np.array(images_clean)
   print('number of blanks: ' + str(counternan))
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

#%% (3) We create the faux colours images

images_nor_pre = np.zeros((images_y_nor.shape[0], images_y_nor.shape[1], images_y_nor.shape[2], 3))
images_nor_pre[..., 0] = np.reshape(images_y_nor, (images_y_nor.shape[0],images_y_nor.shape[1],images_y_nor.shape[2]))
images_nor_pre[..., 1] = np.reshape(images_i_nor, (images_y_nor.shape[0],images_y_nor.shape[1],images_y_nor.shape[2]))
images_nor_pre[..., 2] = np.reshape(images_r_nor, (images_y_nor.shape[0],images_y_nor.shape[1],images_y_nor.shape[2]))

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
    if (epoch < 10):
        return 0.001
    if ((epoch >= 10) and (epoch < 20)):
        return 0.00050
    if (epoch >= 20) and (epoch < 30):
        return 0.00005
    if (epoch >= 30):
        return 0.00001
    
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3), padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
tf.keras.layers.BatchNormalization()
model.add(Dropout(0.4))
model.add(Conv2D(32, (3,3), activation='relu', padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
tf.keras.layers.BatchNormalization()
model.add(Dropout(0.4))
model.add(Conv2D(64, (3,3), activation='relu', padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
tf.keras.layers.BatchNormalization()
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(16, activation='relu'))
tf.keras.layers.BatchNormalization()
model.add(Dropout(0.4))
model.add(Dense(4, activation='relu'))
tf.keras.layers.BatchNormalization()
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

Learning_scheduler = tf.keras.callbacks.LearningRateScheduler(LearningRate)

#%%
hist = model.fit(images_nor, labels_nor, epochs=40, validation_data=(images_val, labels_val), shuffle=True, callbacks=[Learning_scheduler])
#%% (8b) We create the CNN model (LENSbuona)

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

#%% (9) We train the CNN model

hist = model.fit(images_nor, labels_nor, epochs=160, validation_data=(images_val, labels_val), shuffle=True)


#%%
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3), padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
tf.keras.layers.BatchNormalization()
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu', padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
tf.keras.layers.BatchNormalization()
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(16, activation='relu'))
tf.keras.layers.BatchNormalization()
model.add(Dense(1, activation='sigmoid'))
tf.keras.layers.BatchNormalization()

model.compile(optimizer=Adam(learning_rate=0.00001), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

hist = model.fit(images_nor, labels_nor, epochs=80, validation_data=(images_val, labels_val), shuffle=True)

#%% (10) We evaluate the trend of the loss functions

plt.figure(3)
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
plt.suptitle('Loss Function', fontsize=14)
plt.legend(loc='upper left')
plt.xlabel('Epochs')
plt.show()

plt.figure(4)
plt.plot(hist.history['accuracy'], color='teal', label='acc')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_acc')
plt.suptitle('Accuracy', fontsize=14)
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

model.save("C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/LENSQUASfitsTRI1010boost")