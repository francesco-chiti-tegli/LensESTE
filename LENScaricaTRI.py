# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:13:28 2023

@author: Francesco Chiti Tegli
"""

import astropy.io
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
from astropy.io import fits
import random
from random import randrange
from random import shuffle
from random import choice
import cv2

plt.rcParams['figure.dpi'] = 1000

NOfits = ['2319488154672985088.fits',
 '3472945331896957696.fits',
 '4872570853202075008.fits',
 '4873985409274483456.fits',
 '4874747890520450816.fits',
 '4877987734675447936.fits',
 '5030667431136714240.fits',
 '6793787656886649728.fits',
 '2319488154672985088.fits',
 '3481757672779443200.fits',
 '5030667431136714240.fits',
 '6793787656886649728.fits',
 '3472945331896957696.fits',
 '4872570853202075008.fits',
 '4873985409274483456.fits',
 '4874747890520450816.fits',
 '4877987734675447936.fits']
 

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

def loadFITSwithout(path):
    trainImages = []
    for r, d, f in os.walk(path):
       for file in f:
           if ((file in NOfits) == False):
               filepath = os.path.join(r, file)
               fit = fits.open(filepath)
               img = np.flipud(np.array(fit[0].data))
               img = np.reshape(img, (img.shape[0],img.shape[1], 1))
               trainImages.append(img)
    return trainImages

def loadFITSwithoutANDNAME(path):
    trainNames = []
    for r, d, f in os.walk(path):
       for file in f:
           if ((file in NOfits) == False):
               trainNames.append(file)
    return trainNames
    
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
               

cutout_y = np.array(loadFITSwithout("C:/Users/fchit/Downloads/candidates/candidates/y")).astype('float64')
cutout_i = np.array(loadFITSwithout("C:/Users/fchit/Downloads/candidates/candidates/i")).astype('float64')
cutout_r = np.array(loadFITSwithout("C:/Users/fchit/Downloads/candidates/candidates/r")).astype('float64')
cutout_name = loadFITSwithoutANDNAME("C:/Users/fchit/Downloads/candidates/candidates/y")

NOcutout = [543,1196,1213,1257,1327]

def cleaningCORR(images, images_name = cutout_name, imageNOTname = True):
   counternan = 0
   for i in np.arange(images.shape[0]):
       if (np.isnan(images[i].min()) == True):
           counternan = counternan + 1
           for j in np.arange(images.shape[1]):
               for k in np.arange(images.shape[2]):
                   if (np.isnan(images[i, j, k]) == True):
                       images[i, j, k] = np.nanmax(images[i])
   print('numbers of nan (saturated) : ' + str(counternan))

   counternan = 0
   images_clean = []
   images_name_clean = []
   print('indexes of blanks: ')
   for i in np.arange(images.shape[0]):
       if (np.isnan(images[i].min()) == False) and ((i in NOcutout) == False):
           images_clean.append(images[i])
           images_name_clean.append(images_name[i])
       else:
           counternan = counternan + 1
           print(i)
   images_clean = np.array(images_clean)
   print('numbers of blanks: ' + str(counternan))
   
   if (imageNOTname == True):
       return images_clean
   else:
       return images_name_clean

print('y-band:')
cutout_y_clean = cleaningCORR(cutout_y)
print('i-band:')
cutout_i_clean = cleaningCORR(cutout_i)
print('r-band:')
cutout_r_clean = cleaningCORR(cutout_r)

cutout_name_clean = cleaningCORR(cutout_y, images_name = cutout_name, imageNOTname = False)

#%%
def normalizeFITS(img):
    xmax, xmin = img.max(), img.min()
    img_nor = (img - xmin)/(xmax - xmin)
    return img_nor

def resizeANDnormalizeFITS(img):
    fimg = cv2.resize(img, (128,128))
    fimg = normalizeFITS(fimg)
    fimg = np.reshape(fimg, (fimg.shape[0],fimg.shape[1], 1))
    return fimg

cutout_y_nor = []
cutout_i_nor = []
cutout_r_nor = []

for i in np.arange(cutout_i_clean.shape[0]):
    cutout_y_nor.append( resizeANDnormalizeFITS(cutout_y_clean[i]) )
    cutout_i_nor.append( resizeANDnormalizeFITS(cutout_i_clean[i]) )
    cutout_r_nor.append( resizeANDnormalizeFITS(cutout_r_clean[i]) )
    
cutout_y_nor = np.array(cutout_y_nor)
cutout_i_nor = np.array(cutout_i_nor)
cutout_r_nor = np.array(cutout_r_nor)

cutout_nor = np.zeros((cutout_i_nor.shape[0], cutout_i_nor.shape[1], cutout_i_nor.shape[2], 3))
cutout_nor[..., 0] = np.reshape(cutout_y_nor, (cutout_y_nor.shape[0],128,128))
cutout_nor[..., 1] = np.reshape(cutout_i_nor, (cutout_i_nor.shape[0],128,128))
cutout_nor[..., 2] = np.reshape(cutout_r_nor, (cutout_r_nor.shape[0],128,128))

ready_model = tf.keras.models.load_model('C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/LENSQUASfitsTRI1010')

#%% Visualize the CNN model (PICTURE)
from PIL import ImageFont
import visualkeras
font = ImageFont.truetype("C:/Users/fchit/Downloads/Hack/Hack-Regular.ttf",25, encoding="unic")
visualkeras.layered_view(ready_model, spacing=5, font=font, legend=True)
#%%

ready_predict = np.ndarray.flatten(ready_model.predict(cutout_nor))
ready_predict_bi = np.zeros(len(ready_predict))
for i in np.arange(len(ready_predict)):
    if ready_predict[i]<0.5:
        ready_predict_bi[i]=0
    else:
        ready_predict_bi[i]=1

plt.figure(1)
plt.suptitle('Results TRI', fontsize=14)
plt.hist(ready_predict, 50)
plt.show()

plt.figure(2)
plt.suptitle('Over 90 TRI', fontsize=14)
eps = 0.0001
plt.hist(ready_predict, bins=100, range=(1.0-eps,1.0+eps))
plt.show()

counter_ready = 0

good = []
good_score = []
good_index = []

bad = []
bad_name = []
bad_score = []
bad_index = []

for i in np.arange(len(ready_predict_bi)):
    counter_ready += abs(ready_predict_bi[i] - 0)
    if (ready_predict_bi[i] == 0):
        bad.append(cutout_nor[i])
        bad_name.append(cutout_name_clean[i])
        bad_score.append(ready_predict[i])
        bad_index.append(i)
    else:
        good.append(cutout_nor[i])
        good_score.append(ready_predict[i])
        good_index.append(i)
        
badTOtrain = [0,4,5,9,10,14,16,17,19,26,27,31,37,40,44,45,46,47,50,51,53,57,
              61,68,69,71,72,81,84,100,103,106,108,109,110,111,112,113,114,115,
              118,120,122,123,124,125,127,129,130,132,133,134,135,136,139,140,
              142,144,147,150,152,154,155,166,169,170,171,176,177,182,185,195,
              198,200,201,202,203,211]

def extractBADtoTRAIN(badlist, namelist):
    newtrainlist = []
    for i in np.arange(len(namelist)):
        if (i in badlist):
            newtrainlist.append(namelist[i])
    return newtrainlist

badTOtrain_NAMES = extractBADtoTRAIN(badlist = badTOtrain, namelist = bad_name)

good = np.array(good)
good_score = np.array(good_score)
good_index = np.array(good_index)
bad = np.array(bad)
bad_score = np.array(bad_score)
bad_index = np.array(bad_index)

counter_ready = counter_ready / len(ready_predict_bi)
print('counter_ready = ' + str(counter_ready))
print( )

#%%
for i in np.arange(len(ready_predict)):
    if ((ready_predict[i] < 0.8) and (ready_predict[i] > 0.2)):
        print(i)
        print(ready_predict[i])
        print( )

print('---------------------------------------------------- good')
perfect = 0
perfect_array = []
perfect_names = []
almost_perfect_array = []

for i in np.arange(len(good)):
    if (good_score[i] < 0.8):
        print(i)
        print(good_score[i])
        print( )
    if (good_score[i] == 1.):
        perfect = perfect + 1
        perfect_array.append(good[i])
        perfect_names.append(cutout_name_clean[i])
    if ((good_score[i] > 0.99) and (good_score[i] < 1.)):
        almost_perfect_array.append(good[i])
        
perfect_array = np.array(perfect_array)
perfect_names = np.array(perfect_names)
almost_perfect_array = np.array(almost_perfect_array)

print('perfect = ' + str(perfect))
print( )
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ bad')
awful = 0
bad_2 = []
bad_2_score = []
for i in np.arange(len(bad)):
    if (bad_score[i] > 0.2):
        print(i)
        print(bad_score[i])
        print( )
    if (bad_score[i] == 0.):
        awful = awful + 1
    if (bad_score[i] < 0.2):
        bad_2.append(bad[i])
        bad_2_score.append(bad_score[i])

bad_2 = np.array(bad_2)
bad_2_score = np.array(bad_2_score)

print('awful = ' + str(awful))
print( )

#%% Let us confront it with the other model, made in June

model_june = tf.keras.models.load_model('C:/Users/fchit/Downloads/LENSQUAS6623')
cutout_big = []
for i in np.arange(cutout_nor.shape[0]):
    cutout_big.append(cv2.resize(cutout_nor[i], (240,240) ))
cutout_big = np.array(cutout_big)
june_predict = np.ndarray.flatten(model_june.predict(cutout_big))
june_predict_bi = np.zeros(len(june_predict))
for i in np.arange(len(june_predict_bi)):
    if june_predict[i]<0.5:
        june_predict_bi[i]=0
    else:
        june_predict_bi[i]=1

plt.figure(3)
plt.suptitle('Results JUNE', fontsize=14)
plt.hist(june_predict, 50)
plt.show()

differences_binary_good = 0
differences_binary_bad = 0
for i in np.arange(june_predict_bi.shape[0]):
    if (june_predict_bi[i] != ready_predict_bi[i]):
        if (ready_predict_bi[i] < 0.5):
            differences_binary_bad = differences_binary_bad + 1
        else:
            differences_binary_good = differences_binary_good + 1
print()
print('model JUNE differences with TRI1010')
print('bad for TRI1010: ' + str(differences_binary_bad) + ', good for TRI1010: ' + str(differences_binary_good))
print()
#%% And with the model TRI1010 boosted

model_boost = tf.keras.models.load_model('C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/LENSQUASfitsTRI1010boost')
boost_predict = np.ndarray.flatten(model_boost.predict(cutout_nor).astype('float64'))
boost_predict_bi = np.zeros(len(boost_predict))
for i in np.arange(len(boost_predict_bi)):
    if boost_predict[i]<0.5:
        boost_predict_bi[i]=0
    else:
        boost_predict_bi[i]=1

plt.figure(4)
plt.suptitle('Results BOOST', fontsize=14)
plt.hist(boost_predict, 50)
plt.show()

differences_binary_good = 0
differences_binary_bad = 0
for i in np.arange(boost_predict_bi.shape[0]):
    if (boost_predict_bi[i] != ready_predict_bi[i]):
        if (ready_predict_bi[i] < 0.5):
            differences_binary_bad = differences_binary_bad + 1
        else:
            differences_binary_good = differences_binary_good + 1
print()
print('model TRI1010 differences with TRI1010 boost')
print('bad for TRI1010: ' + str(differences_binary_bad) + ', good for TRI1010: ' + str(differences_binary_good))
print()

bad_boost = []
good_boost = []
perfect_boost = []
perfect_boost_name = []

for i in np.arange(len(boost_predict_bi)):
    if (boost_predict_bi[i] == 0):
        bad_boost.append(cutout_nor[i])
    else:
        good_boost.append(cutout_nor[i])
    if (boost_predict[i] == 1.):
        perfect_boost.append(cutout_nor[i])
        perfect_boost_name.append(cutout_name_clean[i])
bad_boost = np.array(bad_boost)
good_boost = np.array(good_boost)
perfect_boost = np.array(perfect_boost)
perfect_boost_name = np.array(perfect_boost_name)

differences_binary_good = 0
differences_binary_bad = 0
for i in np.arange(boost_predict_bi.shape[0]):
    if (boost_predict_bi[i] != june_predict_bi[i]):
        if (boost_predict_bi[i] < 0.5):
            differences_binary_bad = differences_binary_bad + 1
        else:
            differences_binary_good = differences_binary_good + 1
print()
print('model JUNE differences with TRI1010 boost')
print('bad for TRI1010 boost: ' + str(differences_binary_bad) + ', good for TRI1010 boost: ' + str(differences_binary_good))
print()

deg_array = []
deg_ignore = []

for i in np.arange(boost_predict.shape[0]):
    if ((boost_predict[i] in deg_ignore) == False):
        temp = []
        temp.append(boost_predict[i])
        deg_ignore.append(boost_predict[i])
        
        deg_score = 0
        for j in np.arange(boost_predict.shape[0]):
            if (boost_predict[j] == boost_predict[i]):
                deg_score = deg_score + 1
                
        temp.append(deg_score)
        deg_array.append(temp)
        del temp

deg_array = np.array(deg_array).astype('float64')
deg_array = deg_array[deg_array[:, 1].argsort()]
       
#%% Write a list of PERFECT boosted candidates
'''
with open ("C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/boost_perfect_candidates.txt", 'w') as fo:
   for d in perfect_boost_name:
     fo.write(str(d) + ',')
'''

#You will have to manually delete the last comma (',') in the text file

#%% Write a complete list of results of the boosted CNN

final_boost = []

for i in np.arange(boost_predict.shape[0]):
    temp_boost = []
    temp_boost.append(cutout_name_clean[i])
    temp_boost.append(boost_predict[i].astype('float64'))
    
    final_boost.append(temp_boost)
    
    del temp_boost

final_boost = np.array(final_boost)

#%%

import csv
with open("C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/boost_ALL_candidates.txt", mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',',  quotechar='"',
                                 quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerows(final_boost)

#%% I confront the different datasets composed by Francesco Guarneri

def confrontoTESTsets(path1, path2, form1, form2):
    names1 = []
    for r, d, f in os.walk(path1):
       for file in f:
          names1.append(file.strip(form1))
    
    names2 = []
    for r, d, f in os.walk(path2):
       for file in f:
          names2.append(file.strip(form2))
    
    nonexistent = []
    existent = []
    if (len(names1) >= len(names2)):
       for i in np.arange(len(names1)):
           exists = names1[i] in names2
           if (exists == False):
               nonexistent.append(names1[i])
           else:
               existent.append(names1[i])
    else:
        for i in np.arange(len(names2)):
            exists = names2[i] in names1
            if (exists == False):
                nonexistent.append(names2[i])
            else:
                existent.append(names2[i])

    return existent, nonexistent

existent, nonexistent = confrontoTESTsets( path1='C:/Users/fchit/Downloads/candidates/candidates/i', path2='C:/Users/fchit/Downloads/cutoutActualCandPS', form1='.fits', form2='.jpg' )
print('Confronto fra i due candidate set, di giugno e settembre:')
print('esistenti in entrambi i set: ' + str(len(existent)))

existent, nonexistent = confrontoTESTsets( path1='C:/Users/fchit/Downloads/candidates/candidates/i', path2='C:/Users/fchit/Downloads/candidates/i', form1='.fits', form2='.fits' )
print('Confronto fra il train set ed il candidates set (sept) in banda i:')
print('esistenti in entrambi i set: ' + str(len(existent)))

plt.figure(5)
plt.suptitle('Results', fontsize=14)
plt.hist(ready_predict, 50, alpha = 0.5, label='TRI1010')
plt.hist(june_predict, 50, alpha = 0.5, label='june')
plt.hist(boost_predict, 50, alpha = 0.5, label='TRIboost')
plt.legend()
plt.show()

#%% Generate attention maps

w = randrange(cutout_nor.shape[0])
img_prova = cutout_nor[w]

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.gradcam import Gradcam
from matplotlib import cm

replace2linear = ReplaceToLinear()

saliency = Saliency(model_boost, model_modifier=replace2linear, clone=True)

def score_function(output):
    return (output[0][0])

saliency_map = saliency(score_function, img_prova, smooth_samples=20, smooth_noise=0.20)
saliency_map = np.squeeze(saliency_map, axis=0)

gradcam = Gradcam(model_boost, model_modifier=replace2linear, clone=True)
cam = gradcam(score_function, img_prova,  penultimate_layer=-1)
cam = np.squeeze(cam, axis=0)

heatmap = np.uint8(cm.jet(cam)[..., :3] * 255)

f, ax = plt.subplots(nrows=1, ncols=3)
ax[0].set_title('Cutout' + ' ' + str(w))
ax[0].axis('off')
ax[0].imshow(img_prova)
ax[1].set_title('Saliency')
ax[1].axis('off')
ax[1].imshow(saliency_map)
ax[2].set_title('Attention Map')
ax[2].axis('off')
ax[2].imshow(img_prova)
ax[2].imshow(heatmap, cmap='jet', alpha=0.5)
plt.tight_layout()
plt.show()

print(boost_predict[w])



