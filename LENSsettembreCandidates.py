# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:30:16 2024

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
import pandas as pd

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
#Nofits are the cutouts fits not matching in the three bands

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

cutout_y = np.array(loadFITSwithout("C:/Users/fchit/Downloads/candidates/candidates/y")).astype('float64')
cutout_i = np.array(loadFITSwithout("C:/Users/fchit/Downloads/candidates/candidates/i")).astype('float64')
cutout_r = np.array(loadFITSwithout("C:/Users/fchit/Downloads/candidates/candidates/r")).astype('float64')
cutout_name = loadFITSwithoutANDNAME("C:/Users/fchit/Downloads/candidates/candidates/y")

NOcutout = [543,1196,1213,1257,1327]
#Nocutouts are the cutouts with at least one blank band

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

del cutout_y, cutout_y_clean, cutout_y_nor, cutout_i, cutout_i_clean, cutout_i_nor, cutout_r, cutout_r_clean, cutout_r_nor, cutout_name

#%% (1) Opening the boosted model compiled in late 2023/eraly 2024, named "TRI1010boost"

boost_model = tf.keras.models.load_model('C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/LENSQUASfitsTRI1010boost')
boost_predict = np.ndarray.flatten(boost_model.predict(cutout_nor).astype('float64'))
boost_predict_bi = np.zeros(len(boost_predict))
for i in np.arange(len(boost_predict_bi)):
    if boost_predict[i]<0.5:
        boost_predict_bi[i]=0
    else:
        boost_predict_bi[i]=1

boost_bad = []
boost_good = []
boost_perfect = []
boost_perfect_name = []

for i in np.arange(len(boost_predict_bi)):
    if (boost_predict_bi[i] == 0):
        boost_bad.append(cutout_nor[i])
    else:
        boost_good.append(cutout_nor[i])
    if (boost_predict[i] == 1.):
        boost_perfect.append(cutout_nor[i])
        boost_perfect_name.append(cutout_name_clean[i])
boost_bad = np.array(boost_bad)
boost_good = np.array(boost_good)
boost_perfect = np.array(boost_perfect)
boost_perfect_name = np.array(boost_perfect_name)

boost_results = [boost_bad,boost_good,boost_perfect,boost_perfect_name]
del boost_bad, boost_good, boost_perfect, boost_perfect_name

tf.keras.backend.clear_session()

#%% (2) Opening the new model compiled in september 2024

sept_model = tf.keras.models.load_model('C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/LENSseptember120')
sept_predict = np.ndarray.flatten(sept_model.predict(cutout_nor).astype('float64'))
sept_predict_bi = np.zeros(len(sept_predict))
for i in np.arange(len(sept_predict_bi)):
    if sept_predict[i]<0.5:
        sept_predict_bi[i]=0
    else:
        sept_predict_bi[i]=1

sept_bad = []
sept_good = []
sept_perfect = []
sept_perfect_name = []

for i in np.arange(len(sept_predict_bi)):
    if (sept_predict_bi[i] == 0):
        sept_bad.append(cutout_nor[i])
    else:
        sept_good.append(cutout_nor[i])
    if (sept_predict[i] == 1.):
        sept_perfect.append(cutout_nor[i])
        sept_perfect_name.append(cutout_name_clean[i])
sept_bad = np.array(sept_bad)
sept_good = np.array(sept_good)
sept_perfect = np.array(sept_perfect)
sept_perfect_name = np.array(sept_perfect_name)

sept_results = [sept_bad,sept_good,sept_perfect,sept_perfect_name]
del sept_bad, sept_good, sept_perfect, sept_perfect_name

tf.keras.backend.clear_session()

#%% (4) Let's compare the two models

plt.figure(3)
plt.suptitle('Results comparison', fontsize=14)
plt.hist(sept_predict, 50, alpha = 0.5, label='september120')
plt.hist(boost_predict, 50, alpha = 0.5, label='TRI1010boost')
plt.legend()
plt.show()

differences_binary_good = 0
differences_binary_bad = 0
for i in np.arange(cutout_nor.shape[0]):
    if (sept_predict_bi[i] != boost_predict_bi[i]):
        if (sept_predict_bi[i] < 0.5):
            differences_binary_bad = differences_binary_bad + 1
        else:
            differences_binary_good = differences_binary_good + 1
print()
print('Differences good/bad between the models "september120" and "TRI1010boost"')
print('bad for september120: ' + str(differences_binary_bad) + ', good for september120: ' + str(differences_binary_good))
print()
del differences_binary_bad, differences_binary_good

perfect_differences = 0
for i in np.arange(sept_results[3].shape[0]):
    if (sept_results[3][i] != boost_results[3][i]):
        perfect_differences = perfect_differences + 1
print()
print('Differences perfects between the models "september120" and "TRI1010boost"')
print('perfect for september120 but not for TRI1010boost: ' + str(perfect_differences))
print()
del perfect_differences

#%% (5) Wrtiting a list of perfect candidates and a list of all candidates with their scores

'''
with open ("C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/september_perfect_candidates.txt", 'w') as fo:
   for d in sept_results[3]:
     fo.write(str(d) + ',')
del fo, d
'''

'''
with open ("C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/september_score_candidates.txt", 'w') as fo:
   for i in np.arange(len(cutout_name_clean)):
     fo.write(str(cutout_name_clean[i]) + ',' + str(sept_predict[i]) +  '\n')
'''


#Last comma (',') in the text files needs to be manually deleted after writing

#%% (6) Let us investigate further with the known lenses
# To verify:
#3586098513053631232.fits

with fits.open("C:/Users/fchit/Downloads/0_MatchedWithGaia_Metadata.fits") as hdulist:
    df = pd.DataFrame(hdulist[1].data)

known_name = np.array(df.source_id)

known_name_fits = []
for i in np.arange(known_name.shape[0]):
    known_name_fits.append(known_name[i].astype('str') + '.fits')
    
cutout_name_clean_set = set(cutout_name_clean)
matching = [[element, cutout_name_clean.index(element)] for element in known_name_fits if element in cutout_name_clean_set]
del known_name, cutout_name_clean_set

non_zero = 0
matching_scores = []
for i in np.arange(len(matching)):
    if (sept_predict[matching[i][1]] > 0.5):
        non_zero = non_zero + 1
        matching_scores.append(sept_predict[matching[i][1]])
    if (sept_predict[matching[i][1]] < 0.8):
        print(matching[i][1])
print('Elements both in Known and in Candidates: ' + str(len(matching)) + ' , of which with positives: ' + str(non_zero))
del non_zero
matching_scores = np.array(matching_scores)
del i

plt.figure(4)
plt.suptitle('Matched elements in Known and in Candidates', fontsize=14)
plt.hist(matching_scores, 50)
plt.show()

#%% (7) Let us write a list of the scores of the known lenses' score
'''
with open ("C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/september_known_score.txt", 'w') as fo:
   for i in np.arange(len(matching)):
     fo.write(str(matching[i][0]) + ',' + str(matching_scores[i]) +  '\n')
'''