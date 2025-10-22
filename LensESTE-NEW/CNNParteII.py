# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 12:59:55 2025

@author: Francesco Chiti Tegli
"""

import tensorflow as tf
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt
import os
from astropy.io import fits
import random
from random import randrange
from random import shuffle
from random import choice
import cv2
import pandas as pd

base_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/BaseDirGuarneri/"
work_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/"
NED_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/NEDGuarneri"
SIMBAD_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/SIMBADGuarneri"

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


#%% (1) Open names and NED/SIMBAD classes

with fits.open(os.path.join(base_path, "0_MatchedWithGaia_Metadata.fits")) as hdulist:
    df2 = pd.DataFrame(hdulist[1].data)
ned_simbad = pd.read_csv(os.path.join(work_path, "Catalogue_after_queries.csv"))

known_name = np.array(df2.source_id)

known_name_fits = []
for i in np.arange(known_name.shape[0]):
    known_name_fits.append(known_name[i].astype('str') + '.fits')
    
cutout_name_clean_set = set(cutout_name_clean)
matching = [[element, cutout_name_clean.index(element)] for element in known_name_fits if element in cutout_name_clean_set]
del cutout_name_clean_set
cutout_name_clean_int = np.array([int(name.replace('.fits', '')) for name in cutout_name_clean], dtype=np.int64)
print()

coords = SkyCoord(ra=ned_simbad["ra"].values*u.deg, dec=ned_simbad["dec"].values*u.deg)
xyz = np.vstack([coords.cartesian.x, coords.cartesian.y, coords.cartesian.z]).T
tree = cKDTree(xyz)

r = np.deg2rad(4.95635/3600)
neighbors = tree.query_ball_tree(tree, r)

labels = -np.ones(len(ned_simbad), dtype=int)
group_id = 0
for i in range(len(ned_simbad)):
    if labels[i] == -1:
        stack = [i]
        while stack:
            idx = stack.pop()
            if labels[idx] == -1:
                labels[idx] = group_id
                for j in neighbors[idx]:
                    if labels[j] == -1:
                        stack.append(j)
        group_id += 1

ned_simbad['group_id'] = labels
group_sizes = ned_simbad['group_id'].value_counts()
multi_object_groups = group_sizes[group_sizes >= 2].index
ned_simbad_multi = ned_simbad[ned_simbad['group_id'].isin(multi_object_groups)].copy()

print("Total connected components (all sizes):", ned_simbad['group_id'].nunique())
print("Groups with size >= 2:", ((group_sizes.sort_index()) >= 2).sum())
print("Distribution (size -> n_groups):")
print((group_sizes.sort_index()).value_counts().sort_index())

valid_classes = {'Unknown', 'Unclas IR/UV/X', 'Quasar/AGN'}
def classify_group(sub):
    ned_vals = sub['class_ned'].fillna('Unknown').str.strip()
    simbad_vals = sub['class_simbad'].fillna('Unknown').str.strip()
    
    all_flags = []
    only_unknown_overridden = True
    other_star_flags = []
    
    for ned, simbad in zip(ned_vals, simbad_vals):
        if ned == 'Quasar/AGN' or simbad == 'Quasar/AGN':
            all_flags.append('Q')
            only_unknown_overridden = False
        elif ned in valid_classes and simbad in valid_classes:
            all_flags.append('I')
            only_unknown_overridden = False
        elif ned not in valid_classes and simbad not in valid_classes:
            all_flags.append('1QT')
            only_unknown_overridden = False
        else:
            if (ned not in valid_classes and simbad in ['Unknown','Unclas IR/UV/X']) or \
               (simbad not in valid_classes and ned in ['Unknown','Unclas IR/UV/X']):
                all_flags.append('O')
            else:
                all_flags.append('O')
                only_unknown_overridden = False

        if (ned not in valid_classes and simbad == 'Star') or (simbad not in valid_classes and ned == 'Star'):
            other_star_flags.append(True)
        else:
            other_star_flags.append(False)
    
    if all(other_star_flags):
        return '0Q'
    
    if only_unknown_overridden:
        return '0Q'
    
    if all(f == 'Q' for f in all_flags):
        return '2QP'
    
    if 'Q' in all_flags and all(f in ['Q','I','O'] for f in all_flags):
        return '2QI'
    
    if all(f == 'I' for f in all_flags):
        return '2QU'
    
    if all(f in ['I','O'] for f in all_flags) and not 'Q' in all_flags:
        ned_has_valid = any(ned in valid_classes - {'Unknown','Unclas IR/UV/X'} for ned in ned_vals)
        ned_has_unknown = any(ned in ['Unknown','Unclas IR/UV/X'] for ned in ned_vals)
        if ned_has_valid and ned_has_unknown:
            return '2QU'
        elif all(ned in ['Unknown','Unclas IR/UV/X'] for ned in ned_vals):
            return '2QU'
        else:
            return '2QI'
    
    if '1QT' in all_flags:
        return '1QT'
    
    return '0Q'

group_classes = ned_simbad_multi.groupby('group_id').apply(classify_group)
ned_simbad_multi = ned_simbad_multi.merge(group_classes.rename('class_lens'), on='group_id', how='left')

valid_lens_classes = {'2QP', '2QI', '2QU'}
valid_groups = ned_simbad_multi.loc[ned_simbad_multi['class_lens'].isin(valid_lens_classes), 'group_id'].unique()
ned_simbad_clean = ned_simbad_multi[ned_simbad_multi['group_id'].isin(valid_groups)]
ned_simbad_clean = ned_simbad_clean[ned_simbad_clean['source_id'].isin(cutout_name_clean_int)]

print("Total multi-object groups classified:", len(multi_object_groups))
print("Total objects after class_lens filtering:", len(ned_simbad_clean))
print("Unique fields in cleaned sample:", ned_simbad_clean['group_id'].nunique())
print("Distribution of lens classes in cleaned sample:")
print(ned_simbad_clean['class_lens'].value_counts())
print()

good_ids = ned_simbad_clean['source_id'].astype(np.int64).values
cutout_name_clean_int = np.array([int(name.replace('.fits', '')) for name in cutout_name_clean], dtype=np.int64)
good_indices = [i for i, name in enumerate(cutout_name_clean_int) if name in good_ids]
good_cutouts = cutout_nor[good_indices]
print(f"Final number of cutouts (after NED/SIMBAD match): {len(good_cutouts)}")

#%% (2) Feed the cutouts to the CNN

sept_model = tf.keras.models.load_model('C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/CNN/LENSseptember120')
sept_predict = np.ndarray.flatten(sept_model.predict(good_cutouts).astype('float64'))
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
        sept_bad.append(good_cutouts[i])
    else:
        sept_good.append(good_cutouts[i])
    if (sept_predict[i] == 1.):
        sept_perfect.append(good_cutouts[i])
        sept_perfect_name.append(cutout_name_clean[i])
sept_bad = np.array(sept_bad)
sept_good = np.array(sept_good)
sept_perfect = np.array(sept_perfect)
sept_perfect_name = np.array(sept_perfect_name)
sept_results = [sept_bad,sept_good,sept_perfect,sept_perfect_name]
del sept_bad, sept_good, sept_perfect, sept_perfect_name
tf.keras.backend.clear_session()

n_tot = len(sept_predict)
n_lens = int(np.sum(sept_predict_bi))
n_non_lens = n_tot - n_lens
print(f"Total objects: {n_tot}")
print(f"Lenses: {n_lens} ({100*n_lens/n_tot:.1f}%)")
print(f"Non-lenses: {n_non_lens} ({100*n_non_lens/n_tot:.1f}%)")
print()

class_map = dict(zip(ned_simbad_clean['source_id'], ned_simbad_clean['class_lens']))
good_cutout_names = [cutout_name_clean[i] for i in good_indices]
cutout_classes = [class_map.get(int(name.replace('.fits', '')), 'Other') for name in good_cutout_names]
scores_2qp = [sept_predict[i] for i, c in enumerate(cutout_classes) if c == '2QP']
scores_2qi = [sept_predict[i] for i, c in enumerate(cutout_classes) if c == '2QI']
scores_2qu = [sept_predict[i] for i, c in enumerate(cutout_classes) if c == '2QU']

fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
axes[0].hist(sept_predict, bins=30, color='grey', alpha=0.6)
axes[0].axvline(0.5, color='red', linestyle='--')
axes[0].set_title('All candidates')
axes[1].hist(scores_2qp, bins=30, color='orange', alpha=0.7)
axes[1].axvline(0.5, color='red', linestyle='--')
axes[1].set_title('2QP')
axes[2].hist(scores_2qi, bins=30, color='royalblue', alpha=0.7)
axes[2].axvline(0.5, color='red', linestyle='--')
axes[2].set_title('2QI')
axes[3].hist(scores_2qu, bins=30, color='mediumseagreen', alpha=0.7)
axes[3].axvline(0.5, color='red', linestyle='--')
axes[3].set_title('2QU')
for ax in axes:
    ax.set_yscale('log')
    ax.set_ylabel('Count')
axes[-1].set_xlabel('CNN score')
plt.tight_layout()
plt.show()

def count_good_bad(scores, threshold=0.5):
    scores = np.array(scores)
    n_good = np.sum(scores >= threshold)
    n_bad = np.sum(scores < threshold)
    return n_good, n_bad
n_2qp_good, n_2qp_bad = count_good_bad(scores_2qp)
n_2qi_good, n_2qi_bad = count_good_bad(scores_2qi)
n_2qu_good, n_2qu_bad = count_good_bad(scores_2qu)
print(f"2QP: {n_2qp_good} good, {n_2qp_bad} bad")
print(f"2QI: {n_2qi_good} good, {n_2qi_bad} bad")
print(f"2QU: {n_2qu_good} good, {n_2qu_bad} bad")


#%% (3) Let us investigate further with the known lenses
# To verify:
#3586098513053631232.fits

known_ids = np.array(df2.source_id, dtype=np.int64)
good_cutout_ids = np.array([int(cutout_name_clean[i].replace('.fits','')) for i in good_indices], dtype=np.int64)

matching_indices = [i for i, gid in enumerate(good_cutout_ids) if gid in known_ids]
matching_scores = sept_predict[matching_indices]

n_positive = np.sum(matching_scores > 0.5)
print(f'Elements both in Known and in Candidates: {len(matching_indices)}, of which positives (>0.5): {n_positive}')

plt.figure(figsize=(6,4))
plt.suptitle('Matched Known and Candidates')
plt.hist(matching_scores, bins=50, color='royalblue')
plt.xlabel('CNN score')
plt.ylabel('Count')
plt.show()

#%% (4) Saving the post-CNN list

positive_indices = [i for i, score in enumerate(sept_predict) if score >= 0.5]

positive_names = [good_cutout_names[i] for i in positive_indices]
positive_ids = [good_ids[i] for i in positive_indices]
positive_classes = [cutout_classes[i] for i in positive_indices]
positive_scores = [sept_predict[i] for i in positive_indices]

df_cnn = pd.DataFrame({
    'cutout_name': positive_names,
    'source_id': positive_ids,
    'class_lens': positive_classes,
    'cnn_score': positive_scores})
ned_subset = ned_simbad_clean.set_index('source_id')
df_cnn['ra'] = [ned_subset.loc[sid, 'ra'] for sid in positive_ids]
df_cnn['dec'] = [ned_subset.loc[sid, 'dec'] for sid in positive_ids]

df_cnn.to_csv(os.path.join(work_path, "Catalogue_after_CNN.csv"), index=False)
print(f"Saved {len(df_cnn)} positive candidates")
