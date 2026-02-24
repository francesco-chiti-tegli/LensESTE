# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 12:59:55 2025

@author: Francesco Chiti Tegli
"""

import tensorflow as tf
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
from itertools import combinations
import astropy.units as u
from astroquery.sdss import SDSS
import time
from astroquery.vizier import Vizier
from matplotlib import pyplot as plt
import os
from astropy.io import fits
from matplotlib.gridspec import GridSpec
from astroquery.simbad import Simbad
from astropy.table import Table
import cv2
import pandas as pd

base_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/BaseDirGuarneri/"
work_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/"
NED_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/NEDGuarneri"
SIMBAD_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/SIMBADGuarneri"
DLA_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/DLASpectraGuarneri"

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
print()
print("Original number of lenses (regardless of cutout presence):")
print(f"2QU: {np.sum(group_classes == '2QU')}")
print(f"2QP: {np.sum(group_classes == '2QP')}")
print(f"2QI: {np.sum(group_classes == '2QI')}")
print(f"Other: {np.sum(~np.isin(group_classes, ['2QP', '2QI', '2QU']))}")
print()
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
print()

n_tot = len(good_cutouts)
n_good = np.sum(sept_model.predict(good_cutouts).flatten() >= 0.5)
print('Total: ' + str(n_tot))
print('Good: ' + str(n_good))
print()

#%% (3) Visual inspection of the negatively flagged candidates

bad_2qp = []
bad_2qi = []
bad_2qu = []

for i, score_bi in enumerate(sept_predict_bi):
    if score_bi == 0:
        c = cutout_classes[i]
        if c == '2QP':
            bad_2qp.append(good_cutouts[i])
        elif c == '2QI':
            bad_2qi.append(good_cutouts[i])
        elif c == '2QU':
            bad_2qu.append(good_cutouts[i])

bad_2qp = np.array(bad_2qp)
bad_2qi = np.array(bad_2qi)
bad_2qu = np.array(bad_2qu)

print(f"2QP bad: {len(bad_2qp)}")
print(f"2QI bad: {len(bad_2qi)}")
print(f"2QU bad: {len(bad_2qu)}")

#%% (4) Let us investigate further with the known lenses
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

#%% (5) More statistics of known lenses amongst our candidates

clean_ids = np.array(ned_simbad_clean['source_id'], dtype=np.int64)
known_in_clean = np.intersect1d(clean_ids, known_ids)
common_ids = np.intersect1d(ned_simbad_multi['source_id'].values, known_ids)

print(f"Known objects: {len(known_ids)}")
print(f"Known objects in original catalogue: {len(common_ids)}")
print(f"Known objects with PANSTARRS cutout: {len(known_in_clean)}")
print()

known_classified = ned_simbad_multi[ned_simbad_multi['source_id'].isin(common_ids)][['source_id', 'group_id', 'class_lens', 'ra', 'dec']].copy()

print(f"Known objects in original catalogue: {len(common_ids)}")
print("Lens class distribution:")
print(known_classified['class_lens'].value_counts())
print()

known_with_cutout_ids = np.intersect1d(known_ids, ned_simbad_clean['source_id'].values)

known_with_cutout = ned_simbad_clean[ned_simbad_clean['source_id'].isin(known_with_cutout_ids)].copy()

print(f"Known objects with PANSTARRS cutout: {len(known_in_clean)}")
print("Lens class distribution:")
print(known_with_cutout['class_lens'].value_counts())

#%% (6) Saving the post-CNN CUTOUT list
'''
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

df_cnn.to_csv(os.path.join(work_path, "Catalogue_after_CNN_positive.csv"), index=False)
print(f"Saved {len(df_cnn)} positive candidates")
'''
#%% (7) Saving the full catalogue with CNN and Gaia info

df_objects = ned_simbad_multi.copy()

group_sizes = df_objects['group_id'].value_counts()
df_objects['n_objects_in_group'] = df_objects['group_id'].map(group_sizes)

df_objects['has_cutout'] = -1
df_objects['cnn_flag'] = -1

good_ids_set = set(good_ids)
cnn_flag_map = dict(zip(good_cutout_ids, sept_predict_bi))

for group in df_objects['group_id'].unique():
    group_mask = df_objects['group_id'] == group
    group_ids = df_objects.loc[group_mask, 'source_id']

    cutout_flags = [cnn_flag_map[x] for x in group_ids if x in cnn_flag_map]
    if cutout_flags:
        df_objects.loc[group_mask, 'has_cutout'] = 1
        df_objects.loc[group_mask, 'cnn_flag'] = int(np.max(cutout_flags))
    elif df_objects.loc[group_mask, 'class_lens'].isin(valid_lens_classes).any():
        df_objects.loc[group_mask, 'has_cutout'] = 0
        df_objects.loc[group_mask, 'cnn_flag'] = -1

df_objects.to_csv(os.path.join(work_path, 'QUBRICS_lenses.csv'), index=False)
df_objects.to_json(os.path.join(work_path, 'QUBRICS_lenses.json'), orient='records', lines=True)

df_cnn_pos = df_objects[df_objects['cnn_flag'] == 1].copy()
df_cnn_pos.to_csv(os.path.join(work_path, 'QUBRICS_lenses_CNNpos.csv'), index=False)
df_cnn_pos.to_json(os.path.join(work_path, 'QUBRICS_lenses_CNNpos.json'), orient='records', lines=True)
group_separations = []
for gid, g in df_cnn_pos.groupby('group_id'):
    if len(g) < 2:
        continue
    coords = SkyCoord(ra=g['ra'].values * u.deg, dec=g['dec'].values * u.deg)
    seps = [coords[i].separation(coords[j]).arcsec for i, j in combinations(range(len(coords)), 2)]
    mean_sep = np.mean(seps)
    group_separations.append((gid, mean_sep))
df_sep = pd.DataFrame(group_separations, columns=['group_id', 'mean_sep_arcsec'])
global_mean = df_sep['mean_sep_arcsec'].mean()
global_median = df_sep['mean_sep_arcsec'].median()
print(f"Mean separation  : {global_mean:.3f} arcsec")
print(f"Median separation: {global_median:.3f} arcsec")
print()

n_southern = len(df_cnn_pos[df_cnn_pos['dec'] < 0])
n_total = len(df_cnn_pos)
print(f"Objects in the Southern Hemisphere (w/ CNN positive) : {n_southern} / {n_total}")

#%% (8) Checkin redshift differences

df_pairs = df_cnn_pos.groupby('group_id').filter(lambda g: len(g) == 2)
df_pairs['z_sep'] = df_pairs.groupby('group_id')['redshift_qsoc'].transform(lambda z: abs(z.iloc[0] - z.iloc[1]))
z_seps_all = df_pairs.groupby('group_id')['z_sep'].first()

lens_types = df_pairs['class_lens'].unique()
fig, axes = plt.subplots(len(lens_types)+1, 1, figsize=(7, 10), sharex=True)
bins = np.logspace(np.log10(max(z_seps_all.min(),1e-5)), np.log10(z_seps_all.max()), 42)
axes[0].hist(z_seps_all, bins=bins, color='grey', alpha=0.6)
axes[0].set_title('All CNN-positive pairs')
for i, lt in enumerate(lens_types, 1):
    seps = df_pairs[df_pairs['class_lens']==lt].groupby('group_id')['z_sep'].first()
    axes[i].hist(seps, bins=bins, alpha=0.7)
    axes[i].set_title(lt)
for ax in axes:
    ax.set_xscale('log')
    ax.set_ylabel('Count')
axes[-1].set_xlabel('Redshift separation')
plt.tight_layout()
plt.show()

#%% (9) Gaia G magnitude distribution

cnn_objects = df_objects[df_objects["cnn_flag"] == 1]
pairs = df_objects[(df_objects["cnn_flag"] == 1) & (df_objects["n_objects_in_group"] == 2)]
pairs = pairs.sort_values(by=["group_id"]).reset_index(drop=True)

brightest_mags, faintest_mags = [], []
i = 0
while i < len(pairs) - 1:
    if pairs.loc[i, "group_id"] == pairs.loc[i + 1, "group_id"]:
        mag1, mag2 = pairs.loc[i, "phot_g_mean_mag"], pairs.loc[i + 1, "phot_g_mean_mag"]
        if mag1 < mag2:
            brightest_mags.append(mag1); faintest_mags.append(mag2)
        else:
            brightest_mags.append(mag2); faintest_mags.append(mag1)
        i += 2
    else: i += 1

fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
axes[0].hist(pairs["phot_g_mean_mag"], bins=100, edgecolor='black', alpha=0.7)
axes[0].set_ylabel("Counts"); axes[0].set_xlim(13.75, 22.5)
axes[0].set_title("Gaia G magnitude distribution")
axes[1].hist(brightest_mags, bins=100, alpha=0.6, label="Bright", edgecolor='black')
axes[1].hist(faintest_mags, bins=100, alpha=0.6, label="Faint", edgecolor='black')
axes[1].set_xlabel("Gaia G magnitude")
axes[1].set_ylabel("Counts")
axes[1].set_xlim(13.75, 22.5); axes[1].legend(loc='lower left')
xticks = np.arange(14, 23, 2); axes[0].set_xticks(xticks); axes[1].set_xticks(xticks)

for ax in axes:
    inset = inset_axes(ax, width="40%", height="35%", bbox_to_anchor=(0.05, 0.05, 0.9, 0.9),
                       bbox_transform=ax.transAxes, loc='upper left')
    if ax == axes[0]:
        inset.hist(pairs["phot_g_mean_mag"], bins=100, edgecolor='black', alpha=0.7)
    else:
        inset.hist(brightest_mags, bins=100, alpha=0.6, edgecolor='black')
        inset.hist(faintest_mags, bins=100, alpha=0.6, edgecolor='black')
    inset.set_xlim(12, 14)
    inset.set_ylim(0, 2.5)
    inset.set_yticks([0, 1, 2])
    inset.set_xticks([12, 13, 14])
    inset.tick_params(axis='both', labelsize=8)
    xy_main = (13.75, 0); xy_inset_left = (12, 0); xy_inset_right = (14, 0)
    for xy_inset in [xy_inset_left, xy_inset_right]:
        ax.add_artist(ConnectionPatch(xyA=xy_inset, coordsA=inset.transData,
                                      xyB=xy_main, coordsB=ax.transData,
                                      color="0.3", lw=1))
plt.tight_layout()
img_dir = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/Immagini_Nuove_Guarneri/"
fig.savefig(img_dir + "MagDistPairs_new.pdf", bbox_inches="tight")
plt.show()

#%% (10) Parallax and Proper motion over error distribution

cnn_objects = df_objects[df_objects["cnn_flag"] == 1]
known_ids = np.array(df2.source_id, dtype=np.int64)
all_ids = np.array(cnn_objects.source_id, dtype=np.int64)
known_mask = np.isin(all_ids, known_ids)
unknown_mask = ~known_mask

x = cnn_objects['parallax_over_error'].values
y = cnn_objects['pm_over_error'].values
gmag = cnn_objects['phot_g_mean_mag'].values

fig, ax = plt.subplots(figsize=(7,6))
sc = ax.scatter(x[unknown_mask], y[unknown_mask], c=gmag[unknown_mask], cmap='viridis', s=10, alpha=0.8)
ax.scatter(x[known_mask], y[known_mask], color='black', s=10, alpha=0.8)
ax.set_xlim(-10,125)
ax.set_ylim(-5,85)
ax.set_xlabel('Parallax over error')
ax.set_ylabel('Proper motion over error')
ax.set_title('Parallax vs Proper Motion')

ax.legend(handles=[plt.Line2D([], [], marker='o', color='w', markerfacecolor='black', markersize=8, label='Known'),
                   plt.Line2D([], [], marker='o', color='w', markerfacecolor='lightgreen', markersize=8, label='Candidates')], loc='lower right')

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Gaia G magnitude')
inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
inset.scatter(x[unknown_mask], y[unknown_mask], c=gmag[unknown_mask], cmap='viridis', s=10, alpha=0.8)
inset.scatter(x[known_mask], y[known_mask], color='black', s=10, alpha=0.8)
inset.set_xlim(-5, 10)
inset.set_ylim(-1, 25)
inset.set_xticks(np.arange(-5,11,5))
inset.set_yticks([0,5,10,15,20,25])

rect = Rectangle((-5,-1), 15, 26, linewidth=1, edgecolor='black', facecolor='none', linestyle='--', alpha=0.2)
ax.add_patch(rect)
mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="black", lw=1, alpha=0.2)
fig.savefig(img_dir + "ParallaxPM_new.pdf")
plt.show()

mask = (cnn_objects["parallax_over_error"] >= -5) & (cnn_objects["parallax_over_error"] <= 5)
mask_known = mask[known_mask]
mask_unknown = mask[unknown_mask]
percent = 100 * mask.sum() / len(cnn_objects)
percent_known = 100 * mask_known.sum() / known_mask.sum()
percent_unknown = 100 * mask_unknown.sum() / unknown_mask.sum()
print(f"Percentage of objects with parallax_over_error between -5 e 5: {percent:.2f}%")
print(f"Percentage known: {percent_known:.2f}%")
print(f"Percentage unknown: {percent_unknown:.2f}%")
print()

#%% (11) Matching our candidate list with the known lenses list

with fits.open("C:/Users/fchit/Downloads/ListUniqueLenses.fits") as hdul:
    table_known = Table(hdul[1].data)
for col in table_known.colnames:
    if table_known[col].ndim > 1 and table_known[col].shape[1] == 1:
        table_known[col] = table_known[col].flatten()
df_known_lenses = table_known.to_pandas()
mask_b = np.abs(df_known_lenses["b"]) < 25
mask_sep = df_known_lenses["sep"] > 5
df_known_lenses_clean = df_known_lenses[~(mask_b | mask_sep)].copy()
n_lowb = mask_b.sum()
n_largesep = mask_sep.sum()
n_both = (mask_b & mask_sep).sum()
print("Too low gal latitude: " + str(n_lowb))
print("Too great separation: " + str(n_largesep))
print("Both: " + str(n_both))
print("Remaining lenses: " + str(len(df_known_lenses_clean)))
print("\n")

fine_objects = df_objects[df_objects["cnn_flag"] == 1].copy().reset_index(drop=True)
semifine_objects = df_objects[df_objects["cnn_flag"] == -1].copy().reset_index(drop=True)
df_groups = fine_objects.groupby("group_id").agg(ra_mean=("ra", "mean"), dec_mean=("dec", "mean")).reset_index()
df_semigroups = semifine_objects.groupby("group_id").agg(ra_mean=("ra", "mean"), dec_mean=("dec", "mean")).reset_index()
coords_groups = SkyCoord(ra=df_groups["ra_mean"].values * u.deg, dec=df_groups["dec_mean"].values * u.deg)
coords_semigroups = SkyCoord(ra=df_semigroups["ra_mean"].values * u.deg, dec=df_semigroups["dec_mean"].values * u.deg)
coords_known = SkyCoord(ra=df_known_lenses_clean["RAd"].values * u.deg, dec=df_known_lenses_clean["DECd"].values * u.deg)

idx_group, idx_known, d2d, _ = coords_known.search_around_sky(coords_groups, seplimit=5 * u.arcsec)
matches_groups = pd.DataFrame({
    "group_id": df_groups.iloc[idx_group]["group_id"].values,
    "group_ra": df_groups.iloc[idx_group]["ra_mean"].values,
    "group_dec": df_groups.iloc[idx_group]["dec_mean"].values,
    "known_name": df_known_lenses_clean.iloc[idx_known]["Name"].values,
    "known_origin": df_known_lenses_clean.iloc[idx_known]["origin"].values,
    "sep_arcsec": d2d.arcsec}).sort_values("sep_arcsec").groupby("group_id").first().reset_index()
print("Recovered known lens fields: " + str(matches_groups["group_id"].nunique()))
matched_group_ids = set(matches_groups["group_id"])

idx_semigroup, idx_semiknown, d2dsemi, _ = coords_known.search_around_sky(coords_semigroups, seplimit=5 * u.arcsec)
matches_semigroups = pd.DataFrame({
    "group_id": df_semigroups.iloc[idx_semigroup]["group_id"].values,
    "group_ra": df_semigroups.iloc[idx_semigroup]["ra_mean"].values,
    "group_dec": df_semigroups.iloc[idx_semigroup]["dec_mean"].values,
    "known_name": df_known_lenses_clean.iloc[idx_semiknown]["Name"].values,
    "known_origin": df_known_lenses_clean.iloc[idx_semiknown]["origin"].values,
    "sep_arcsec": d2dsemi.arcsec}).sort_values("sep_arcsec").groupby("group_id").first().reset_index()
print("Missed in the PANSTARRS footprint: " + str(matches_semigroups["group_id"].nunique()))
missed_group_ids = set(matches_semigroups["group_id"])
matched_known_idx_any = set(idx_known) | set(idx_semiknown)
df_known_not_in_candidates = df_known_lenses_clean.iloc[
    sorted(set(range(len(df_known_lenses_clean))) - matched_known_idx_any)
].copy().reset_index(drop=True)
print("Not in the good or outside the PS footprint: " + str(len(df_known_not_in_candidates)))
print("\n")

data_found = fits.getdata(base_path + "Mag_FoundMatch.fits")
cols = {}
for name in data_found.dtype.names:
    arr = np.asarray(data_found[name])
    if arr.ndim == 1:
        cols[name] = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        cols[name] = arr[:, 0]
    elif arr.ndim == 2:
        for j in range(arr.shape[1]):
            cols[f"{name}_{j}"] = arr[:, j]
df_found = pd.DataFrame(cols).copy()

def make_little_endian_df(df):
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if hasattr(s.dtype, "byteorder") and s.dtype.byteorder == ">":
            out[c] = s.to_numpy().byteswap().newbyteorder()
    return out

df_found = make_little_endian_df(df_found)


df_found["ra"] = pd.to_numeric(df_found["ra"], errors="coerce")
df_found["dec"] = pd.to_numeric(df_found["dec"], errors="coerce")
df_found = df_found.dropna(subset=["ra", "dec"]).reset_index(drop=True)
coords_found = SkyCoord(ra=df_found["ra"].values * u.deg, dec=df_found["dec"].values * u.deg)
idx_g, d2d_g, _ = coords_found.match_to_catalog_sky(coords_groups)
idx_s, d2d_s, _ = coords_found.match_to_catalog_sky(coords_semigroups)
gid_g = df_groups.iloc[idx_g]["group_id"].to_numpy()
gid_s = df_semigroups.iloc[idx_s]["group_id"].to_numpy()
max_sep = 5.0 * u.arcsec
ok_g = d2d_g < max_sep
ok_s = d2d_s < max_sep
df_found["group_id_match"] = pd.Series([pd.NA] * len(df_found), dtype="Int64")
both = ok_g & ok_s
choose_g = both & (d2d_g <= d2d_s)
choose_s = both & (d2d_s < d2d_g)
df_found.loc[choose_g, "group_id_match"] = gid_g[choose_g]
df_found.loc[choose_s, "group_id_match"] = gid_s[choose_s]
only_g = ok_g & ~both
only_s = ok_s & ~both
df_found.loc[only_g, "group_id_match"] = gid_g[only_g]
df_found.loc[only_s, "group_id_match"] = gid_s[only_s]
df_found["in_ps_footprint"] = df_found["group_id_match"].isin(matched_group_ids)
df_found["out_ps_footprint"] = df_found["group_id_match"].isin(missed_group_ids)
df_found["Gmag_bright"] = pd.to_numeric(df_found["Gmag_bright"], errors="coerce")
df_found["Gmag_faint"] = pd.to_numeric(df_found["Gmag_faint"], errors="coerce")
df_found["sep"] = pd.to_numeric(df_found["sep"], errors="coerce")
df_found_rec = df_found[df_found["in_ps_footprint"]].copy()
df_found_miss = df_found[df_found["out_ps_footprint"]].copy()
brightest_mags_rec = df_found_rec["Gmag_bright"].to_numpy()
faintest_mags_rec = df_found_rec["Gmag_faint"].to_numpy()
separations_rec = df_found_rec["sep"].to_numpy()
brightest_mags_miss = df_found_miss["Gmag_bright"].to_numpy()
faintest_mags_miss = df_found_miss["Gmag_faint"].to_numpy()
separations_miss = df_found_miss["sep"].to_numpy()
data_missed_no_cand = fits.getdata(base_path + "Mag_MissedNoQSOCand.fits")
data_missed_single = fits.getdata(base_path + "Mag_MissedSingleMatch.fits")

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
axes[0].scatter(brightest_mags_miss, faintest_mags_miss, s=30, alpha=0.9, facecolors='none', edgecolors='red')
axes[0].scatter(brightest_mags_rec, faintest_mags_rec, s=30, alpha=0.9, facecolors='none', edgecolors='blue')
axes[0].scatter(data_missed_no_cand['Gmag_bright'], data_missed_no_cand['Gmag_faint'], s=30, alpha=0.9, facecolors='none', edgecolors='orange')
axes[0].scatter(data_missed_single['Gmag_bright'], data_missed_single['Gmag_faint'], s=30, alpha=0.9, facecolors='none', edgecolors='green')
axes[0].set_ylabel("Gaia G magnitude (faint)")

axes[1].scatter(brightest_mags_miss, separations_miss, s=30, alpha=0.9, facecolors='none', edgecolors='red', label="Outside PS footprint")
axes[1].scatter(brightest_mags_rec, separations_rec, s=30, alpha=0.9, facecolors='none', edgecolors='blue', label="Recovered known lenses")
axes[1].scatter(data_missed_no_cand['Gmag_bright'], data_missed_no_cand['sep'], s=30, alpha=0.9, facecolors='none', edgecolors='orange', label="Missed - No candidate")
axes[1].scatter(data_missed_single['Gmag_bright'], data_missed_single['sep'], s=30, alpha=0.9, facecolors='none', edgecolors='green', label="Missed - Single match")
axes[1].set_ylabel("Separation [arcsec]")
axes[1].set_xlabel("Gaia G magnitude (bright)")
axes[1].set_xlim(15.5, 21.25)
axes[1].legend()
plt.tight_layout()
plt.show()

#%%

to1d = lambda a: np.asarray(a).ravel()

fig = plt.figure(figsize=(7,10))
gs = GridSpec(4, 2, height_ratios=[1.2,4,0.2,4],
              width_ratios=[4,1.2], hspace=0.001, wspace=0.001)
ax_hx1 = fig.add_subplot(gs[0,0])
ax_sc1 = fig.add_subplot(gs[1,0])
ax_hy1 = fig.add_subplot(gs[1,1], sharey=ax_sc1)
ax_sc2 = fig.add_subplot(gs[3,0], sharex=ax_sc1)
ax_hy2 = fig.add_subplot(gs[3,1], sharey=ax_sc2)
ax_sc1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

x_miss = to1d(brightest_mags_miss)
x_rec  = to1d(brightest_mags_rec)
x_no   = to1d(data_missed_no_cand['Gmag_bright'])
x_sing = to1d(data_missed_single['Gmag_bright'])
y1_miss = to1d(faintest_mags_miss)
y1_rec  = to1d(faintest_mags_rec)
y1_no   = to1d(data_missed_no_cand['Gmag_faint'])
y1_sing = to1d(data_missed_single['Gmag_faint'])
y2_miss = to1d(separations_miss)
y2_rec  = to1d(separations_rec)
y2_no   = to1d(data_missed_no_cand['sep'])
y2_sing = to1d(data_missed_single['sep'])

nbins_marg = 15
bins_x = np.linspace(15.5, 21.25, nbins_marg)
bins_y1 = np.linspace(np.nanmin(np.concatenate([y1_miss,y1_rec,y1_no,y1_sing])), np.nanmax(np.concatenate([y1_miss,y1_rec,y1_no,y1_sing])), nbins_marg)
bins_y2 = np.linspace(np.nanmin(np.concatenate([y2_miss,y2_rec,y2_no,y2_sing])), np.nanmax(np.concatenate([y2_miss,y2_rec,y2_no,y2_sing])), nbins_marg)

others_x  = [x_rec, x_no, x_sing]
others_y1 = [y1_rec, y1_no, y1_sing]
others_y2 = [y2_rec, y2_no, y2_sing]
others_c  = ['blue','orange','green']
zmap = {'orange': 1,'blue': 2,'red': 3,'green': 4}

for x, y, c in zip(others_x, others_y1, others_c):
    ax_sc1.scatter(x, y, s=30, alpha=0.9, facecolors='none', edgecolors=c, zorder=zmap[c])
    ax_hx1.hist(x, bins=bins_x, facecolor=c, edgecolor='k', alpha=0.6, zorder=zmap[c])
    ax_hy1.hist(y, bins=bins_y1,
                facecolor=c, edgecolor='k', alpha=0.6,
                orientation='horizontal',
                zorder=zmap[c])

ax_sc1.scatter(x_miss, y1_miss, s=30, alpha=0.9, facecolors='none', edgecolors='red', zorder=zmap['red'])
ax_hx1.hist(x_miss, bins=bins_x, facecolor='red', edgecolor='k', alpha=0.6, zorder=zmap['red'])
ax_hy1.hist(y1_miss, bins=bins_y1, facecolor='red', edgecolor='k', alpha=0.6, orientation='horizontal', zorder=zmap['red'])

for x, y, c in zip(others_x, others_y2, others_c):
    ax_sc2.scatter(x, y, s=30, alpha=0.9, facecolors='none', edgecolors=c, zorder=zmap[c])
    ax_hy2.hist(y, bins=bins_y2, facecolor=c, edgecolor='k', alpha=0.6, orientation='horizontal', zorder=zmap[c])

ax_sc2.scatter(x_miss, y2_miss, s=30, alpha=0.9, facecolors='none', edgecolors='red', zorder=zmap['red'])
ax_hy2.hist(y2_miss, bins=bins_y2, facecolor='red', edgecolor='k', alpha=0.6, orientation='horizontal', zorder=zmap['red'])
ax_sc1.set_ylabel("Gaia G magnitude (faint)")
ax_sc2.set_ylabel("Separation [arcsec]")
ax_sc2.set_xlabel("Gaia G magnitude (bright)")
ax_sc1.set_xlim(15.5, 21.25)

ax_hx1.tick_params(axis='x', labelbottom=False)
ax_hx1.tick_params(axis='y', labelleft=False)
ax_hy1.tick_params(axis='x', labelbottom=False)
ax_hy1.tick_params(axis='y', labelleft=False)
ax_hy2.tick_params(axis='x', labelbottom=False)
ax_hy2.tick_params(axis='y', labelleft=False)

ax_sc2.legend(["Recovered known lenses", "Missed - No candidate", "Missed - Single match", "Outside PS footprint"])
plt.savefig("C:/Users/fchit/Downloads/CompareKnown_new.pdf", format="pdf", bbox_inches="tight")
plt.show()



#%% (12) Matching with the GAIA Focused Product (Gaia, Krone-Martins - 2023)

from astropy.table import Table

with fits.open("C:/Users/fchit/Downloads/ListUniqueLenses.fits") as hdul: table_known = Table(hdul[1].data)
for col in table_known.colnames:
    if table_known[col].ndim > 1 and table_known[col].shape[1] == 1: table_known[col] = table_known[col].flatten()
df_known_lenses = table_known.to_pandas()

fine_objects = df_objects[df_objects["cnn_flag"] == 1]

coords_known = SkyCoord(ra=df_known_lenses["RAd"].values * u.deg, dec=df_known_lenses["DECd"].values * u.deg)
coords_lens = SkyCoord(ra=fine_objects["ra"].values * u.deg, dec=fine_objects["dec"].values * u.deg)
idx_lens, idx_known, d2d, _ = coords_known.search_around_sky(coords_lens, seplimit=5 * u.arcsec)

print(f"Matches found: {len(idx_known)}")

matches_known = pd.DataFrame({"known_name": df_known_lenses.iloc[idx_known]["Name"].values, "known_origin": df_known_lenses.iloc[idx_known]["origin"].values, "known_ra": df_known_lenses.iloc[idx_known]["RAd"].values, "known_dec": df_known_lenses.iloc[idx_known]["DECd"].values, "lens_source_id": fine_objects.iloc[idx_lens]["source_id"].values, "lens_class": fine_objects.iloc[idx_lens]["class_lens"].values, "sep_arcsec": d2d.arcsec})

print()
print("Lenses class of found matches:")
print(matches_known["lens_class"].value_counts())

matched_ids = set(matches_known["lens_source_id"])

pairs = cnn_objects[cnn_objects["n_objects_in_group"] == 2].sort_values("group_id").reset_index(drop=True)
brightest_mags, faintest_mags, separations = [], [], []

i = 0
while i < len(pairs) - 1:
    if pairs.loc[i, "group_id"] == pairs.loc[i+1, "group_id"]:
        mag1, mag2 = pairs.loc[i, "phot_g_mean_mag"], pairs.loc[i+1, "phot_g_mean_mag"]
        ra1, dec1 = pairs.loc[i, "ra"], pairs.loc[i, "dec"]
        ra2, dec2 = pairs.loc[i+1, "ra"], pairs.loc[i+1, "dec"]
        if mag1 < mag2: brightest_mags.append(mag1); faintest_mags.append(mag2)
        else: brightest_mags.append(mag2); faintest_mags.append(mag1)
        sep = SkyCoord(ra1*u.deg, dec1*u.deg).separation(SkyCoord(ra2*u.deg, dec2*u.deg)).arcsec
        separations.append(sep)
        i += 2
    else: i += 1

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

axes[0].scatter(brightest_mags, faintest_mags, s=20, alpha=0.8, facecolors='none', edgecolors='red')
axes[0].set_ylabel("Gaia G magnitude (faint)")

axes[1].scatter(brightest_mags, separations, s=20, alpha=0.8, facecolors='none', edgecolors='red')
axes[1].set_ylabel("Separation [arcsec]")
axes[1].set_xlabel("Gaia G magnitude (bright)")
axes[1].set_xlim(15.5, 21.25)

plt.tight_layout()
plt.show()



#%% (12) Looking for DLA with SDSS spectrum in our candidates (DO NOT RUN AGAIN!)

lens_df = df_objects[(df_objects['cnn_flag'] == 1)].copy()
coords = SkyCoord(ra=lens_df['ra'].values*u.deg, dec=lens_df['dec'].values*u.deg)

Vizier.ROW_LIMIT = -1
dla_catalog = 'J/A+A/547/L1'

dla_matches = []
for i, c in enumerate(coords):
    if i % 10 == 0:
        print(i)
    if i % 100 == 0:
        time.sleep(2)
    try:
        result = Vizier.query_region(c, radius=1.5*u.arcsec, catalog=dla_catalog)
        dla_matches.append(True if result else False)
    except:
        dla_matches.append(False)
lens_df['has_dla'] = dla_matches
for group_id in lens_df.loc[lens_df['has_dla'], 'group_id'].unique():
    lens_df.loc[lens_df['group_id'] == group_id, 'has_dla'] = True
lens_with_dla = lens_df[lens_df['has_dla']].copy()
lens_with_dla['sdss_redshift'] = None
lens_with_dla['has_sdss_spectrum'] = False

for i, (idx, row) in enumerate(lens_with_dla.iterrows()):
    c = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg)
    attempts = 0
    success = False
    while not success and attempts < 5:
        try:
            spec = SDSS.get_spectra(coordinates=c, radius=2*u.arcsec)
            xid = SDSS.query_region(c, radius=1.5*u.arcsec, spectro=True, fields=['z'])
            if xid is not None and len(xid) > 0:
                z_val = xid[0]['z']
                if hasattr(z_val, '__len__'):
                    z_val = float(z_val[0])
                else:
                    z_val = float(z_val)
                lens_with_dla.at[idx, 'sdss_redshift'] = z_val
            if spec:
                lens_with_dla.at[idx, 'has_sdss_spectrum'] = True
                spec_path = os.path.join(DLA_path, f"spectrum_{row['source_id']}_{row['redshift_qsoc']}_{row['group_id']}.fits")
                spec[0].writeto(spec_path, overwrite=True)
            success = True
        except:
            attempts += 1
            time.sleep(5)

    if i % 10 == 0:
        time.sleep(2)

lens_with_dla['delta_z'] = lens_with_dla['sdss_redshift'] - lens_with_dla['redshift_qsoc']
southern = lens_with_dla[lens_with_dla['dec'] < 0]

print(f"Lenses with DLA and SDSS spectrum: {lens_with_dla['has_sdss_spectrum'].sum()} on {len(lens_with_dla)}")
print(f"In the southern hemisphere: {len(southern)} on {len(lens_with_dla)}")

summary_df = lens_with_dla[lens_with_dla['has_sdss_spectrum']].copy()
summary_df['spectrum_path'] = summary_df.apply(lambda row: os.path.join(DLA_path, f"spectrum_{row['source_id']}_{row['redshift_qsoc']}_{row['group_id']}.fits"), axis=1)
summary_df[['source_id', 'redshift_qsoc', 'spectrum_path']].to_csv(os.path.join(DLA_path, "DLAsummary.csv"), index=False)

'''
import voigtfit as vfit
data = all_spectra[1].data
flux = data['flux']
loglam = data['loglam']
wavelength = 10**loglam

z_qso = float(lens_with_dla.iloc[4]['redshift_qsoc'])
wavelength_rest = wavelength / (1 + z_qso)
spectrum = np.column_stack([wavelength_rest, flux, np.ones_like(flux)*np.std(flux)])

dataset = vfit.DataSet()
dataset.add_data(wavelength_rest, flux, sigma=np.ones_like(flux)*np.std(flux), normalized=False)
dataset.set_qso(z_qso)
'''

#%% (13) QUBRICS Collection - let's see which quasars are lensed so that the magnitude distribution changes

data_collection = fits.getdata(os.path.join(base_path, "COLLECTION_M277_v2.fits"), 1)
mag = data_collection["mag_abs_145"]
mask = np.isfinite(mag)
ra_coll = data_collection["RA_old"][mask]
dec_coll = data_collection["DEC_old"][mask]
mag_coll = mag[mask]
apl_coll = data_collection["AMPLIFICATION"][mask]
lens_flag_coll = data_collection["LENS"][mask]
lens_flag_confirmed = lens_flag_coll.copy()
lens_flag_candidate = np.zeros(len(ra_coll), bool)
print('Already flagged lenses in Catalogue: ' + str(lens_flag_coll.sum()))
print()
north_coll = mag_coll[dec_coll > 0]
south_coll = mag_coll[dec_coll < 0]
'''
bins = np.histogram_bin_edges(mag_coll, bins=23)
fig, ax = plt.subplots()
ax.hist(mag_coll, bins=bins, color="grey", label='All', alpha=0.3)
ax.hist(north_coll, bins=bins, color="blue", label='North', alpha=0.3)
ax.hist(south_coll, bins=bins, color="red", label='South', alpha=0.3)
ax.set_yscale("log")
ax.set_xlabel("M145")
ax.set_ylabel("N")
ax.legend(loc='upper left')
plt.show()
'''
matching_rad_arcsec = 5

Vizier.ROW_LIMIT = -1
v = Vizier(row_limit=-1, columns=["RA_ICRS", "DE_ICRS", "Grade"])
gral = v.get_catalogs("J/A+A/685/A130/table3")[0]
lens_flag_gral = np.isin(gral["Grade"], ["A+", "A-", "B+", "B-"])
print("GraL candidates:", lens_flag_gral.sum(), "of grades", list(np.unique(gral["Grade"])))
coords_gral = SkyCoord(gral["RA_ICRS"], gral["DE_ICRS"])
coords_coll = SkyCoord(ra_coll * u.deg, dec_coll * u.deg)
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_gral)
matched = sep < matching_rad_arcsec * u.arcsec
gral_matched = np.zeros(len(gral), dtype=bool)
gral_matched[idx[matched]] = True
lens_flag_from_gral = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_gral[matched] = lens_flag_gral[idx[matched]]
new_lenses = lens_flag_from_gral & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_candidate |= lens_flag_from_gral
lens_flag_coll |= lens_flag_from_gral
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()

v = Vizier(row_limit=-1, columns=["_RA", "_DE", "Grade"])
andika = v.get_catalogs("J/A+A/678/A103/tableb1")[0]
lens_flag_andika = np.isin(andika["Grade"], ["A", "A*", "B", "B*"])
print("Andika candidates:", lens_flag_andika.sum(), "of grades", list(np.unique(andika["Grade"])))
coords_andika = SkyCoord(andika["_RA"], andika["_DE"])
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_andika)
matched = sep < matching_rad_arcsec * u.arcsec
lens_flag_from_andika = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_andika[matched] = lens_flag_andika[idx[matched]]
new_lenses = lens_flag_from_andika & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_candidate |= lens_flag_from_andika
lens_flag_coll |= lens_flag_from_andika
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()

v = Vizier(row_limit=-1, columns=["RAJ2000", "DEJ2000", "Rank"])
ps1 = v.get_catalogs("J/MNRAS/486/4987/table1")[0]
lens_flag_ps1 = np.isin(ps1["Rank"], ["A", "B", "C"])
print("Rusu candidates:", lens_flag_ps1.sum(), "of grades", list(np.unique(ps1["Rank"])))
coords_ps1 = SkyCoord(ps1["RAJ2000"], ps1["DEJ2000"])
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_ps1)
matched = sep < matching_rad_arcsec * u.arcsec
lens_flag_from_ps1 = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_ps1[matched] = lens_flag_ps1[idx[matched]]
new_lenses = lens_flag_from_ps1 & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_candidate |= lens_flag_from_ps1
lens_flag_coll |= lens_flag_from_ps1
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()

v = Vizier(row_limit=-1, columns=["RAJ2000", "DEJ2000", "Rank"])
ps1_conf = v.get_catalogs("J/MNRAS/486/4987/tableb1")[0]
lens_flag_ps1 = np.isin(ps1_conf["Rank"], ["A", "B", "C"])
print("Rusu old confirmed:", lens_flag_ps1.sum(), "of grades", list(np.unique(ps1_conf["Rank"])))
coords_ps1 = SkyCoord(ps1_conf["RAJ2000"], ps1_conf["DEJ2000"])
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_ps1)
matched = sep < matching_rad_arcsec * u.arcsec
lens_flag_from_ps1 = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_ps1[matched] = lens_flag_ps1[idx[matched]]
new_lenses = lens_flag_from_ps1 & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_confirmed |= lens_flag_from_ps1
lens_flag_coll |= lens_flag_from_ps1
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()

v = Vizier(row_limit=-1, columns=["RAJ2000", "DEJ2000", "Grade"])
desi = v.get_catalogs("J/A+A/672/A123/table1")[0]
lens_flag_desi = np.isin(desi["Grade"], ["A", "B", 'C'])
print("DESI-LS DR9 candidates:", lens_flag_desi.sum(), "of grades", list(np.unique(desi["Grade"])))
coords_desi = SkyCoord(desi["RAJ2000"], desi["DEJ2000"])
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_desi)
matched = sep < matching_rad_arcsec * u.arcsec
lens_flag_from_desi = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_desi[matched] = lens_flag_desi[idx[matched]]
new_lenses = lens_flag_from_desi & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_candidate |= lens_flag_from_desi
lens_flag_coll |= lens_flag_from_desi
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()

v = Vizier(row_limit=-1, columns=["RAJ2000", "DEJ2000"])
williams = v.get_catalogs("J/MNRAS/477/L70/table1")[0]
lens_flag_williams = np.ones(len(williams), dtype=bool)
print("Williams lenses:", lens_flag_williams.sum())
coords_williams = SkyCoord(williams["RAJ2000"], williams["DEJ2000"])
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_williams)
matched = sep < matching_rad_arcsec * u.arcsec
lens_flag_from_williams = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_williams[matched] = lens_flag_williams[idx[matched]]
new_lenses = lens_flag_from_williams & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_confirmed |= lens_flag_from_williams
lens_flag_coll |= lens_flag_from_williams
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()

v = Vizier(columns=["_RA", "_DE", "Prob"])
richards = v.get_catalogs("J/AJ/131/49/data")[0]
lens_flag_richards = richards["Prob"] > 0.1
print("Richards lenses:", lens_flag_richards.sum())
coords_richards = SkyCoord(richards["_RA"], richards["_DE"])
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_richards)
matched = sep < matching_rad_arcsec * u.arcsec
lens_flag_from_richards = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_richards[matched] = lens_flag_richards[idx[matched]]
new_lenses = lens_flag_from_richards & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_candidate |= lens_flag_from_richards
lens_flag_coll |= lens_flag_from_richards
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()

v = Vizier(row_limit=-1, columns=["_RA", "_DE", "Grade"])
dawes = v.get_catalogs("J/ApJS/269/61/table2")[0]
lens_flag_dawes = np.isin(dawes["Grade"], ["A", "B", "C"])
print("Dawes candidates:", lens_flag_dawes.sum(), "of grades", list(np.unique(dawes["Grade"])))
coords_dawes = SkyCoord(dawes["_RA"], dawes["_DE"])
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_dawes)
matched = sep < matching_rad_arcsec * u.arcsec
lens_flag_from_dawes = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_dawes[matched] = lens_flag_dawes[idx[matched]]
new_lenses = lens_flag_from_dawes & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_candidate |= lens_flag_from_dawes
lens_flag_coll |= lens_flag_from_dawes
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()

v = Vizier(row_limit=-1, columns=["_RA", "_DE"])
rusu16 = v.get_catalogs("J/MNRAS/458/2/table3a")[0]
lens_flag_rusu16 = np.ones(len(rusu16), dtype=bool)
print("Rusu+16 confirmed lenses:", lens_flag_rusu16.sum())
coords_rusu16 = SkyCoord(rusu16["_RA"], rusu16["_DE"])
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_rusu16)
matched = sep < matching_rad_arcsec * u.arcsec
lens_flag_from_rusu16 = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_rusu16[matched] = lens_flag_rusu16[idx[matched]]
new_lenses = lens_flag_from_rusu16 & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_confirmed |= lens_flag_from_rusu16
lens_flag_coll |= lens_flag_from_rusu16
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()

def sdss_name_to_coord(name):
    name = name.replace("SDSS ", "").strip()
    try:
        return SkyCoord(name, unit=(u.hourangle, u.deg), frame="icrs")
    except Exception:
        return None
v = Vizier(row_limit=-1, columns=["SDSS", "Reject"])
sqls = v.get_catalogs("J/AJ/143/119/table5")[0]
lens_flag_sqls = np.ones(len(sqls), dtype=bool)
print("Inada+12 confirmed lenses:", lens_flag_sqls.sum())
coords_sqls = []
for name in sqls["SDSS"]:
    c = sdss_name_to_coord(name)
    if c is not None:
        coords_sqls.append(c)
coords_sqls = SkyCoord(coords_sqls)
print("Resolved SQLS lenses from SDSS names:", len(coords_sqls))
if len(coords_sqls) == 0:
    raise RuntimeError("No SQLS coordinates resolved from SDSS names")
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_sqls)
matched = sep < matching_rad_arcsec * u.arcsec
lens_flag_from_sqls = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_sqls[matched] = True
new_lenses = lens_flag_from_sqls & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_confirmed |= lens_flag_from_sqls
lens_flag_coll |= lens_flag_from_sqls
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()

v = Vizier(row_limit=-1, columns=["Name"])
veron = v.get_catalogs("VII/258/table2")[0]
lens_flag_veron = np.ones(len(veron), dtype=bool)
print("Veron-Cetty+10 lensed QSOs:", lens_flag_veron.sum())
Simbad.add_votable_fields("ra", "dec")
ra_v = []
dec_v = []
for name in veron["Name"]:
    res = Simbad.query_object(str(name))
    if res is None or len(res) == 0:
        continue
    ra_v.append(res["ra"][0])
    dec_v.append(res["dec"][0])
print("Resolved Véron lensed QSOs via SIMBAD:", len(ra_v))
coords_veron = SkyCoord(np.array(ra_v) * u.deg, np.array(dec_v) * u.deg)
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_veron)
matched = sep < matching_rad_arcsec * u.arcsec
lens_flag_from_veron = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_veron[matched] = True
new_lenses = lens_flag_from_veron & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_candidate |= lens_flag_from_veron
lens_flag_coll |= lens_flag_from_veron
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()

v = Vizier(row_limit=-1, columns=["RAJ2000", "DEJ2000"])
millon20 = v.get_catalogs("J/A+A/640/A105/stars")[0]
lens_flag_millon20 = np.ones(len(millon20), dtype=bool)
print("Millon+20 lensed QSOs:", lens_flag_millon20.sum())
coords_millon20 = SkyCoord(millon20["RAJ2000"], millon20["DEJ2000"], unit=(u.hourangle, u.deg), frame="icrs")
idx, sep, _ = coords_coll.match_to_catalog_sky(coords_millon20)
matched = sep < matching_rad_arcsec * u.arcsec
lens_flag_from_millon20 = np.zeros(len(ra_coll), dtype=bool)
lens_flag_from_millon20[matched] = lens_flag_millon20[idx[matched]]
new_lenses = lens_flag_from_millon20 & (~lens_flag_coll)
print("New lenses in this step:", new_lenses.sum())
lens_flag_confirmed |= lens_flag_from_millon20
lens_flag_coll |= lens_flag_from_millon20
print("Cumulative candidates :", lens_flag_candidate.sum())
print("Cumulative confirmed  :", lens_flag_confirmed.sum())
print("End step lenses:", lens_flag_coll.sum())
print()


plt.figure(figsize=(8,5))
alpi = 0.2
plt.scatter(gral["RA_ICRS"], gral["DE_ICRS"], s=10, alpha=alpi, color="green", label="Makarov+23")
plt.scatter(andika["_RA"], andika["_DE"], s=10, alpha=alpi, color="blue", label="Andika+23")
plt.scatter(ps1["RAJ2000"], ps1["DEJ2000"], s=10, alpha=alpi, color="goldenrod", label="Rusu+19")
plt.scatter(ps1_conf["RAJ2000"], ps1_conf["DEJ2000"], s=10, alpha=alpi, color="purple", label="Rusu+19")
plt.scatter(desi["RAJ2000"], desi["DEJ2000"], s=10, alpha=alpi, color="maroon", label="He+23")
plt.scatter(williams["RAJ2000"], williams["DEJ2000"], s=10, alpha=alpi, color="tan", label="Williams+18")
plt.scatter(richards["_RA"], richards["_DE"], s=10, alpha=alpi, color="coral", label="Richards+06")
plt.scatter(dawes["_RA"], dawes["_DE"], s=10, alpha=alpi, color="olivedrab", label="Dawes+23")
plt.scatter(rusu16["_RA"], rusu16["_DE"], s=10, alpha=alpi, color="slateblue", label="Rusu+16")
plt.scatter(coords_sqls.ra.deg, coords_sqls.dec.deg, s=10, alpha=alpi, color="royalblue", label="Inada+12")
plt.scatter(np.array(ra_v), np.array(dec_v), s=10, alpha=alpi, color="peru", label="Veron+10")
plt.scatter(ra_coll, dec_coll, s=10, alpha=alpi, color="red", label="Collection")
plt.scatter(ra_coll[lens_flag_coll], dec_coll[lens_flag_coll], s=20, color="black", label="Coll lenses")
plt.xlabel("RA [deg]")
plt.ylabel("Dec [deg]")
plt.legend()
plt.show()

#%% (14) Save Collection file

provenance = np.array([""] * len(ra_coll), dtype="U256")
def _add_src(flag, tag):
    flag = np.asarray(flag, dtype=bool)
    tag = str(tag)
    m = flag & (provenance == "")
    provenance[m] = tag
    m = flag & (provenance != "")
    already = np.char.find(provenance[m], tag) >= 0
    newvals = np.char.add(np.char.add(provenance[m], "|"), tag)
    provenance[m] = np.where(already, provenance[m], newvals)
_add_src(lens_flag_confirmed, "CATALOG_LENS")
_add_src(lens_flag_from_gral, "GraL")
_add_src(lens_flag_from_andika, "Andika23")
_add_src(lens_flag_from_ps1, "Rusu19")
_add_src(lens_flag_from_desi, "DESI_DR9")
_add_src(lens_flag_from_williams, "Williams18")
_add_src(lens_flag_from_richards, "Richards06")
_add_src(lens_flag_from_dawes, "Dawes23")
_add_src(lens_flag_from_rusu16, "Rusu16")
_add_src(lens_flag_from_sqls, "SQLS_Inada12")
_add_src(lens_flag_from_veron, "Veron10")
_add_src(lens_flag_from_millon20, "Millon20")

lens_flag_coll_nogral = (
    lens_flag_confirmed |
    lens_flag_from_andika |
    lens_flag_from_ps1 |
    lens_flag_from_desi |
    lens_flag_from_williams |
    lens_flag_from_richards |
    lens_flag_from_dawes |
    lens_flag_from_rusu16 |
    lens_flag_from_sqls |
    lens_flag_from_veron |
    lens_flag_from_millon20)

lens_full = np.zeros(len(data_collection), dtype=bool)
lens_full[mask] = lens_flag_coll_nogral
src_full = np.array([""] * len(data_collection), dtype="U256")
src_full[mask] = provenance
tab = Table(data_collection)
tab["LENS"] = lens_full
tab["LENS_SRC"] = src_full
outpath = os.path.join(base_path, "COLLECTION_M277_v2_lensflags.fits")
tab.write(outpath, overwrite=True)

path = os.path.join(base_path, "COLLECTION_M277_v2_lensflags.fits")
tab = Table.read(path)
m = (np.array(tab["LENS"], dtype=bool)) & (np.array(tab["LENS_SRC"]).astype(str) != "")
ra  = np.array(tab["RA_old"], dtype=float)[m]
dec = np.array(tab["DEC_old"], dtype=float)[m]
src = np.array(tab["LENS_SRC"]).astype(str)[m]
order = np.lexsort((ra, dec))
ra, dec, src = ra[order], dec[order], src[order]
c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs", equinox="J2000")
ra_str  = c.ra.to_string(unit=u.hour, sep=" ", pad=True, precision=3)
dec_str = c.dec.to_string(unit=u.deg,  sep=" ", pad=True, alwayssign=True, precision=2)
for rs, ds, s in zip(ra_str, dec_str, src):
    print(f"{rs}  {ds}  {s}")
print('Total: ' + str(len(ra_str)))

print("GraL-only count:", np.sum((~np.array(tab["LENS"], bool)) & (np.array(tab["LENS_SRC"]).astype(str) == "GraL")))
print('Total: ' + str(tab["LENS"].astype('int').sum()))


















#%%

hdul = fits.open("C:/Users/fchit/Downloads/ListUniqueLenses.fits")
known_lens_guar = hdul[1].data

ra_lens_guar = known_lens_guar["RAd"]
dec_lens_guar = known_lens_guar["DECd"]
coords_known_lens_guar = SkyCoord(ra_lens_guar * u.deg, dec_lens_guar * u.deg, frame="icrs")

coords_clean = SkyCoord(ned_simbad_clean["ra"].values * u.deg, ned_simbad_clean["dec"].values * u.deg, frame="icrs")
group_ids_clean = ned_simbad_clean["group_id"].values
source_ids_clean = ned_simbad_clean["source_id"].astype(np.int64).values

cnn_flag = sept_predict >= 0.5
cnn_flag_map = {int(good_cutout_ids[i]): bool(cnn_flag[i]) for i in range(len(good_cutout_ids))}
cnn_flag_clean = np.array([cnn_flag_map.get(int(sid), False) for sid in source_ids_clean])

idx, sep, _ = coords_known_lens_guar.match_to_catalog_sky(coords_clean)
idx = idx.ravel()
sep = sep.ravel()
matched = sep < matching_rad_arcsec * u.arcsec

matched_clean_indices = idx[matched]
matched_group_ids = group_ids_clean[matched_clean_indices]

valid_known_groups = set([gid for gid in np.unique(matched_group_ids) if np.any(cnn_flag_clean[group_ids_clean == gid])])

print("Known lenses recovered:", len(valid_known_groups))
print(sorted(valid_known_groups))

hdul.close()















































