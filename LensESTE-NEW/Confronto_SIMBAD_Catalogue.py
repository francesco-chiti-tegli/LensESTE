# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 16:54:28 2025

@author: Francesco Chiti Tegli
"""

import numpy as np
import ast
import os
import pandas as pd
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ned import Ned
from astroquery.simbad import Simbad
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

base_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/BaseDirGuarneri/"
work_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/"
NED_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/NEDGuarneri"
SIMBAD_path = "C:/Users/fchit/OneDrive/Desktop/ComplementiAstronomia/GuaneriResume/SIMBADGuarneri"
df = pd.read_csv(os.path.join(base_path, "Catalogue.csv"))
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors='ignore')

dups_coords = df.duplicated(subset=['ra','dec']).sum()
dups_rows  = df.duplicated().sum()
print("exact duplicate (ra,dec) rows:", dups_coords)
print("exact duplicate entire rows:", dups_rows)

coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg)
xyz = np.vstack([coords.cartesian.x, coords.cartesian.y, coords.cartesian.z]).T
tree = cKDTree(xyz)

r = np.deg2rad(4.95635/3600)
neighbors = tree.query_ball_tree(tree, r)

labels = -np.ones(len(df), dtype=int)
group_id = 0
for i in range(len(df)):
    if labels[i] == -1:
        # flood fill
        stack = [i]
        while stack:
            idx = stack.pop()
            if labels[idx] == -1:
                labels[idx] = group_id
                for j in neighbors[idx]:
                    if labels[j] == -1:
                        stack.append(j)
        group_id += 1

df['group_id'] = labels
n_groups_total = df['group_id'].nunique()
group_sizes = df['group_id'].value_counts().sort_index()

print("Total connected components (all sizes):", n_groups_total)
print("Groups with size >= 2:", (group_sizes >= 2).sum())
print("Distribution (size -> n_groups):")
print(group_sizes.value_counts().sort_index())

#%% (1) Matching with NED (DO NOT RUN AGAIN!)

def download_ned(df, id_col, ra_col, dec_col, path, radius_arcsec=5.0):
    os.makedirs(path, exist_ok=True)
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        _id, ra, dec = row[id_col], row[ra_col], row[dec_col]
        coord = SkyCoord(ra*u.deg, dec*u.deg)
        url = (f"http://ned.ipac.caltech.edu/cgi-bin/objsearch?"
               f"search_type=Near+Position+Search&in_csys=Equatorial&in_equinox=J2000.0"
               f"&lon={ra}d&lat={dec}d&radius={radius_arcsec/60.0}&out_csys=Equatorial"
               f"&out_equinox=J2000.0&img_stamp=YES")
        nedcount, nedtype, nedz, ned_lol = 0, None, None, []
        try:
            res = Ned.query_region(coord, radius=radius_arcsec*u.arcsec).to_pandas()
            if res is not None and len(res) > 0:
                nedcount = len(res)
                ra_colname = next((c for c in res.columns if "RA" in c), None)
                dec_colname = next((c for c in res.columns if "DEC" in c), None)
                if ra_colname and dec_colname:
                    seps = coord.separation(SkyCoord(res[ra_colname].values*u.deg, res[dec_colname].values*u.deg))
                    for sep, typ in zip(seps.arcsec, res.get("Type", [""]*len(res))):
                        ned_lol.append((float(sep), str(typ)))
                    idx_closest = seps.argmin()
                    nedtype = str(res.get("Type", ["Unknown"])[idx_closest])
                    nedz = str(res.get("Redshift", [None])[idx_closest])
                    res.to_csv(os.path.join(path, f"NED_id-{_id}_ra{ra:.3f}_dec{dec:.3f}.csv"), index=False)
        except Exception as e:
            print(f"[!] Errore su ID {_id}: {type(e).__name__} -> {e}")
            continue
        results.append({
            "_id": _id,
            "ra": ra,
            "dec": dec,
            "ned_url": url,
            "nedcount": nedcount,
            "nedtype": nedtype,
            "nedz": nedz,
            "ned_sep_arcsec": ned_lol})
    return pd.DataFrame(results)
ned_matches = download_ned(df, "source_id", "ra", "dec", path=NED_path)
ned_matches.to_csv(os.path.join(work_path, "ned_crossmatch.csv"), index=False)
print(len(ned_matches), "oggetti scaricati da NED")


#%% (1.5) Opening NED matches

ned_matches = pd.read_csv(os.path.join(work_path, "ned_crossmatch.csv"))
ned_matches[ned_matches["nedcount"] > 0][["nedtype", "nedz"]].head()

ned_matches.head()
print()
already_in_ned = ned_matches[ned_matches["nedcount"] > 0]
print("Already in NED: " + str(len(already_in_ned)))
new_candidates = ned_matches[ned_matches["nedcount"] == 0]
print("New canidates: " + str(len(new_candidates)))
print()

def classify_from_nedtype(t):
    if pd.isna(t) or not isinstance(t, str) or t.strip() == "":
        return "Unknown"
    
    t = t.upper()
    types = [x.strip() for x in t.split(",")]
    
    if any(q in types for q in ["QSO", "QGROUP", "Q_LENS"]):
        return "Quasar/AGN"
    if any(star in types for star in ["*", "V*", "WD*", "BLUE*"]):
        return "Star"
    if "G" in types and all(x not in ["GGROUP", "GCLSTR", "G_LENS", "GPAIR", "GCL", "GLENS", "QSO", "Q_LENS", "QGROUP"] for x in types):
        return "Galaxy"
    if any(x in types for x in ["IRS", "UVS", "XRAYS", "VIS", "VISS"]):
        return "Unclas IR/UV/X"
    return "Other"

ned_matches["class_ned"] = ned_matches["nedtype"].apply(classify_from_nedtype)
ned_matches["class_ned"] = ned_matches["class_ned"].fillna("Other")
print(ned_matches["class_ned"].value_counts())

ned_matches["ned_sep_arcsec"] = ned_matches["ned_sep_arcsec"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)

plt.figure(figsize=(8,5))
for cls in ["Quasar/AGN", "Star", "Galaxy"]:
    seps = []
    for matches, c in zip(ned_matches["ned_sep_arcsec"], ned_matches["class_ned"]):
        if c == cls and isinstance(matches, list) and len(matches) > 0:
            seps.extend([float(sep) for sep, _ in matches if isinstance(sep, (float, int))])
    if seps:
        plt.hist(seps, bins=np.linspace(0, 5, 40), alpha=0.8, label=cls)

plt.xlabel("Angular distance [arcsec]")
plt.ylabel("Matches")
plt.yscale("log")
plt.title("Average distance distribution for each class - NED")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#%% (2) Matching with SIMBAD (DO NOT RUN AGAIN)

def download_simbad(df, id_col, ra_col, dec_col, path, radius_arcsec=5.0):
    os.makedirs(path, exist_ok=True)
    results = []
    sim = Simbad()
    sim.TIMEOUT = 60
    sim.reset_votable_fields()
    sim.add_votable_fields('otype', 'rvz_redshift')
    for _, row in tqdm(df.iterrows(), total=len(df)):
        _id, ra, dec = row[id_col], row[ra_col], row[dec_col]
        coord = SkyCoord(ra*u.deg, dec*u.deg)
        url = (f"https://simbad.u-strasbg.fr/simbad/sim-coo?"
               f"Coord={ra:.6f}+{dec:.6f}&Radius={radius_arcsec}&Radius.unit=arcmin")
        simcount = 0
        simtype = None
        simz = None
        sim_matches = None
        
        try:
            res = sim.query_region(coord, radius=radius_arcsec*u.arcsec)
            if res is not None and len(res) > 0:
                simcount = len(res)
                seps = coord.separation(SkyCoord(res['ra'], res['dec'], unit=(u.hourangle, u.deg)))
                idx_closest = seps.argmin()

                sim_matches = []
                for i in range(len(res)):
                    otype = str(res['otype'][i]) if 'otype' in res.colnames else ''
                    sim_matches.append((seps[i].arcsec, otype))
                
                if "otype" in res.colnames:
                    simtype = str(res["otype"][idx_closest])
                if "rvz_redshift" in res.colnames:
                    simz = str(res["rvz_redshift"][idx_closest])
                res.write(os.path.join(path, f"SIMBAD_id-{_id}_ra{ra:.3f}_dec{dec:.3f}.csv"), overwrite=True)
        except Exception:
            pass
        results.append({
            "_id": _id,
            "ra": ra,
            "dec": dec,
            "simbad_url": url,
            "simbadcount": simcount,
            "simbad_type": simtype,
            "simbad_z": simz,
            "simbad_sep_arcsec": sim_matches if sim_matches is not None else []})
    return pd.DataFrame(results)

simbad_matches = download_simbad(df, "source_id", "ra", "dec", path=SIMBAD_path)
simbad_matches.to_csv(os.path.join(work_path, "simbad_crossmatch.csv"), index=False)
print(len(simbad_matches), "objects in the catalog")

def find_ambiguous_matches(df):
    ambiguous_rows = []
    for idx, row in df.iterrows():
        matches = row['simbad_sep_arcsec']
        if matches and len(matches) > 1:
            types = set([t for _, t in matches])
            if len(types) > 1:
                ambiguous_rows.append({"_id": row['_id'], "ra": row['ra'], "dec": row['dec'],
                    "simbadcount": row['simbadcount'],
                    "types_found": types,
                    "matches": matches})
    return pd.DataFrame(ambiguous_rows)

ambiguous_simbad = find_ambiguous_matches(simbad_matches)
print(len(ambiguous_simbad), "sources with ambiguous matches")

#%% (2.5) Opening SIMBAD matches

simbad_matches = pd.read_csv(os.path.join(work_path, "simbad_crossmatch.csv"))
already_in_simbad = simbad_matches[simbad_matches["simbadcount"] > 0]
print("Already in SIMBAD:", len(already_in_simbad))
new_candidates = simbad_matches[simbad_matches["simbadcount"] == 0]
print("New candidates:", len(new_candidates))
print()

def classify_from_simbadtype(matches):
    if not isinstance(matches, (list, tuple)) or len(matches) == 0:
        return "Unknown"
    
    matches_sorted = sorted(matches, key=lambda x: x[0])
    
    def classify_type(t):
        if pd.isna(t) or not isinstance(t, str) or t.strip() == "":
            return "Unknown"
        t = t.upper()
        if any(x in t for x in ["QSO", "Q?", "Q", "AGN", "Sy1", "Sy2", "BLL", "LeQ", "LeI", "LS?", "LI?", "Le?"]):
            return "Quasar/AGN"
        elif any(x in t for x in ["G", "PoG", "GAL", "EMG", "ClG", "GRG", "POG", "H2G", "GIC", "PaG", "GLC", "LeG"]):
            return "Galaxy"
        elif any(x in t for x in ["*", "**", "Cl*", "RR*", "V*", "WD", "WD*", "SN*", "HB*", "CE*", "AS*", "EM*", "GL?", "YS*?", "RR*", "RR?", "LP*", "LM*", "HII"]):
            return "Star"
        elif any(x in t for x in ["X", "UV", "IR", "FIR", "SMM", "OPT", "SH"]):
            return "Unclas IR/UV/X"
        else:
            return "Other"
    
    first_sep, first_type = matches_sorted[0]
    first_class = classify_type(first_type)
    
    if first_class == "Galaxy" and any(x in first_type.upper() for x in ["GLE", "GLS", "GLS"]):
        for i in range(1, min(3, len(matches_sorted))):
            sep_i, type_i = matches_sorted[i]
            class_i = classify_type(type_i)
            
            if class_i == "Quasar/AGN":
                return "Quasar/AGN"
            elif class_i == "Galaxy" and i == 2:
                sep_second = matches_sorted[1][0]
                if abs(sep_i - sep_second) <= 0.05:
                    return "Quasar/AGN"
                else:
                    return "Galaxy"
        return "Unclas IR/UV/X"
    
    return first_class

simbad_matches["simbad_sep_arcsec"] = simbad_matches["simbad_sep_arcsec"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else [])
simbad_matches["class_simbad"] = simbad_matches["simbad_sep_arcsec"].apply(classify_from_simbadtype)
print(simbad_matches["class_simbad"].value_counts())

def parse_simbad_z(val):
    try:
        z = float(val)
        if z < 0 or z > 10:
            return None
        return z
    except:
        return None

simbad_matches["simbad_z_clean"] = simbad_matches["simbad_z"].apply(parse_simbad_z)

all_seps = []
for matches in simbad_matches["simbad_sep_arcsec"]:
    if isinstance(matches, list) and len(matches) > 0:
        all_seps.extend([sep for sep, _ in matches])
all_seps = np.array(all_seps)
mean_seps = []
for matches in simbad_matches["simbad_sep_arcsec"]:
    if isinstance(matches, list) and len(matches) > 0:
        mean_seps.append(np.mean([sep for sep, _ in matches]))
mean_seps = np.array(mean_seps)

plt.figure(figsize=(8,5))
for cls in ["Quasar/AGN", "Star", "Galaxy"]:
    seps = []
    for matches, c in zip(simbad_matches["simbad_sep_arcsec"], simbad_matches["class_simbad"]):
        if c == cls and isinstance(matches, list) and len(matches) > 0:
            seps.extend([sep for sep, _ in matches])
    if seps:
        plt.hist(seps, bins=np.linspace(0, 5, 40), alpha=0.8, label=cls)

plt.xlabel("Average distance [arcsec]")
plt.ylabel("Matches")
plt.yscale('log')
plt.title("Average distance distribution for each class - SIMBAD")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#%% (3) Merging the two catalogue queries

merged = pd.merge(ned_matches, simbad_matches, on="_id", how="outer", suffixes=("_ned", "_simbad"))

in_ned = merged["nedcount"].fillna(0) > 0
in_simbad = merged["simbadcount"].fillna(0) > 0

only_ned = in_ned & ~in_simbad
only_simbad = in_simbad & ~in_ned
both = in_ned & in_simbad
neither = ~in_ned & ~in_simbad

print(f"Only in NED: {only_ned.sum()}")
print(f"Only in SIMBAD: {only_simbad.sum()}")
print(f"In both: {both.sum()}")
print(f"In neither: {neither.sum()}")

class_order = ['Unclas IR/UV/X', 'Galaxy', 'Star', 'Quasar/AGN', 'Other', 'Unknown']
matrix = pd.crosstab(merged["class_simbad"], merged["class_ned"],).reindex(index=class_order, columns=class_order, fill_value=0)

plt.figure(figsize=(8, 6))
im = plt.imshow(matrix, cmap="viridis")
for i in range(len(class_order)):
    for j in range(len(class_order)):
        plt.text(j, i, matrix.iloc[i, j], ha="center", va="center", color="black" if matrix.iloc[i, j] > matrix.values.max()/2 else "white")

plt.xticks(range(len(class_order)), class_order, rotation=45, ha="right")
plt.yticks(range(len(class_order)), class_order)
plt.xlabel("NED class")
plt.ylabel("SIMBAD class")
plt.title("Cross-classification of candidates (SIMBAD vs NED)")
plt.colorbar(im, label="Cadnidate number")
plt.gca().invert_yaxis()

n = len(class_order)
rect_x = [0 - 0.5, n - 2 + 0.5, n - 2 + 0.5, 0 - 0.5, 0 - 0.5]
rect_y = [n - 1 - 0.5, n - 1 - 0.5, 0 - 0.5, 0 - 0.5, n - 1 - 0.5]

plt.plot(rect_x, rect_y, color="red", linewidth=2)

plt.tight_layout()
plt.show()

merged.to_csv(os.path.join(work_path, "merged_NED_SIMBAD_catalog.csv"), index=False)


#%% (4) Create a file that stores the 

df_save = df.merge(ned_matches[["_id", "class_ned"]], left_on="source_id", right_on="_id", how="left")
df_save_II = df_save.merge(simbad_matches[["_id", "class_simbad"]], left_on="source_id", right_on="_id", how="left")

df_save_II = df_save_II.drop(columns=["_id_x", "_id_y"], errors='ignore')

df_save_II.to_csv(os.path.join(work_path, "Catalogue_after_queries.csv"), index=False)

#%% (5) Objects distribution

def candidate_color(row):
    if row["class_ned"] == "Quasar/AGN" and row["class_simbad"] == "Quasar/AGN":
        return "green"
    elif row["class_ned"] == "Quasar/AGN" and row["class_simbad"] != "Quasar/AGN":
        return "yellow"
    elif ((row["class_ned"] == "Unclassified IR/UV/X" and row["class_simbad"] == "Unknown")
          or (row["class_ned"] == "Unknown" and row["class_simbad"] == "Unknown")):
        return "red"
    else:
        return "pink"

merged['color'] = merged.apply(candidate_color, axis=1)
df['color'] = df['source_id'].map(merged.set_index('_id')['color'])
df_plot = df[df['color'] != 'pink']

plt.figure(figsize=(8, 6))
plt.scatter(df_plot['ra'], df_plot['dec'], c=df_plot['color'], s=30, edgecolor='k', alpha=0.8)

midis_patch = mpatches.Rectangle(
    (53.15, -27.80),
    0.05,
    0.04,
    linewidth=1.5, edgecolor='red', facecolor='red',
    alpha=0.3, label='MIDIS')
plt.gca().add_patch(midis_patch)
jades_s_patch = mpatches.Rectangle(
    (53.125 - 0.025, -27.80 - 0.02),
    0.05, 0.04,
    angle=-61.5,
    linewidth=1.5,
    edgecolor='blue',
    facecolor='blue',
    alpha=0.3,
    label='JADES-S'
)
plt.gca().add_patch(jades_s_patch)
jades_n_patch = mpatches.Rectangle(
    (189.23 - 0.025, 62.23 - 0.02),
    0.05, 0.04,
    angle=-119,
    linewidth=1.5,
    edgecolor='green',
    facecolor='green',
    alpha=0.3,
    label='JADES-N'
)
plt.gca().add_patch(jades_n_patch)

plt.xlabel("RA [deg]")
plt.ylabel(r"$\delta$ [deg]")
plt.title("Candidate positions")

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Quasar NED+SIMBAD', markerfacecolor='green', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Quasar only NED', markerfacecolor='yellow', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Unknown / Unclassified', markerfacecolor='red', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor='pink', markersize=8),
    mpatches.Patch(facecolor='red', edgecolor='red', alpha=0.3, label='MIDIS'),
    mpatches.Patch(facecolor='blue', edgecolor='blue', alpha=0.3, label='JADES-S'),
    mpatches.Patch(facecolor='green', edgecolor='green', alpha=0.3, label='JADES-N')]

plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()






