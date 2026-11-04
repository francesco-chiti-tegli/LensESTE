# QUBRICS: lensed and dual quasars

**An all-sky Gaia DR3 catalogue of close quasar pair candidates with imaging screening**

This repository holds the catalogue of Chiti Tegli et al. (2026): 1222 fields of Gaia DR3 quasar candidates with a neighbouring quasar candidate within 5 arcsec, at Galactic latitude |b| > 25 deg, screened in Legacy Surveys DR10 and Pan-STARRS1 images where these exist and given a priority for spectroscopic follow-up. The same tables, in fixed-width format with a CDS ReadMe, will be available at the CDS (VizieR).

## Files

The folder `Final Products` contains two comma-separated tables:

| File | Rows | Content |
|---|---|---|
| `qubrics_cqm_fields.csv` | 1222 | one row per field (pair or multiplet), with class, screening state, CNN scores, flags, cross-matches and spectroscopic outcomes |
| `qubrics_cqm_members.csv` | 2474 | one row per Gaia DR3 member, linked to its field by `field_id`, with Gaia data, NED and SIMBAD classes and spectroscopic classes and redshifts |

Empty cells are missing values; Boolean columns are `True` or `False`.

## Screening state and how to select candidates

Each field receives the first state of this list that applies:

| State | Fields | Meaning |
|---|---|---|
| `unassessed` | 45 | no usable cutout in either survey |
| `unresolved` | 208 | the largest separation is below the resolution limit (`resolution_limit_arcsec`) |
| `unpromising` | 48 | flagged as a likely contaminant at the strict cut in every survey that assesses the field |
| `uncertain` | 69 | flagged in one of two surveys, only at the loose cut, or rejected only by the second PS1 classifier |
| `caution` | 40 | not flagged, but a source more than ten times brighter than the central one lies in the cutout |
| `promising` | 812 | all other fields |

The states are a priority order for follow-up, not a selection: the 812 `promising` fields are the best candidates, and all the scores are given so that other cuts can be applied. Note that a *flag* marks a likely contaminant (a star, a galaxy or an artefact), so the candidates are the fields that are **not** flagged. For example, in Python:

```python
import pandas as pd

fields = pd.read_csv("Final Products/qubrics_cqm_fields.csv")
members = pd.read_csv("Final Products/qubrics_cqm_members.csv")

promising = fields[fields.state == "promising"]
promising_members = members[members.field_id.isin(promising.field_id)]
```

## Field classes

Fields are classified from the NED and SIMBAD classes of their members. A member is a quasar if at least one database gives Quasar/AGN, or if an entry of a lensed quasar lies within 2 arcsec; it is a non quasar if both give Star, Galaxy or Other; uncertain if one does and the other has no class; unknown if neither has a class. Fields with a non quasar member, or with uncertain members only, are not in the catalogue.

| Class | Fields | Meaning |
|---|---|---|
| `2QP` | 511 | all members are quasars |
| `2QI` | 47 | at least one member is a quasar and the others are unknown or uncertain; also fields without quasars that mix unknown and uncertain members, if at least one uncertain class comes from NED |
| `2QU` | 664 | all members are unknown; also fields without quasars that mix unknown and uncertain members, if all uncertain classes come from SIMBAD |

## Columns of `qubrics_cqm_fields.csv`

| Column | Unit | Description |
|---|---|---|
| `field_id` | | field identifier |
| `ra`, `dec` | deg | centre of the field (ICRS) |
| `n_members` | | number of Gaia DR3 quasar candidates in the field |
| `max_sep_arcsec` | arcsec | largest separation between members |
| `g_bright`, `g_faint` | mag | Gaia G magnitude of the brightest and of the faintest member |
| `field_class` | | `2QP`, `2QI` or `2QU` (see above) |
| `state` | | screening state (see above) |
| `resolution_limit_arcsec` | arcsec | 1.1625 for Legacy where a Legacy cutout exists, 1.2 for PS1 otherwise |
| `has_legacy`, `has_ps1` | | usable Legacy Surveys DR10 or Pan-STARRS1 cutout |
| `cnn_legacy`, `cnn_ps1` | | fine-tuned CNN score (1 = lens-like) |
| `flag_score_legacy`, `flag_score_ps1` | | flag score (1 = contaminant): mean probability of a gradient-boosted tree classifier and of a logistic regression with two-Moffat fit quantities at the Gaia positions and the CNN score as inputs; resolved fields only |
| `flag_legacy`, `flag_ps1` | | flagged at the strict cut, which flags 5% of the images visually judged lens-like |
| `flag_loose` | | flagged at the loose cut (10%) in a survey |
| `bright_neighbour` | | a source more than ten times brighter than the central one is in the cutout |
| `ps1_single_cnn` | | second PS1 classifier, a single CNN: 1 kept, 0 rejected, -1 no PS1 cutout |
| `known_lens` | | name of the known lensed quasar in the field; several names are separated by `;` |
| `gaia_fpr_grade` | | grade in the lens candidates of the Gaia Focused Product Release (Gaia Collaboration, 2024) |
| `other_catalogues` | | other catalogues listing the field, codes separated by `;` (below) |
| `w1_minus_w2` | mag | CatWISE2020 W1-W2 colour of the nearest source within 6 arcsec, which usually blends both members |
| `w1_minus_w2_blue` | mag | bluer W1-W2 of the members, when CatWISE2020 resolves both within 2 arcsec |
| `spec_outcome` | | outcome of the public spectra: `two quasars, same z` (abs(dz)/(1+z) < 0.0067), `two quasars, different z`, `one quasar, other without spectrum`, `contains a star`, `contains a galaxy`, `no spectrum` |
| `n_spec_members` | | members with a public spectroscopic class |
| `followup_outcome` | | outcome of our own long-slit spectra (Appendix A of the paper), for the seven fields observed with LDSS-3 and IMACS at Las Campanas Observatory in 2023 and with DOLORES at the Telescopio Nazionale Galileo in 2026: `two quasars, same z`, `contains a star`, `one quasar, other not observed` (only one member on the slit), `blended spectrum of a star` (unresolved field, both members on the slit); empty if not observed |

Codes in `other_catalogues`: L23 Lemon et al. (2023); H23 He et al. (2023); D23 Dawes et al. (2023); M23 Makarov et al. (2023); W24 Wu et al. (2024), DULAG; A23 Andika et al. (2023); C22 Chen et al. (2022), VODKA; H20 Hwang et al. (2020); D18 Ducourant et al. (2018); S21 Stern et al. (2021), GraL; H06 Hennawi et al. (2006); H10 Hennawi et al. (2010); M08 Myers et al. (2008); E17 Eftekharzadeh et al. (2017); G14 Gattano et al. (2014); C23 Chen et al. (2023); C26 Chen et al. (2026); Z26 Zhu et al. (2026); P24 Pfeifle et al. (2024), Big MAC.

## Columns of `qubrics_cqm_members.csv`

| Column | Unit | Description |
|---|---|---|
| `source_id` | | Gaia DR3 source identifier |
| `field_id` | | field identifier, as in the field table |
| `ra`, `dec` | deg | Gaia DR3 position (ICRS, epoch 2016.0) |
| `phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag` | mag | Gaia DR3 G, BP and RP magnitudes |
| `parallax_over_error` | | parallax divided by its error |
| `pm_over_error` | | total proper motion divided by its error |
| `redshift_qsoc` | | Gaia DR3 QSOC redshift |
| `has_xp` | | Gaia XP spectrum available |
| `class_ned`, `class_simbad` | | class of the nearest NED or SIMBAD counterpart within 5 arcsec: Quasar/AGN, Galaxy, Star, Other, Unclas IR/UV/X (unclassified X-ray, infrared or ultraviolet source) or Unknown (no counterpart) |
| `spec_class`, `spec_z`, `spec_source` | | public spectroscopic class and redshift within 1.5 arcsec, from DESI DR1 (redshift warning flag zero), SDSS DR16Q, LAMOST, 2QZ and Milliquas v8 (spectroscopic types only), in this order of priority; a spectrum within 1.5 arcsec of two members is not used |
| `followup_class`, `followup_z`, `followup_z_err`, `followup_instrument` | | class (`QSO` or `STAR`), redshift with its uncertainty and instrument of our own spectrum of the member (Table A.1 of the paper); for the unresolved field 639 the class refers to the blended spectrum of both members and is given for the brighter one |

