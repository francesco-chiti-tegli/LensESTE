# LensESTE
INAF Trieste QUBRICS Lenses project.

## Final Products
The final catalogues of the QUBRICS multiplets project, with both the full catalogue and the CNN-positively-flagged one.

## LensESTE-NEW
The most recent update pre-resumbission is in the folder LensESTE-NEW, and contains the intermediate catalogues, the codes for the catalogue query pre-CNN (Confronto_SIMBAD_Catalogue.py) and the post-CNN classification (CNNParteII.py).

The original catalogue by Francesco Guarneri is in Catalogue.csv

## LensESTE-OLD
The codes for the old version are in LensESTE-OLD.
The CNN codes (November 2024) are displayed with "september" or "settembre" in their name.
The "TRI" codes are for three-channels image with y,i,r bands.
The "open" code is for the training of the model, while "carica" is for the testing with candidates cutouts.
boost.txt is the boosting list for the second train.
boost_perfect_candidates.txt is the list of 670 perfect cutouts with the boosted model.
boost_ALL_candidates.txt is the list of all candidates, except the corrupted or blank ones.
