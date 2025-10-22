# LensESTE
INAF Trieste QUBRICS Lens project.

## LensESTE-NEW
The most recent update pre-resumbission is in the folder LensESTE-NEW, and contains the catalogues, the codes for the catalogue query pre-CNN (Confronto_SIMBAD_Catalogue.py) and the post-CNN classification (CNNParteII.py).
The final post-CNN catalogue of 903 candidate lenses is in Catalogue_post_CNN.csv
The original catalogue by Francesco Guarneri is in Catalogue.csv

## LensESTE-OLD
The codes for the old version are in LensESTE-OLD.
The CNN codes (November 2024) are displayed with "september" or "settembre" in their name.
The "TRI" codes are for three-channels image with y,i,r bands.
The "open" code is for the training of the model, while "carica" is for the testing with candidates cutouts.
boost.txt is the boosting list for the second train.
boost_perfect_candidates.txt is the list of 670 perfect cutouts with the boosted model.
boost_ALL_candidates.txt is the list of all candidates, except the corrupted or blank ones.
