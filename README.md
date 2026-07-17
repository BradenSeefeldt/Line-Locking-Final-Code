This repo holds all of the analysis code for the "Radiation pressure powers quasar broad absorption
line winds but fails to drive galaxy feedback" paper. The contents are as follows:

- LineLocking_FinalVersion.py: library containing most of the functions used in the analysis including mock spectra creation, KDE info stripping, column density calculations, etc

- LineLocking_Notebook_final.ipynb: Primary notebook that reads in raw spectra, extracts info, creates KDEs, creates mock spectra, stacks.

- Total_Absorbed_Flux.ipynb: Notebook for calculating the mass outflow rate.

- cluster.ipynb: Notebook to explore clustering on the 2 outflow populations.

- Dust_Reddening.ipynb: Notebooke dedicated to exploring continuum absorption

- Wind_Kinematics.ipynb: Notebooke to model the wind outflow and show the line locking only occurs over a small velocity range.
