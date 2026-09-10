----
autocorrelation.py
calculates autocorrelation between u,v,w for multiple 3d Sonics

----
flux_correction_FUNCTIONS.py
storage for functions, for calculating dynamic sensible heat and dynamic latent heat flux and applying common flux corrections (SND and WPL)

----
functions.py
storage for base functions used in turbulence postprocessing, e.g. linear detrending or calculation of fluctuations

----
general_analysis_plots.py
some basic analyses and visualizations

----
MRD_functions.py
storage for functions, needed for the implementation of the Multiresolution Decomposition in the original turbulence postprocessing workflow (scripts postprocess.py and main.py), in reality never used

----
MRD_main.py
calculation of the MRDs for HEFEXIII data

----
postprocess.py
original main script for executing the turbulence postprocessing workflow written by Samuele Mosso, University of Innsbruck. Also storage for some subfunction needed in this workflow, e.g. gap filling, calculation of turbulent fluxes, co-variances or stationarity. Adapted by Leopold Schlagbauer, execution of the turbulence postprocessing workflow was then moved to main.py

----
main.py
main script for executing the turbulence postprocessing workflow, adapted from the original postprocess.py. Requires a .txt config file

---- 
config_{}.txt
config file with Settings for the turbulence postprocessing, such as averaging intervall or averaging method etc. 

----
spectral_analysis.py
subfunctions for calculating spectra of the turbulence data

----
structure_functions.py
subfunctions for calculating structure functions and deriving Dissipation rates


