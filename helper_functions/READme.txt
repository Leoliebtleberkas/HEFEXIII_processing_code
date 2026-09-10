----
calculation_functions.py
storage for helper functions used for calculations, e.g. from derived (thermodynamic) variables 

----
planarfit_xr_functions.py
storage for functions needed for planar fit rotations after Wilczak (2001). Used if the main.py in the turbulence_postprocessing Directory is executed with unrotated data. If you want to execute planar fit Rotation for a dataset directly you also have the (better) Option to do this with the planarfit_main.py in the "tower" Directory

----
postprocess_plots.py
storage for some plotting routines, e.g. of spectra. was not really used for HEFEXIII until now.

----
quality_control.py
storage for the Quality Control subroutines such as threshold cleaning or the despiking algorithm. Used in the general tower processing and turbulence postprocessing