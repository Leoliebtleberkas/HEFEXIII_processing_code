### Description

This repository contains the python code, which was used for processing the the field data collected during the third HinterEisFerner EXperiment (HEFEXIII) on Hintereisferner, Tyrol Austria in summer 2025. The data itself is stored on Zenodo (add link). Each of the directories contains a seprate READme file, which describes the single python scripts. Care should be taken with hardcoded paths (mostly pointing to directories where data is stored), which appear in many of the scripts and need to be adjusted by the user (sorry for bad coding style). This should however easily be fixed. 

The following gives a short overview of the single sub-dircetories.

#### Lidar
Scripts for processing and analyzing the data of the two Doppler wind lidars, which were operated during HEFEXIII
#### UAV
Scripts for processing and analyzing the data of the unmanned aircraft systems (UAVs), which were operated during HEFEXIII
#### helper functions
storage of scripts, which contain very general functionalities applicable to more than one type of observation data
#### tower
Scripts for processing and analyzing the data of the 9 m tall turbulence tower (named T272), which was operated during HEFEXIII. Contains scripts for file conversion, coordinate rotations and quality control
#### turbulence_postprocessing
Scripts for processing of turbulence data of the 5 3D sonic anemometers from T272, including preparatory steps like gap filling and detrending, calculation of turbulent fluxes and flux corrections and calculation of other derived turbulence properties such as e.g. (Co-)spectra.  
