----
merge_lidar.py
script for merging the original daily datafiles of WindRanger200 and StreamlineXR data (currently PPI/VAD Scans only) into one datafile for every device. uses subroutines from load_verticalprofiles.py for more conveniently loading the daily files all at once

----
load_vertical_profiles.py
subroutines for loading the daily files of WindRanger200 and StreamLine data  

----
lidar_cleaning.py
main file for cleaning PPI (=VAD) scan data of lidar data using the functions stored in lidar_cleaning_functions.py. Also produces time averaged datafiles and datafiles combining data from the WindRanger200 (lower Levels) and the StreamlineXR (upper Levels)

----
lidar_cleaning_functions.py
subroutines for lidar data quality Control.

----
lidar_analysis.py
main script for lidar analysis, currently only for selecting a case study and plotting time-height diagrams of lidar data, relying on plotting routines from windranger_visu.py

----
windranger_visu.py
plotting routines for lidar data (not only windranger, also streamline)

----
plot_config.py
definition and storage of lidar colormaps, currently only for horizontal wind speed and vertical wind Speed

----
plot_base_functions.py
storage of plotting subroutines, currently only ued for time_height plots of lidar data