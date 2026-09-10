----
h5_to_nc.py
conversion of .h5 files to .nc. Was used for converting all L1 files to .nc

----
smartflux_to_nc.py
conversion of the raw data of LiCor and Metek which was logged by the Smartflux (.ghg files) to .nc files, creates the L1 Version of the 1m Metek data

----
metek_intercomparison.py
quick Analysis of a sensor intercomparison conducted after the HEFEXIII Campaign. One sensor (the one at 3 m during HEFEXIII) had a temperature bias of +1 K, which was corrected in this script as well! Output of this script was used as Input for coordinate rotations!

----
coordinate_rotation_xr_main.py
1st part rotates wind data of metek3D sonics and gill2D sonics into a cartesian and northfacing coordinate System --> u eastwards wind, v northwards wind, w vertical wind (pointing upwards). The result is then used for planar fit rotation and turbulence postprocessing.
2nd part applies Quality Control on all Tower data and creates data for a General Analysis of Tower data

----
coordinate_rotation_xr_functions.py
helper functions for main script

----
planarfit.py
helper functions and main function for applying planar fit Rotation on wind components of the L2 dataset (the result from coordinate_rotation_xr_main). Also applies Quality Control after Rotation. This is important for wind, mainly for the vertical, because some observations, which are appearing as outliers before the Rotation are actually "cleaned" by the Rotation into a mean streamline System and would falsely be kicked out if Quality Control applied before planar fit Rotation. 
Example: We observe a strong gust of 20 m/s along the slope, i.e. parallel to the slope. Slope angle is 6° and hence this would be observed as a Peak of ~ -2 m/s = 20 m/s * sin(-6) in the w component of the Input data, which is in a cartesian, northfacing and perfectly horizontally aligned coordinate System. 
If the Quality Control would be applied before the planar fit rotation, it would likely kick out this value. But for turbulence processing later on we are interested in the along-slope flow, i.e. w is normal to the slope. In this coordinate System w is 0 in that case and kicking out this value would be without reason. Hence Quality Control is applied after planar fit Rotation.
