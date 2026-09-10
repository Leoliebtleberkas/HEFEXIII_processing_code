# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:36:18 2026

@author: leopo
"""

import xarray
import pandas
import os 
from pathlib import Path 

#working directory to load_verticalprofiles.py
os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\Lidar")

from load_verticalprofiles import load_vertical_profiles

#%% load data
folder_streamline = Path(r"D:\HEFEXIII\Streamline\L2\v1\SNR_CNS_filter")
folder_windranger = Path(r"D:\HEFEXIII\WindRanger\WindRanger")

pattern1 = r"VAD149_User3_RGL18_SNR24_CNS60_CN999_NVRAD4_R20.0_AVG5v04_202508\d{2}\.nc"
pattern2 = r"VAD149_User3_RGL36_SNR24_CNS60_CN999_NVRAD4_R20.0_AVG5v04_202508\d{2}\.nc"

data = load_vertical_profiles(folder_streamline = folder_streamline,
                              folder_windranger = None,
                              averaged_data = True,
                              file_pattern_streamline18 = pattern1,
                              file_pattern_streamline36 = pattern2)

ds_streamline18 = data["streamline18"].sortby("time")
ds_streamline36 = data["streamline32"].sortby("time")

#17.08. gate range change from 18 - 36: get time to prevent wrong data merging
folder_streamline2 = Path(r"D:\HEFEXIII\Streamline\L2\v0")
import xarray as xr
ds_18 = xr.open_dataset(folder_streamline2 / r"VAD149_User3_RGL18_SNR0_CNS60_AVG10v04_20250817_clean.nc")
ds_36 = xr.open_dataset(folder_streamline2 / r"VAD149_User3_RGL32_SNR0_CNS60_AVG10v04_20250817_clean.nc")

endtime_18 = ds_18.time[-1]
starttime_36 = ds_36.time[0]

ds_streamline18 = ds_streamline18.sel(time = slice(None, endtime_18))
ds_streamline36 = ds_streamline36.sel(time = slice(starttime_36, None))

#ds_windranger = data["windranger"].sortby("time")

#%% save as single datasets
savepath_streamline = Path(r"D:\HEFEXIII\Streamline\L2\v1\SNR_CNS_filter")
#savepath_windranger = Path(r"D:\HEFEXIII\WindRanger\L1")

ds_streamline18.to_netcdf(savepath_streamline / r"Streamline_VAD18m_raw.nc")
ds_streamline36.to_netcdf(savepath_streamline / r"Streamline_VAD36m_raw.nc")
#ds_windranger.to_netcdf(savepath_windranger / r"Windranger_avg.nc")


#%% try plotting
from windranger_visu import plot_timeheight, ms_to_knots
from plot_config import cmap_horwind

output_windranger = Path(r"D:\HEFEXIII\plots\Lidar\windranger\test")
output_streamline18 = Path(r"D:\HEFEXIII\plots\Lidar\streamline\VAD_streamline18\test")
output_streamline32 = Path(r"D:\HEFEXIII\plots\Lidar\streamline\VAD_streamline32\test")


#%% resample to 1 hr
#ds_windranger = ds_windranger.resample(time = "1h").mean()
ds_streamline18 = ds_streamline18.resample(time = "1h").mean()
ds_streamline32 = ds_streamline32.resample(time = "1h").mean()

#save them as well
savepath_streamline = Path(r"D:\HEFEXIII\Streamline")
savepath_windranger = Path(r"D:\HEFEXIII\WindRanger")

ds_streamline18.to_netcdf(savepath_streamline / r"Streamline_VAD18m_1h.nc")
ds_streamline32.to_netcdf(savepath_streamline / r"Streamline_VAD32m_1h.nc")
#ds_windranger.to_netcdf(savepath_windranger / r"Windranger_avg_1h.nc")
#%%calculations and plot

#%%windranger
ds_windranger["U_knots"] = ms_to_knots(ds_windranger["u"])
ds_windranger["V_knots"] = ms_to_knots(ds_windranger["v"])
ds_windranger["W_knots"] = ms_to_knots(ds_windranger["w"])
ds_windranger["VEL_knots"] = ms_to_knots(ds_windranger["spd"])
plot_kwargs = {"fontsize": 15, "figsize":(16,5)}
filled_contour_kwargs = {"levels":[0, 20], "spacing":1, 
                         "cbar_spacing":[0, 20], "label_spacing":[0, 20],
                         "extend":"max", "calc_norm":"yes", "cmap":cmap_horwind#coolwarm
                         }

plot_timeheight(ds_windranger, height_var = "alt", filled_contour_var = "spd",
                plot_output = output_windranger, 
                plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs)


#%%streamline18
ds_streamline18["U_knots"] = ms_to_knots(ds_streamline18["u"])
ds_streamline18["V_knots"] = ms_to_knots(ds_streamline18["v"])
ds_streamline18["W_knots"] = ms_to_knots(ds_streamline18["w"])
ds_streamline18["VEL_knots"] = ms_to_knots(ds_streamline18["spd"])
plot_kwargs = {"fontsize": 15, "figsize":(16,5)}
filled_contour_kwargs = {"levels":[0, 20], "spacing":1, 
                         "cbar_spacing":[0, 20], "label_spacing":[0, 20],
                         "extend":"max", "calc_norm":"yes", "cmap":cmap_horwind#coolwarm
                         }

plot_timeheight(ds_streamline18, height_var = "alt", filled_contour_var = "spd",
                plot_output = output_streamline18, 
                plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs)


#%%streamline32
ds_streamline32["U_knots"] = ms_to_knots(ds_streamline32["u"])
ds_streamline32["V_knots"] = ms_to_knots(ds_streamline32["v"])
ds_streamline32["W_knots"] = ms_to_knots(ds_streamline32["w"])
ds_streamline32["VEL_knots"] = ms_to_knots(ds_streamline32["spd"])
plot_kwargs = {"fontsize": 15, "figsize":(16,5)}
filled_contour_kwargs = {"levels":[0, 20], "spacing":1, 
                         "cbar_spacing":[0, 20], "label_spacing":[0, 20],
                         "extend":"max", "calc_norm":"yes", "cmap":cmap_horwind#coolwarm
                         }

plot_timeheight(ds_streamline32, height_var = "alt", filled_contour_var = "spd",
                plot_output = output_streamline32, 
                plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs)
