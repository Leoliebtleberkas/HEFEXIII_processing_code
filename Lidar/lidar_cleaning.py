# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:07:15 2026

@author: leopo
"""

#import packages
import xarray as xr
import numpy as np
import pandas as pd
import os 
from pathlib import Path
from matplotlib import pyplot as plt
from metpy.calc import wind_direction, wind_speed
from metpy.units import units

os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\Lidar")

from lidar_cleaning_functions import *
from windranger_visu import plot_timeheight, ms_to_knots
from plot_config import cmap_horwind

#%% ---- WINDRANGER ----
#read data
folder_windranger = Path(r"D:\HEFEXIII\WindRanger")
ds_windranger = xr.open_dataset(folder_windranger / r"Windranger.nc")


# simple plots for overview
output_windranger = Path(r"D:\HEFEXIII\plots\Lidar\windranger\test_plots")
plot_vars = ["u", "SNR", "spd"]

#test_plots(ds_windranger, plot_vars, output_plot = output_windranger)
    

#%% windranger cleaning

#threshold control 
ds_windranger_cleaned = threshold_control(ds_windranger.copy(), 
                                          threshold_u = 25, threshold_v = 25,
                                          threshold_spd = 25,
                                          threshold_quality = 1,
                                          threshold_valid = 0.6)
                                          #threshold_snr = -24,
                                          #)
                                          

#additional despiking
ds_windranger_cleaned, nan_counter = despiking(ds_windranger_cleaned, 
                                               n_start=3, alt_despike = True, 
                                               n_height_start = 2, window = "1h")

#simple plots again
#test_plots(ds_windranger_cleaned, plot_vars = plot_vars, 
#           output_plot = output_windranger, file_ext = "cleaned_avg")

#save cleaned windranger data
ds_windranger_cleaned.to_netcdf(folder_windranger / r"Windranger_cleaned.nc")

#%% averaging and timeheight plots 10min data
os.chdir(r"C:\Users\leopo\PhD\glacier_space\code\Lidar")


output_windranger = Path(r"D:\HEFEXIII\plots\Lidar\windranger\timeheight_10mincustom")

#10min
ds_windranger_cleaned_10min = resample_and_save(ds_windranger_cleaned, "10min",
                                                output_path = folder_windranger,
                                                savename = "Windranger_cleaned")

#30min
ds_windranger_cleaned_30min = resample_and_save(ds_windranger_cleaned, "30min",
                                                output_path = folder_windranger,
                                                savename = "Windranger_cleaned")
#1h
ds_windranger_cleaned_1h = resample_and_save(ds_windranger_cleaned, "1h",
                                             output_path = folder_windranger,
                                             savename = "Windranger_cleaned")

#----settings horwind-----
plot_kwargs = {"fontsize": 20, "figsize":(20,10)}
filled_contour_kwargs = {"levels":[0, 20], "spacing":1, 
                         "cbar_spacing":[0, 20], "label_spacing":5,#[0, 20],
                         "extend":"max", "calc_norm":"yes", "cmap":cmap_horwind#coolwarm
                         }

for day, ds_day in ds_windranger_cleaned_10min.groupby("time.date"):
    print(day)
    print(f"u_max: {ds_day["u"].max()}")
    print(f"v_max: {ds_day["v"].max()}")
    #ds_day = ds_windranger_cleaned.sel(time = day)
    plot_timeheight(ds_day, height_var = "alt", filled_contour_var = "spd",
                plot_output = output_windranger, barbs = True, barb_unit="m/s",
                plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs)
  
#%%----settings vertical velocity-----
plot_kwargs = {"fontsize": 20, "figsize":(20,10)}
filled_contour_kwargs = {"levels":[-5, 5], "spacing":1, 
                         "cbar_spacing":[-5, 5], "label_spacing":1,#[0, 20],
                         "extend":"both", "calc_norm":"no", "cmap": "coolwarm"
                         }

for day, ds_day in ds_windranger_cleaned_10min.groupby("time.date"):
    print(day)
    #ds_day = ds_windranger_cleaned.sel(time = day)
    plot_timeheight(ds_day, height_var = "alt", filled_contour_var = "w",
                plot_output = output_windranger, barbs = True, barb_unit="m/s",
                plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs)
    
#----settings SNR-----
plot_kwargs = {"fontsize": 20, "figsize":(20,10)}
filled_contour_kwargs = {"levels":[-24, 24], "spacing":4, 
                         "cbar_spacing":[-24, 24], "label_spacing":4,#[0, 20],
                         "extend":"both", "calc_norm":"no", "cmap":"viridis"
                         }

for day, ds_day in ds_windranger_cleaned_10min.groupby("time.date"):
    if len(ds_day.time) < 2:
        print(f"no data for {day}...skip")
        continue
    print(day)
    #ds_day = ds_windranger_cleaned.sel(time = day)
    plot_timeheight(ds_day, height_var = "alt", filled_contour_var = "SNR",
                plot_output = output_windranger, barbs = True, barb_unit="m/s",
                plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs)
    
    
# ----settings VALID Radial components-----
plot_kwargs = {"fontsize": 20, "figsize":(20,10)}
filled_contour_kwargs = {"levels":[0, 0], "spacing":0.05, 
                         "cbar_spacing":[0, 0], "label_spacing":0.2,
                         "extend":"both", "calc_norm":"no", "cmap":"viridis"
                         }

for day, ds_day in ds_windranger_cleaned_10min.groupby("time.date"):
    if len(ds_day.time) < 2:
        print(f"no data for {day}...skip")
        continue
    print(day)
    #ds_day = ds_windranger_cleaned.sel(time = day)
    plot_timeheight(ds_day, height_var = "alt", filled_contour_var = "VALID",
                plot_output = output_windranger, barbs = True, barb_unit="m/s",
                plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs)
    
#%% ----- STREAMLINE -----
#read data
folder_streamline = Path(r"D:\HEFEXIII\Streamline\L2\v1\SNR_CNS_filter") 

ds_streamline18 = xr.open_dataset(folder_streamline / r"Streamline_VAD18m_raw.nc")
ds_streamline32 = xr.open_dataset(folder_streamline / r"Streamline_VAD36m_raw.nc")

#%% raw data resampled to 10 min for ploting
ds_streamline18_10min = resample_and_save(ds_streamline18, "10min",
                                          output_path = folder_streamline,
                                          savename = "Streamline18_raw")

ds_streamline32_10min = resample_and_save(ds_streamline32, "10min",
                                          output_path = folder_streamline,
                                          savename = "Streamline36_raw")

#%% threshold correction

ds_streamline18_cleaned = threshold_control(ds_streamline18.copy(), 
                                            #threshold_u = 30, threshold_v = 30,
                                            threshold_w = 10,
                                            threshold_spd = 30, threshold_spd_low = 25,
                                            alt_limit = 2900,
                                            threshold_nvrad = 4)

ds_streamline32_cleaned = threshold_control(ds_streamline32.copy(), 
                                            #threshold_u = 30, threshold_v = 30,
                                          threshold_w = 10,
                                          threshold_spd = 30, threshold_spd_low = 25,
                                          alt_limit = 2900,
                                          threshold_nvrad = 4)
                                         
#%%additional despiking
ds_streamline18_cleaned, nan_counter = despiking(ds_streamline18_cleaned, 
                                                 n_start=3, alt_despike = True,
                                                 n_height_start=2, window = "1h")
ds_streamline32_cleaned, nan_counter = despiking(ds_streamline32_cleaned, 
                                                 n_start=3, alt_despike = True,
                                                 n_height_start=2, window = "1h")
                                         

#%%interpolated versions over whole campaign
ds_streamline18_interp = interpolate_height(ds_streamline18_cleaned, 
                                            ds_streamline32_cleaned, 
                                            height_coord = "alt")
ds_streamline32_interp = interpolate_height(ds_streamline32_cleaned, 
                                            ds_streamline18_cleaned, 
                                            height_coord = "alt")

#save
ds_streamline18_cleaned.to_netcdf(folder_streamline/"Streamline18_cleaned.nc")
ds_streamline32_cleaned.to_netcdf(folder_streamline/"Streamline36_cleaned.nc")
ds_streamline18_interp.to_netcdf(folder_streamline/"Streamline18_interp.nc")
ds_streamline32_interp.to_netcdf(folder_streamline/"Streamline36_interp.nc")

#%% resampling and saving
#10min
ds_streamline32_cleaned_10min = resample_and_save(ds_streamline32_cleaned, "10min",
                                                output_path = folder_streamline,
                                                savename = "Streamline36_cleaned")
ds_streamline18_cleaned_10min = resample_and_save(ds_streamline18_cleaned, "10min",
                                                output_path = folder_streamline,
                                                savename = "Streamline18_cleaned")
ds_streamline18_interp_10min = resample_and_save(ds_streamline18_interp, "10min",
                                                output_path = folder_streamline,
                                                savename = "Streamline18_interp")
ds_streamline32_interp_10min = resample_and_save(ds_streamline32_interp, "10min",
                                                output_path = folder_streamline,
                                                savename = "Streamline36_interp")

#30min
ds_streamline18_cleaned_30min = resample_and_save(ds_streamline18_cleaned, "30min",
                                                output_path = folder_streamline,
                                                savename = "Streamline18_cleaned")
ds_streamline32_cleaned_30min = resample_and_save(ds_streamline32_cleaned, "30min",
                                                output_path = folder_streamline,
                                                savename = "Streamline36_cleaned")
ds_streamline18_interp_30min = resample_and_save(ds_streamline18_interp, "30min",
                                                output_path = folder_streamline,
                                                savename = "Streamline18_interp")
ds_streamline32_interp_30min = resample_and_save(ds_streamline32_interp, "30min",
                                                output_path = folder_streamline,
                                                savename = "Streamline36_interp")
#1h
ds_streamline18_cleaned_1h = resample_and_save(ds_streamline18_cleaned, "1h",
                                             output_path = folder_streamline,
                                             savename = "Streamline18_cleaned")
ds_streamline32_cleaned_1h = resample_and_save(ds_streamline32_cleaned, "1h",
                                             output_path = folder_streamline,
                                             savename = "Streamline36_cleaned")
ds_streamline18_interp_1h = resample_and_save(ds_streamline18_interp, "1h",
                                                output_path = folder_streamline,
                                                savename = "Streamline18_interp")
ds_streamline32_interp_1h = resample_and_save(ds_streamline32_interp, "1h",
                                                output_path = folder_streamline,
                                                savename = "Streamline36_interp")

#%% test time height plots streamline 18
os.chdir(r"C:\Users\leopo\PhD\glacier_space\code\Lidar")

from windranger_visu import plot_timeheight, ms_to_knots
from plot_config import cmap_horwind
from metpy.calc import wind_direction, wind_speed
from metpy.units import units

#----settings horwind-----
plot_kwargs = {"fontsize": 20, "figsize":(20,10)}
filled_contour_kwargs = {"levels":[0, 20], "spacing":1, 
                         "cbar_spacing":[0, 20], "label_spacing":5,#[0, 20],
                         "extend":"max", "calc_norm":"yes", "cmap":cmap_horwind#coolwarm
                         }

output_streamline = Path(r"D:\HEFEXIII\plots\Lidar\streamline\test_plots\quality_controlled_v1_SNR_CNS_filter")


for day, ds_day in ds_streamline18_cleaned_10min.groupby("time.date"):
    if len(ds_day.time) < 2:
        print(f"no data for {day}...skip")
        continue
    print(day)
    #ds_day = ds_windranger_cleaned.sel(time = day)
    plot_timeheight(ds_day, height_var = "alt", filled_contour_var = "spd",
                plot_output = output_streamline, barbs = False, barb_unit="m/s",
                plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs,
                ymin = 2720, ymax = 4000)
    
#%%
#----settings vertical wind-----
plot_kwargs = {"fontsize": 20, "figsize":(20,10)}
filled_contour_kwargs = {"levels":[-10, 10], "spacing":1, 
                         "cbar_spacing":[-10, 10], "label_spacing":5,#[0, 20],
                         "extend":"both", "calc_norm":"no", "cmap":"coolwarm"
                         }

output_streamline = Path(r"D:\HEFEXIII\plots\Lidar\streamline\test_plots\quality_controlled_v1_SNR_CNS_filter")


for day, ds_day in ds_streamline32_cleaned_10min.groupby("time.date"):
    if len(ds_day.time) < 2:
        print(f"no data for {day}...skip")
        continue
    print(day)
    #ds_day = ds_windranger_cleaned.sel(time = day)
    plot_timeheight(ds_day, height_var = "alt", filled_contour_var = "w",
                plot_output = output_streamline, barbs = False, barb_unit="m/s",
                plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs,
                ymin = 2720, ymax = 4000)

#%% simple plots for overview
output_streamline = Path(r"D:\HEFEXIII\plots\Lidar\streamline\test_plots")
plot_vars = ["u", "SNR", "spd"]

test_plots(ds_windranger, plot_vars, output_plot = output_windranger)


#%% merge windranger and streamline to one big dataset --> use averaged data

def merge_windranger_streamline(ds_wind, ds_stream, 
                                vars_to_merge = ["u", "v", "w", "spd", "dir"],
                                alt_limit = 2880
):
    
    #remove height
    if "height" in ds_wind.coords:
        ds_wind = ds_wind.drop_vars("height", errors = "ignore")
    if "height" in ds_stream.coords:
        ds_stream = ds_stream.drop_vars("height", errors = "ignore")
    
    ds_wind = ds_wind[vars_to_merge].sel(alt = slice(None, alt_limit))
    ds_stream = ds_stream[vars_to_merge].sel(alt = slice(alt_limit, None))
    
    print(ds_wind.coords)
    print(ds_stream.coords)

    #merge ds
    ds_lidar = xr.concat([ds_wind, ds_stream], dim = "alt")
    
    ds_lidar = ds_lidar.sortby("alt")
    
    return ds_lidar

# 36 m gate ranges
#10min
ds_lidar_10min_36 = merge_windranger_streamline(ds_windranger_cleaned_10min, 
                                             ds_streamline32_interp_10min)
#30min
ds_lidar_30min_36 = merge_windranger_streamline(ds_windranger_cleaned_30min, 
                                             ds_streamline32_interp_30min)
#1h
ds_lidar_1h_36 = merge_windranger_streamline(ds_windranger_cleaned_1h, 
                                             ds_streamline32_interp_1h)

# 18 m gate ranges
#10min
ds_lidar_10min_18 = merge_windranger_streamline(ds_windranger_cleaned_10min, 
                                             ds_streamline18_interp_10min)
#30min
ds_lidar_30min_18 = merge_windranger_streamline(ds_windranger_cleaned_30min, 
                                             ds_streamline18_interp_30min)
#1h
ds_lidar_1h_18 = merge_windranger_streamline(ds_windranger_cleaned_1h, 
                                             ds_streamline18_interp_1h)


#%% save data
folder = Path(r"D:\HEFEXIII\Streamline\streamline_windranger_comb")

#36m
ds_lidar_10min_36.to_netcdf(folder / "lidar_36m_10min.nc")
ds_lidar_30min_36.to_netcdf(folder / "lidar_36m_30min.nc")
ds_lidar_1h_36.to_netcdf(folder / "lidar_36m_1h.nc")
#18m
ds_lidar_10min_18.to_netcdf(folder / "lidar_18m_10min.nc")
ds_lidar_30min_18.to_netcdf(folder / "lidar_18m_30min.nc")
ds_lidar_1h_18.to_netcdf(folder / "lidar_18m_1h.nc")

#%%test time height
#----settings horwind-----
plot_kwargs = {"fontsize": 20, "figsize":(20,10)}
filled_contour_kwargs = {"levels":[0, 20], "spacing":1, 
                         "cbar_spacing":[0, 20], "label_spacing":5,#[0, 20],
                         "extend":"max", "calc_norm":"yes", "cmap":cmap_horwind#coolwarm
                         }

output_lidar = Path(r"D:\HEFEXIII\plots\Lidar\combined\time_height_10min")


for day, ds_day in ds_lidar_10min_18.groupby("time.date"):
    if len(ds_day.time) < 2:
        print(f"no data for {day}...skip")
        continue
    print(day)
    #ds_day = ds_windranger_cleaned.sel(time = day)
    plot_timeheight(ds_day, height_var = "alt", filled_contour_var = "spd",
                plot_output = output_lidar, barbs = False, barb_unit="m/s",
                plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs,
                ymin = 2720, ymax = 4000)
