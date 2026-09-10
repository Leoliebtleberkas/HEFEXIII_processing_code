# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:40:15 2026

@author: leopo
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path 
from matplotlib import pyplot as plt
from metpy.calc import potential_temperature
from metpy.units import units
import os 
#this path has to be adjusted
os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\UAV")
from UAV_cleaning_func import threshold_control, despike
from UAV_plots import *

#data
folder = Path(r"D:\HEFEXIII\UAV\l4")
ds = xr.open_dataset(folder / r"UAV_all_locations_raw.nc")



    
    
#color dict
cmap = plt.get_cmap("gist_rainbow")
colors = cmap(np.linspace(0, 1 ,len(ds.location)))

color_dict = {}
for i,loc in enumerate(ds.location):
    color_dict[str(loc.values)] = colors[i]
    
var_dict = {"spd":"[m/s]", "dir":"[°]", "temp":"[°C]", "qv": "[g/kg]"}


#call
for time in ds.time:
    ds_sel = ds.sel(time = time)
    time_str = pd.Timestamp(time.values).strftime("%d-%m-%Y %H:%M")
    save_str = pd.Timestamp(time.values).strftime("%d-%m-%Y_%H%M")
    output_folder = Path(r"D:\HEFEXIII\plots\UAV\profiles\raw")
    output_path = output_folder / f"{save_str}.png"
    
    #call function
    plot_uav_profiles(ds_sel, var_dict, color_dict, output_path = output_path)
    
#%%cleaning
threshold_dict = {"temp": 20, "qv": 10, "p": 800, "spd": 20, "rh": 100}
lower_threshold_dict = {"temp":3, "qv":2, "p": 600, "rh": 0}

ds_clean = threshold_control(
        ds, threshold_dict, lower_threshold_dict)

#call
for time in ds.time:
    ds_sel = ds_clean.sel(time = time)
    time_str = pd.Timestamp(time.values).strftime("%d-%m-%Y %H:%M")
    save_str = pd.Timestamp(time.values).strftime("%d-%m-%Y_%H%M")
    output_folder = Path(r"D:\HEFEXIII\plots\UAV\profiles\clean")
    output_path = output_folder / f"{save_str}.png"
    
    #call function
    plot_uav_profiles(ds_sel, var_dict, color_dict, output_path = output_path,
                      axis_limits = True)
    
    
#%%calculate theta again
theta = potential_temperature(ds_clean["p"]*units.hPa, 
                              ds_clean["temp"]*units.degC)
ds_clean["theta"] = theta.metpy.dequantify()

#save threshold cleaned data
ds_clean.to_netcdf(folder / r"UAV_all_locations_clean.nc")
    
#%% vertical despiking --> only above sfc inversion, otherwise this will be kicked out
n_dict = {"temp": 2, "qv": 2, "rh" :2}
ds_clean_despike, nan_counter = despike(
    ds_clean, n_dict, alt_window=50, increase_per_loop=0.3, max_iter=20,
)

#call
for time in ds.time:
    ds_sel = ds_clean_despike.sel(time = time)
    time_str = pd.Timestamp(time.values).strftime("%d-%m-%Y %H:%M")
    save_str = pd.Timestamp(time.values).strftime("%d-%m-%Y_%H%M")
    output_folder = Path(r"D:\HEFEXIII\plots\UAV\profiles\clean_despiked")
    output_path = output_folder / f"{save_str}.png"
    
    #call function
    plot_uav_profiles(ds_sel, var_dict, color_dict, output_path = output_path)
