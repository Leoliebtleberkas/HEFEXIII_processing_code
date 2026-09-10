# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:06:41 2026

@author: leopo
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
 
import os
os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\UAV")

from UAV_plots import *

#%% read data
folder_UAV = Path(r"D:\HEFEXIII\UAV\l4")

ds = xr.open_dataset(folder_UAV / r"UAV_all_locations_clean.nc")

#%% simple overview plots

#reduce to centerline locations
locs = ["Q255", "Q257", "Q267", "Q272", "Q285", "Q293", "Q309", "F309"]
ds_center = ds.sel(location = locs)

#%%
cmap = plt.get_cmap("tab10")
colors = cmap(np.linspace(0, 1, len(ds_center.location)))


color_dict = {}
for i,loc in enumerate(ds_center.location):
    color_dict[str(loc.values)] = colors[i]
    
var_dict = {"spd":"wind speed [m/s]", "dir":"wind dir. [°]", "temp":"T [°C]", "qv": "q [g/kg]"}
xlim_dict = {"spd": None, "dir":360, "temp": None, "qv": None}

file_format = "png"

#call
for time in ds_center.time:
    ds_sel = ds_center.sel(time = time)
    time_str = pd.Timestamp(time.values).strftime("%d-%m-%Y %H:%M")
    save_str = pd.Timestamp(time.values).strftime("%d-%m-%Y_%H%M")
    output_folder = Path(r"D:\HEFEXIII\plots\UAV\profiles\centerline")
    output_path = output_folder / f"UAV_{save_str}.{file_format}"
    
    #call function
    plot_uav_profiles(ds_sel, var_dict, color_dict, output_path = output_path,
                      figsize = (14,5),
                      ncols = 4, nrows = 1, agl = True)
    
