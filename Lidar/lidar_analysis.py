# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:28:52 2026

@author: leopo
"""

#import packages
import xarray as xr
from pathlib import Path
import os
os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\Lidar")

from lidar_cleaning_functions import *
from windranger_visu import plot_timeheight, ms_to_knots
from plot_config import cmap_horwind

#%% load data
folder = Path(r"D:\HEFEXIII\Streamline\streamline_windranger_comb")
ds_lidar = xr.open_dataset(folder / r"lidar_36m_10min.nc")

#%% time height plot
#case study: 09.08.2025
ds_casestudy = ds_lidar.sel(time = slice("2025-08-09", "2025-08-10"))

#----settings horwind-----
output_path = Path(r"D:\HEFEXIII\plots\Lidar\combined\case_study")
plot_kwargs = {"fontsize": 15, "figsize":(14,7)}
filled_contour_kwargs = {"levels":[0, 20], "spacing":1, 
                         "cbar_spacing":[0, 20], "label_spacing":5,#[0, 20],
                         "extend":"max", "calc_norm":"yes", "cmap":cmap_horwind#coolwarm
                         }

plot_timeheight(ds_casestudy, height_var = "alt", filled_contour_var = "spd",
            plot_output = output_path, file_format = "pdf",
            barbs = True, barb_unit="m/s",
            plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs,
            ymin = 2720, ymax = 3720, additional_info=True,
            barb_step_time = 6, barb_step_alt = 2)


#%%
for day, ds_day in ds_lidar.groupby("time.date"):
    print(day)
    print(f"u_max: {ds_day["u"].max()}")
    print(f"v_max: {ds_day["v"].max()}")
    #ds_day = ds_windranger_cleaned.sel(time = day)
    plot_timeheight(ds_day, height_var = "alt", filled_contour_var = "spd",
                plot_output = output_path, barbs = True, barb_unit="m/s",
                plot_kwargs = plot_kwargs, contourf_kwargs = filled_contour_kwargs)
