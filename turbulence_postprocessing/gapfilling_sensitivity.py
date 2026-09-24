# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:20:33 2026

@author: leopo
"""

import xarray as xr
from matplotlib import pyplot as plt
from pathlib import Path 

folder = Path(r"D:\HEFEXIII\Tower\detrending_test")

ds_withnan = xr.open_dataset(folder / r"nan_detrend.nc")
ds_nanfilled1 = xr.open_dataset(folder / r"old_detrend.nc")
ds_gauss = xr.open_dataset(folder / r"gauss_detrend.nc")

#
folder_old = Path(r"D:\HEFEXIII\Tower\turbulence_processed\30min_avg")
ds_nanfilled2 = xr.open_dataset(folder_old / r"fluxes_30min.nc")
ds_nanfilled2 = ds_nanfilled2.sel(time = "2025-08-15")

#%% 
variables = ["wT", "uw", "tke"]

print(ds_withnan[variables].mean(dim= "time"))

print(ds_nanfilled2[variables].mean(dim= "time"))

print(ds_gauss[variables].mean(dim= "time"))

#%% plot
var = "tke"
fig, axs = plt.subplots(nrows=len(ds_withnan.heights), figsize = (10,6))

for ax, h in zip(axs, ds_withnan.heights):
    
    #original with mean filling
    ax.plot(ds_nanfilled2.time, ds_nanfilled2[var].sel(heights = h), color = "r",
            label = "no filling")
    
    
    #new version with ignoring nans
    ax.plot(ds_withnan.time, ds_withnan[var].sel(heights = h), color = "k",
            label = "mean filling")
    
    #new version with gaussian noise filling
    ax.plot(ds_gauss.time, ds_gauss[var].sel(heights = h), color = "b",
            label = "gauss filling")
    
    ax.set_title(f"{var} - {h.values} m")
    
ax.legend()

plt.tight_layout(pad = 0.3)