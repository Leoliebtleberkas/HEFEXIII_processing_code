# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:05:18 2026

@author: leopo
"""

import xarray as xr
from pathlib import Path
from matplotlib import pyplot as plt

#read data
folder = Path(r"D:\HEFEXIII\Tower\CR3000_l0")

ds = xr.open_dataset(folder / r"metek_L0_20Hz_SENSOR_COMPARISON.nc")

#only between 14:35 and 15:55
ds = ds.sel(time = slice("2026-06-17 14:35", "2026-06-17 15:50"))

#%%
'''
ATTENTION, HEIGHTS WERE CHANGED IN INTERCOMPARISON
intercomparison 1m = HEFEXIII 9m
intercomparison 3m = HEFEXIII 5m
intercomparison 5m = HEFEXIII 3m
intercomparison 7m = not used in HEFEXIII
'''

#--> change the heights for bias correction
mapping = {1: 9, 3: 5.5, 5: 3}

for old, new in mapping.items():
    ds["heights"] = ds["heights"].where(
        ds["heights"] != old,
        new
    )
    
ds["heights"] = ds["heights"].where(ds["heights"] != 5.5, 5)
ds = ds.sortby("heights")

#kick out the 7 m
ds = ds.sel(heights = [3,5,9])
#%% plot 
plot_vars =["u", "v", "w", "Ts"]
colors = ["k", "b", "r", "orange"]

fig, axs = plt.subplots(nrows = len(plot_vars), figsize = (10, 8))

for i, var in enumerate(plot_vars):
    
    ax = axs[i]
    for j,h in enumerate([3,5,9]):
        ds_h = ds.sel(heights = h)
        ax.plot(ds_h.time, ds_h[var], color = colors[j], label = f"{str(h)} m")
        
        ax.set_ylabel(var)
        
axs[i].legend()
plt.show()


# --> 3 m sensor had warm bias ~1°C compared to the other two meteks
# --> also there is a small bias in u and v wind components

#%% more exploration
for var in plot_vars:
    time_mean = ds[var].mean(dim = "time")
    print(f"{var}: {time_mean}")
    
#%% get the mean temperature bias of 3 m sensor

mean_3m = ds["Ts"].sel(heights = 3).mean()
mean_else = ds["Ts"].sel(heights = [5,9]).mean()

bias = mean_3m - mean_else

# --> warm bias of 0.92°C


#%% bias correction function
def correct_bias_single_height(
        ds, var, h, bias
): 
    
    ds[var] = xr.where(ds["heights"] == h, 
                       ds[var] - bias, ds[var])
    
    return ds
#%% correct the 3 m level with that in the L1 data
filepath = Path(r"D:\HEFEXIII\Tower\CR3000_l1")
metek_20Hz = r"metek_L1_20Hz_orig.nc"

ds_L1 = xr.open_dataset(filepath / metek_20Hz)
ds_L1_corr = ds_L1.copy()

#%% BIAS CORRECTIONS
''' 
add further bias corrections here if needed
'''
ds_L1_corr = correct_bias_single_height(ds_L1_corr, "Ts", 3, bias)


#%% testplot
ds_L1_avg = ds_L1.resample(time = "1h").mean()
ds_L1_corr_avg = ds_L1_corr.resample(time = "1h").mean()
fig, ax = plt.subplots()
ax.plot(ds_L1_avg.time, ds_L1_avg.sel(heights = 3)["Ts"], color = "k")
ax.plot(ds_L1_corr_avg.time, ds_L1_corr_avg.sel(heights = 3)["Ts"], color = "r")


#%% save
ds_L1_corr.to_netcdf(filepath / r"metek_L1_20Hz_orig_biascorrected.nc")