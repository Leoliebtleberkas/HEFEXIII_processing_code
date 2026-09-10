# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:28:45 2026

@author: leopo
"""

#%% import packages
import numpy as np 
import xarray as xr
import os 
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\turbulence_postprocessing")
from MRD_functions import multiresolution, MRD_sonic
from postprocess import fill_gaps
from functions import detrend

#%% read data
folder = Path(r"D:\HEFEXIII\Tower\L3\no_sectorwise")

#20 Hz data metek
file = r"metek_L3_20Hz_30min.nc"
ds = xr.open_dataset(folder / file)

#20 Hz data smatrflux
file_smart = r"smartflux_L3_20Hz_30min.nc"
ds_smart = xr.open_dataset(folder / file_smart)

#file_30Hz = r"metek_L3_30Hz_fixedsystem.nc"
#ds_30Hz = xr.open_dataset(folder / file_30Hz)

#%% only one period and needed variables
#test for the period 07.08. - 19.08: mostly katabatic and southerly wind
#for foehn: 28.08
#for upvalley: 23.08
start = "2025-08-09 00:00"
end = "2025-08-12 23:30"
ds = ds.sel(time = slice(start,end))
ds_smart = ds_smart.sel(time = slice(start, end))
#ds_30Hz = ds_30Hz.sel(time = slice(start, end))
#%% for testing only: merge 20 Hz and 30 Hz data
#ds = ds.interp(time = ds_30Hz.time)
#ds = xr.merge([ds_30Hz, ds], join = "outer")

#%%specify config
config = {
          "window":"1h",
          "avg_method":"detrend",
          "gap_filling":"interp",
          "var_rename": {"Ts":"tc"}
          }

#rename variables
ds = ds.rename(config["var_rename"])
ds_smart = ds_smart.rename(config["var_rename"])
#ds_30Hz = ds_30Hz.rename(config["var_rename"])

#fill gaps
ds = fill_gaps(ds, config, count_nans=False, add_missing_timesteps=True)
ds_smart = fill_gaps(ds_smart, config, count_nans=False, add_missing_timesteps=True)
#ds_30Hz = fill_gaps(ds_30Hz, config, count_nans=False, add_missing_timesteps=True)
#%%temp to K
ds["tc"] += 273.15
ds_smart["tc"] += 273.15
#%% execute mrd
#m in a way that dt * 2**m is as close as possible to desired window
# for 1 h window: m = 16 for 20Hz, m = 17 for 30Hz
m_20Hz = 16
m_30Hz = 17
#window size:
#in case of 20Hz data: also works with calculating m in the subroutine
#--> giving a window (e.g. pd.Timedelta("30min")) is enough then
window_20Hz = pd.Timedelta(np.median(np.diff(ds.time)) * 2**m_20Hz)
#window_30Hz = pd.Timedelta(np.median(np.diff(ds_30Hz.time)) * 2**m_30Hz)

#call mrd subroutines
mrd_all = MRD_sonic(ds, window_20Hz, m_ref=m_20Hz)
#mrd_all30Hz = MRD_sonic(ds_30Hz, window_30Hz, m_ref = m_30Hz)
mrd_all_smart = MRD_sonic(ds_smart, window_20Hz, m_ref=m_20Hz)
#throw out lowest frequency --> unstable, noisy
mrd_all = mrd_all.isel(tau = slice(1, None))
mrd_all_smart = mrd_all_smart.isel(tau = slice(1, None))

#%%separate daytime and nighttime, e.g. day = 08:00 - 20:00
mrd = mrd_all.where(
        #(mrd_all.time.dt.hour < 8) | (mrd_all.time.dt.hour > 19),
        (mrd_all.time.dt.hour >= 8) & (mrd_all.time.dt.hour <= 19), # 
        drop = True)
mrd_smart = mrd_all_smart.where(
        #(mrd_all_smart.time.dt.hour < 8) | (mrd_all_smart.time.dt.hour > 19),
        (mrd_all_smart.time.dt.hour >= 8) & (mrd_all_smart.time.dt.hour <= 19), # 
        drop = True)


#%%for foehn and upvalley test (dont run box above!):
mrd = mrd_all
mrd_smart = mrd_all_smart

#%% median and quantiles of MRD
mrd_median = mrd.median(dim = "time")
mrd_lower = mrd.quantile(q = 0.25, dim = "time")
mrd_upper = mrd.quantile(q = 0.75, dim = "time")

mrd_smart_median = mrd_smart.median(dim = "time")
mrd_smart_lower = mrd_smart.quantile(q = 0.25, dim = "time")
mrd_smart_upper = mrd_smart.quantile(q = 0.75, dim = "time")

#%%plot
plot_var = "CUW"

if plot_var == "CUW":
    savevar = "uw"
    unit = "10^{-3} m^{2} * s^{-2}"
    ylim = [-10, 15]
elif plot_var == "CWW":
    savevar = "ww"
    unit = "10^{-3} m^{2} * s^{-2}"
    ylim = [-10, 15]
elif plot_var == "CUU":
    savevar = "uu"
    unit = "10^{-3} m^{2} * s^{-2}"
    ylim = [-5, 500]
else:
    savevar = "wT"
    unit = "10^{-3} Km*s^{-1}"
    ylim = [-10, 10]
    
color_list = ["black", "chocolate", "darkorange", "gold"]
#fontsize
fs = 20
fig, ax = plt.subplots(figsize = (8, 6))
#loop over heights of 20 Hz
for h, c in zip(mrd.heights.values, color_list):
    #median
    ax.plot(mrd_median.tau, mrd_median[plot_var].sel(heights = h)*1e3, 
            color = c, label = f"{h} m", linewidth = 5)
    #quantiles
    ax.fill_between(mrd_median.tau, 
                    mrd_lower[plot_var].sel(heights = h)*1e3, 
                    mrd_upper[plot_var].sel(heights = h)*1e3,
                    color = c, alpha = 0.2)
    
#30Hz
#median
ax.plot(mrd_smart_median.tau, mrd_smart_median[plot_var].sel(heights = 1)*1e3, 
        color = "brown", label = "1.0 m", linewidth = 5)
#quantiles
ax.fill_between(mrd_smart_median.tau, 
                mrd_smart_lower[plot_var].sel(heights = 1)*1e3, 
                mrd_smart_upper[plot_var].sel(heights = 1)*1e3,
                color = "brown", alpha = 0.2)

#markers for 1min, 5min, 30min
ax.axvline(60, linestyle="--", color="k")
ax.text(60, 0.95, "1 min", rotation=90, va="top", ha="right",
        transform=ax.get_xaxis_transform(), fontsize = fs - 5)

ax.axvline(300, linestyle="--", color="k")
ax.text(300, 0.95, "5 min", rotation=90, va="top", ha="right",
        transform=ax.get_xaxis_transform(), fontsize = fs - 5)

ax.axvline(1800, linestyle="--", color="k")
ax.text(1800, 0.95, "30 min", rotation=90, va="top", ha="right",
        transform=ax.get_xaxis_transform(), fontsize = fs - 5)

#marker for 0
ax.axhline(0, linestyle = "solid", color = "k")

#legend
handles, labels = ax.get_legend_handles_labels()
order = ["0.5 m", "1.0 m", "3.0 m", "5.0 m", "9.0 m"]
handles_sorted = [handles[labels.index(l)] for l in order if l in labels]
labels_sorted = [l for l in order if l in labels]
ax.legend(handles_sorted, labels_sorted, fontsize = fs-4)

#axes labels
#ax.set_ylabel(r"$C_{wT} [10^{-3} K m * s^{-1}]$", fontsize = fs)
ax.set_ylabel(fr"$C_{{{savevar}}} [{unit}]$", fontsize = fs)
ax.set_xlabel(r"$\tau (s)$", fontsize = fs)

#logarithmic x-axis
ax.set_xscale("log")

#ylim
ax.set_ylim(ylim[0], ylim[1])

#ticklabel size
ax.tick_params(axis='both', which='major', labelsize=fs)

#plt.show()

plt.savefig(fr"D:\HEFEXIII\plots\tower\MRD\MRD_{savevar}_day.pdf", 
            bbox_inches = "tight")
plt.close()
