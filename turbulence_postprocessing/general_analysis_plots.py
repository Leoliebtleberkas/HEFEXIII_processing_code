# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:20:36 2026

@author: leopo
"""

#%% packages
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os

#%%read data
folderL1 = Path(r"D:\HEFEXIII\Tower\CR3000_l1")
folderL2 = Path(r"D:\HEFEXIII\Tower\L2")
folderL3 = Path(r"D:\HEFEXIII\Tower\L3\no_sectorwise")
folder_turbulence30min = Path(r"D:\HEFEXIII\Tower\turbulence_processed\30min_avg")

#tower L2 data
ds_tower_1h = xr.open_dataset(folderL2 / r"tower_1h.nc")
ds_tower_10min = xr.open_dataset(folderL2 / r"tower_10min.nc")

#flux data
ds_flux20 = xr.open_dataset(folder_turbulence30min / r"fluxes_30min.nc")
ds_fluxsmart = xr.open_dataset(folder_turbulence30min / r"fluxes_30min_smart.nc")

#correct for actually missing data
ds_flux20 = ds_flux20.where(ds_flux20["meanU"] != 0)
ds_fluxsmart = ds_fluxsmart.where(ds_fluxsmart["meanU"] != 0)



#%% directional constancy
def calc_dc(
        u, v, wspeed = None
):
    #vector averaged wind speed
    vector_average = np.sqrt(u.mean(dim = "time")**2 + v.mean(dim = "time")**2)
    #absolute wind speed average
    if wspeed is not None:
        absolute_average = wspeed.mean(dim = "time")
    else:
        absolute_average = np.sqrt(u**2 + v**2).mean(dim = "time")
    
    #dc rounded to two decimals
    dc = np.round(vector_average / absolute_average, decimals=2)
    
    return dc

dc = calc_dc(ds_tower_10min["u"], ds_tower_10min["v"])
print(dc)

#%%check avg. difference between corrected and uncorrected sensible heat:
relative_change = (ds_fluxsmart["H_corr"] - ds_fluxsmart["H"]) / (ds_fluxsmart["H"])
print(relative_change.mean())

fig, ax = plt.subplots()

ax.plot(ds_fluxsmart.time, ds_fluxsmart["H"].sel(heights = 1))

ax.plot(ds_fluxsmart.time, ds_fluxsmart["H_corr"].sel(heights = 1), color = "r")

#%%merge fluxes and get mean/median of some variables
ds_flux = xr.merge([ds_flux20, ds_fluxsmart], join = "outer")


# mean and median of important turbulence variables
var_to_avg = ["H", "uw", "tke", "LE_corr"]

for var in var_to_avg:
    valid_times = ds_flux[var].notnull().all(dim = "heights")
    
    if var == "LE_corr":
        #average
        avg = ds_flux[var].mean(dim = "time")
        #median
        median = ds_flux[var].mean(dim = "time")
        
        
    else:
        #average
        avg = ds_flux[var].where(valid_times).mean(dim = "time")
        
        #median
        median = ds_flux[var].where(valid_times).median(dim = "time")
    
    print(avg)
    #print(median)
    

    
#%% test RH correction
fig, axs = plt.subplots(nrows = 2, figsize = (10, 4))
ax = axs[0]
ax.plot(ds_tower_10min.time, ds_tower_10min["RH"].sel(heights = 1))
ax.plot(ds_tower_10min.time, ds_tower_10min["RH"].sel(heights = 9), color = "r")
ax1 = axs[1]
ax1.plot(ds_tower_10min.time, ds_tower_10min["q"].sel(heights = 1))


#%% turbulence data to 1h for overview plot
ds_flux_1h = ds_flux.resample(time = "1h", label = "right", closed = "right").mean()

#%% plot timeseries overview
#color dictionary
cmap = plt.get_cmap("Spectral_r")
#cmap = plt.get_cmap("turbo")
#colors = cmap(np.linspace(0.05, 0.95, 8))
colors = cmap(np.concatenate([np.linspace(0.0, 0.3, 4), np.linspace(0.7, 1.0, 4)]))
c_dict = {"0.5":colors[0], "1.0":colors[1], "2.0":colors[2], "3.0":colors[3],
          "4.0":colors[4], "5.0":colors[5], "7.0":colors[6], "9.0":colors[7]}
xticks = pd.date_range(start = "2025-08-07", end = "2025-09-03", freq = "4d")

fig, axs = plt.subplots(nrows = 8, ncols = 1, figsize = (10,10), sharex = True)

#temperature
ax0 = axs[0]
#loop over heights
for h in ds_tower_1h.heights:
    color = c_dict[str(h.values)]
    ax0.plot(ds_tower_1h.time, ds_tower_1h.Ts.sel(heights = h), linewidth = 1.5, 
             label = f"{h.values} m", color = color)
ax0.set_xticks(xticks)
ax0.set_xticklabels([])
ax0.set_ylabel("Ts [°C]", fontsize = 12)
ax0.grid(linestyle = "dashed", which = "both")
#ax0.set_title("air temperature", fontsize = 15)

#relative humidity
ax1 = axs[1]

for h in ds_tower_1h.heights:
    color = c_dict[str(h.values)]
    ax1.plot(ds_tower_1h.time,ds_tower_1h.RH.sel(heights = h), linewidth = 1.5, 
             label = f"{h.values} m", color = color)
ax1.set_xticks(xticks)
ax1.set_xticklabels([])
ax1.set_ylabel("RH [%]", fontsize = 12)
ax1.grid(linestyle = "dashed", which = "both")
#ax1.set_title("relative humidity", fontsize = 15)

#wind speed
ax2 = axs[2]
for h in ds_tower_1h.heights:
    color = c_dict[str(h.values)]
    ax2.plot(ds_tower_1h.time, ds_tower_1h.Wspd.sel(heights = h), linewidth = 1.5, 
             label = f"{h.values} m", color = color)
ax2.set_ylabel("$\overline{U}$ [$m s^{-1}$]", fontsize = 12)
#axes limits
ax2.set_xticks(xticks)
ax2.set_ylim([0, 8])
#ax2.set_yticks([0, 2.5, 5, 7.5, 10])
ax2.grid(linestyle = "dashed", which = "both")
#ax2.set_title("wind speed", fontsize = 15)

#wind direction
ax3 = axs[3]
for h in ds_tower_1h.heights:
    color = c_dict[str(h.values)]
    ax3.scatter(ds_tower_1h.time, ds_tower_1h.Wdir.sel(heights = h), 
                label = f"{h.values} m", color = color, s = 2)
#axes labels
ax3.set_xticks(xticks)
ax3.set_ylabel("deg. [°]", fontsize = 12)
ax3.set_ylim([0, 360])
ax3.set_yticks([0, 90, 180, 270, 360])
#ax3.set_title("wind direction", fontsize = 15)

#dynamic sensible heat
ax4 = axs[4]
for h in ds_flux_1h.heights:
    color = c_dict[str(h.values)]
    #ax4.plot(ds_flux_1h.time, ds_flux_1h.wT.sel(heights = h), linewidth = 1.5,
    if h == 1:
        #var = "H_corr"
        var = "H"
    else:
        var = "H"
    ax4.plot(ds_flux_1h.time, ds_flux_1h[var].sel(heights = h), 
             linewidth = 1.5,
             label = f"{h.values} m", color = color)
ax4.set_ylim([-130, 20])
#ax4.set_ylabel("$K m^{-1}$", fontsize = 14)
ax4.set_ylabel("H [$W m^{-2}$]", fontsize = 12)
#ax4.set_title("sensible heat flux H", fontsize = 15)
ax4.set_xticks(xticks)

#latent heat flux
ax5 = axs[5]
for h in ds_flux_1h.heights:
    color = c_dict[str(h.values)]
    ax5.plot(ds_flux_1h.time, ds_flux_1h["LE"].sel(heights = h), #LE_corr
             linewidth = 1.5,
             label = f"{h.values} m", color = color)
ax5.set_ylabel("LE [$W m^{-2}$]", fontsize = 12)
#ax5.set_title("latent heat flux LE", fontsize = 15)
ax5.set_xticks(xticks)

#horizontal momentum flux
ax6 = axs[6]
for h in ds_flux_1h.heights:
    color = c_dict[str(h.values)]
    ax6.plot(ds_flux_1h.time, ds_flux_1h.uw.sel(heights = h),
             linewidth = 1.5,
             label = f"{h.values} m", color = color)
ax6.set_ylabel("$\overline{u'w'}$ [$m^{2} s^{-2}$]", fontsize = 12)
#ax6.set_title("horizontal momentum flux", fontsize = 15)
ax6.set_xticks(xticks)

#TKE
ax7 = axs[7]
for h in ds_flux_1h.heights:
    color = c_dict[str(h.values)]
    ax7.plot(ds_flux_1h.time, ds_flux_1h.tke.sel(heights = h), linewidth = 1.5, 
             label = f"{h.values} m", color = color)
ax7.set_ylabel("TKE [$m^2 s^{-2}$]", fontsize = 12)
#ax7.set_title("turbulent kinetic energy", fontsize = 15)
ax7.set_xticks(xticks)
#xticklabel of
for ax in fig.axes:
    ax.set_xticklabels("")
    
#shading for IOP
for ax in fig.axes:
    
    #yticks
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    
    #1st IOP
    ax.axvspan(pd.to_datetime("2025-08-07 03:00"), pd.to_datetime("2025-08-10 07:00"),
               ymin = 0, ymax = 1, facecolor = "grey", alpha = 0.3)
    ax.axvspan(pd.to_datetime("2025-08-18 15:00"), pd.to_datetime("2025-08-19 18:00"),
               ymin = 0, ymax = 1, facecolor = "grey", alpha = 0.3)
    
    #mark synoptic upvalley
    ax.axvspan(pd.to_datetime("2025-08-21 20:00"), pd.to_datetime("2025-08-26 23:00"),
               ymin = 0, ymax = 1, facecolor = "blue", alpha = 0.2)
    
    #mark foehn
    ax.axvspan(pd.to_datetime("2025-08-27 03:00"), pd.to_datetime("2025-08-29 22:00"),
               ymin = 0, ymax = 1, facecolor = "red", alpha = 0.2)


#xticks at lowest axes
fig.axes[-1].set_xticklabels(xticks, fontsize = 16)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
#grid on 
for ax in fig.axes:
    ax.grid(linestyle = "dashed", which = "both")
    
#legend
handles, labels = ax0.get_legend_handles_labels()

fig.legend(
    handles,labels,loc="upper right", bbox_to_anchor=(1.0, 1.06),
    ncol=4,fontsize=12,frameon=True
)
#layout/spacing between plots
#fig.tight_layout(rect=[0, 0, 0.95, 0.95])
fig.tight_layout(pad = 0.8)

#plt.show()
plt.savefig(r"D:\HEFEXIII\plots\tower\overview_general\tower_overview.pdf",
            bbox_inches = "tight")
plt.close()


#%% ---------- Station Hintereis and im Hinteren Eis ----------

folder = Path("D:\HEFEXIII\AWS_uibk")
#station hintereis 
df_sthe = pd.read_csv(folder / r"UIBK_Station_Hintereis.csv", 
                 sep = ";", header = 1, index_col = 0)
df_sthe_precip = pd.read_csv(folder / r"precipStHE.csv",
                        sep = ";", header = 1, index_col = 0)
#im hinteren eis
df_ihe = pd.read_csv(folder / r"UIBK_Station_ImHinterenEis.csv", 
                 sep = ";", header = 1, index_col = 0)
df_ihe_wind = pd.read_csv(folder / r"ventus2025_10min.csv",
                        sep = ",", header = 0, index_col = 0)



df_sthe.index = pd.to_datetime(df_sthe.index)
df_sthe_precip.index = pd.to_datetime(df_sthe_precip.index)
df_ihe.index = pd.to_datetime(df_ihe.index)
df_ihe_wind.index = pd.to_datetime(df_ihe_wind.index)


#same times
df_sthe = df_sthe.loc[(df_sthe.index >= "2025-08-06") & (df_sthe.index < "2025-09-01")]
df_sthe_precip = df_sthe_precip.loc[
    (df_sthe_precip.index >= "2025-08-06") & (df_sthe_precip.index < ("2025-09-01"))]
df_ihe = df_ihe.loc[(df_ihe.index >= "2025-08-06") & (df_ihe.index < "2025-09-01")]
df_ihe_wind = df_ihe_wind.loc[(df_ihe_wind.index >= "2025-08-06") & (df_ihe_wind.index < "2025-09-01")]

#fill missing entries
df_ihe = df_ihe.resample("10min").asfreq()
df_ihe_wind = df_ihe_wind.resample("10min").asfreq()

#%% calculate u and v components and resample to hour
from metpy.calc import wind_components, wind_direction
from metpy.units import units

u_sthe, v_sthe = wind_components(df_sthe["wspeed"].values*units("m/s"),
                                 df_sthe["wdir"].values*units.deg)

u_ihe, v_ihe = wind_components(df_ihe_wind["windspeed_act_3"].values*units("m/s"),
                               df_ihe_wind["winddir_act_3"].values*units.deg)

df_sthe["u"] = u_sthe.magnitude
df_sthe["v"] = v_sthe.magnitude
df_ihe_wind["u"] = u_ihe.magnitude
df_ihe_wind["v"] = v_ihe.magnitude

df_sthe_h = df_sthe.resample("1h").mean()
df_sthe_precip_h = df_sthe_precip.resample("1h").sum()

df_ihe_h = df_ihe.resample("1h").mean()
df_ihe_wind_h = df_ihe_wind.resample("1h").mean()

#calculate wind direction again
df_sthe_h["wdir"] = wind_direction(df_sthe_h["u"].values*units("m/s"), 
                                   df_sthe_h["v"].values*units("m/s")).magnitude
df_ihe_wind_h["wdir"] = wind_direction(df_ihe_wind_h["u"].values*units("m/s"), 
                                  df_ihe_wind_h["v"].values*units("m/s")).magnitude

#%%accumulated precip
df_sthe_precip["precip_acc12"] = df_sthe_precip["accumulated_total_nrt"].rolling("12h").sum()
#%%plot station hintereis
xticks = pd.date_range(start = "2025-08-07", end = "2025-09-03", freq = "4d")
fig, axs = plt.subplots(nrows = 5, ncols = 1, figsize = (10, 8))

#radiation
ax0 = axs[0]
ax0.plot(df_sthe_h.index, df_sthe_h["swin_avg"], linewidth = 2, color = "blue")
ax0.set_ylabel("$W m^{-2}$", fontsize = 14)
#ax0.set_title("incoming shortwave radiation", fontsize = 15)


#temperature
ax1 = axs[1]
#StHE
ax1.plot(df_sthe_h.index, df_sthe_h["tair_avg"], linewidth = 2, color = "blue",
         label = "StHE, 2 m")
#iHE
ax1.plot(df_ihe_h.index, df_ihe_h["taact_2m_avg"], linewidth = 2, color = "orange",
         label = "iHE, 5.5 m")
ax1.set_ylabel("[°C]", fontsize = 14)
#ax1.legend(loc = "upper right")
ax1.set_yticks([0, 5, 10, 15])
#ax1.set_title("air temperature", fontsize = 15)

#wind speed
ax2 = axs[2]
#ax3 = ax2.twinx()
#StE
ax2.plot(df_sthe_h.index, df_sthe_h["wspeed"], color = "blue", #label = "wind speed",
         label = "StHE, 3.3 m")
#iHE
ax2.plot(df_ihe_wind_h.index, df_ihe_wind_h["windspeed_act_3"], 
         color = "orange", label = "iHE, 6 m") #label = "wind speed",
#ax2.plot(df.index, df["wspeed_max"], color = "r", label = "max. gust")
ax2.set_ylabel("$[m s^{-1}]$", fontsize = 14)
ax2.set_yticks([0, 5, 10, 15, 20])
ax2.set_ylim(0, 20)
#ax2.legend(loc = "upper right")

#wind dir
ax3 = axs[3]
#StHE
ax3.scatter(df_sthe_h.index, df_sthe_h["wdir"], color = "blue",# label = "wind dir.", 
            s = 3)
#iHE
ax3.scatter(df_ihe_wind_h.index, df_ihe_wind_h["winddir_act_3"], 
         color = "orange", s = 3, label = "wind speed")#, label = "iHE, 6 m")
ax3.set_ylabel("[°]", fontsize = 14)
#ax3.set_xticks(xticks)
ax3.set_xticklabels([])
ax3.set_yticks([0, 90, 180, 270, 360])
ax3.set_ylim(0, 360)
#ax2.legend(loc = "upper right")
#ax3.set_title("wind speed and direction", fontsize = 15)

#precipitation
ax4 = axs[4]
ax4.plot(df_sthe_precip.index, df_sthe_precip["accumulated_total_nrt"], 
         linewidth = 2, color = "blue")
ax4.set_ylabel("$[mm / 12h]$", fontsize = 14)
#ax4.set_title("12 hour accumulated precipitation", fontsize = 15)

for ax in fig.axes:
    ax.grid(linestyle = "dashed", which = "both")
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    
    #yticks
    ax.tick_params(axis="y", labelsize=12)
    
    #add IOP's
    ax.axvspan(pd.to_datetime("2025-08-07 03:00"), pd.to_datetime("2025-08-10 08:00"),
               ymin = 0, ymax = 1, facecolor = "grey", alpha = 0.3)
    ax.axvspan(pd.to_datetime("2025-08-18 15:00"), pd.to_datetime("2025-08-19 18:00"),
               ymin = 0, ymax = 1, facecolor = "grey", alpha = 0.3)
    
    #mark synoptic upvalley
    ax.axvspan(pd.to_datetime("2025-08-21 20:00"), pd.to_datetime("2025-08-26 23:00"),
               ymin = 0, ymax = 1, facecolor = "blue", alpha = 0.2)
    
    #mark foehn
    ax.axvspan(pd.to_datetime("2025-08-27 03:00"), pd.to_datetime("2025-08-29 22:00"),
               ymin = 0, ymax = 1, facecolor = "red", alpha = 0.2)
    #ax.axvspan(pd.to_datetime("2025-08-18 15:00"), pd.to_datetime("2025-08-19 18:00"),
    #           ymin = 0, ymax = 1, facecolor = "grey", alpha = 0.2)
    
#xticks at lowest axes
ax4.set_xticks(xticks)
ax4.set_xticklabels(xticks, fontsize = 16)
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

#add legend
from matplotlib.lines import Line2D

legend_handles = [
    Line2D([0], [0], color="blue", linewidth=2, label="StHE"),
    Line2D([0], [0], color="orange", linewidth=2, label="iHE")
]

fig.legend(
    handles=legend_handles,bbox_to_anchor=(1.0, 0.98),loc="lower right",
    fontsize=12, ncols = 2)

#layout
fig.tight_layout(pad = 0.8)

#save
plt.savefig(r"D:\HEFEXIII\plots\tower\overview_general\StHE_overview.pdf",
            bbox_inches = "tight")
plt.close()

#%%cold spell sums
c1_start = "2025-08-19 21:40"
c1_end = "2025-08-22 08:00"
c2_start = "2025-08-27"
c2_end = "2025-08-30 17:00"

sum1 = df_precip["accumulated_nrt"][c1_start:c1_end].sum()
sum2 = df_precip["accumulated_nrt"][c2_start:c2_end].sum()

print(sum1)
print(sum2)

#%% moisture fürn Wild
folderL2 = Path(r"D:\HEFEXIII\Tower\L2")
folderL3 = Path(r"D:\HEFEXIII\Tower\L3\no_sectorwise")
folder_turbulence30min = Path(r"D:\HEFEXIII\Tower\turbulence_processed\30min_avg")

#tower L2 data
ds_tower_1min = xr.open_dataset(folderL2 / r"tower_1min.nc")
ds_sub = ds_tower_1min.sel(time = slice("2025-08-18", "2025-08-21"), 
                           heights = [2,5,9])

fig, ax = plt.subplots()
for h in [2,5,9]:
    ax.plot(ds_sub.time, ds_sub.sel(heights = h)["RH"], label = f"{h} m")
ax.legend()
ax.set_ylabel("RH [%]")
ax.set_xlabel("time (UTC)")
ax.grid(axis = "y")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
ax.tick_params(axis = "x", rotation = 45)

ds_RH = ds_sub[["AirTC", "RH"]]

ds_RH.to_netcdf(folderL2 / r"radarIOP_RH_Temp.nc")
