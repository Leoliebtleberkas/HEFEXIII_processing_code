# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:07:57 2025

@author: leopo
"""

#import packages
import numpy as np 
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib.dates as mdates 
import os
from pathlib import Path
import glob

from plot_base_functions import *
from plot_config import *

  
    

#%% load data
'''
def load_data(
        date
):
    folder = fr"D:\HEFEXIII\WindRanger\WindRanger\202508{date}"
    file_pattern = folder + fr"\202508{date}_*_averaged.nc"
    print(f"load data from {file_pattern}")
    
    # Alle Dateien finden und sortieren
    files = sorted(glob.glob(file_pattern))
    
    # Jede Datei einzeln laden (Metadatenkonflikte werden ignoriert)
    datasets = [xr.open_dataset(f, engine="netcdf4") for f in files]
    
    # Entlang der Zeitdimension zusammenfügen
    ds = xr.concat(datasets, dim="time")

    return ds
#%%resample to 10min
def resampling(
        ds, resampling_time = "10min"
):
    ds = ds.sortby("time")
    ds = ds.resample(time=resampling_time).mean()
    
    return ds

'''

#%% calculate U,V, W in knots
def ms_to_knots(
        velocity
):
    vel_knots = velocity * 1.94384
    return vel_knots




#%% plotting

def plot_timeheight(
        data, height_var, filled_contour_var, plot_output, file_format = "png",
        barbs = False, barb_unit = "m/s",
        contour_var1 = None, contour_var2 = None,
        location = None, name = None, model = None, 
        plot_kwargs = None, contourf_kwargs = None, max_height = 200,
        ymin = None, ymax= None, barb_step_time = 1, barb_step_alt = 1,
        additional_info = False
):
    
    fig, ax = plt.subplots(figsize=plot_kwargs["figsize"])
    
    #get plot settings
    #filled contour plot
    contourf_levels = get_levels(data[filled_contour_var], 
                                 levels = contourf_kwargs["levels"], 
                                 spacing = contourf_kwargs["spacing"],
                                 cbar_spacing = contourf_kwargs["label_spacing"]
                                 )
    #colorbar
    cbar_ticks = get_cbarticks(contourf_levels, contourf_kwargs["label_spacing"])
    extend = contourf_kwargs["extend"]
    if contourf_kwargs["calc_norm"] == "yes":
        norm = get_norm(extend, contourf_levels)
    else:
        norm = None
        
    #cmap filled contour plot
    cmap = contourf_kwargs["cmap"]
    #if cmap in cmap_dict.keys():
    #    cmap = cmap_dict[cmap]
    #else:
    #    cmap = cmap
    
    #contour1
    if contour_var1:
        contour1_kwargs = plot_kwargs.contour1_kwargs
        contour1_levels = get_levels(data[contour_var1], 
                                     levels = contour1_kwargs["levels"],
                                     spacing = contour1_kwargs["spacing"],
                                     cbar_spacing = contour1_kwargs["label_spacing"]
                                     )
        #spacing and labels
        spacing1 = contour1_kwargs["spacing"]    
        contour1_labels = get_labellevels(contour1_levels, contour1_kwargs["label_spacing"])
        #contour1 color
        contourcolor = contour1_kwargs["color"]
        labelfloat = contour1_kwargs["labelfloat"]
    
    #contour2
    if contour_var2:
        contour2_kwargs = plot_kwargs.contour2_kwargs
        contour2_levels = get_levels(data[contour_var2], 
                                     levels = contour2_kwargs["levels"],
                                     spacing = contour2_kwargs["spacing"],
                                     cbar_spacing = contour2_kwargs["label_spacing"]
                                     )
        #spacing and labels
        spacing2 = contour2_kwargs["spacing"]
        contour2_labels = get_labellevels(contour2_levels, contour2_kwargs["label_spacing"])
        #contour2 color
        contour2color = contour2_kwargs["color"]
        labelfloat2 = contour2_kwargs["labelfloat"]
        
        
    #fontsize
    fontsize = plot_kwargs["fontsize"]

    y, x = np.meshgrid(data[height_var], data.time)
    #x, y = np.meshgrid(data[height_var], data.time, indexing="ij")
    
    #topography in grey
    #ax.fill_between(x[:,0], -100, y2 =x.min(), color = "grey")
    
    #filled contour plot
    contourf = filled_contour_plot(ax, x, y, data[filled_contour_var],
                        cmap = contourf_kwargs["cmap"], levels = contourf_levels, norm = norm,
                        extend=extend, zorder = 0)
    
    #add colorbar
    #cbar = add_colorbar(contourf, ax, cbar_levels = contourf_levels, 
    #                 label = f"{data[filled_contour_var].attrs['units']}", extend = extend, 
    #                 location = "right", orientation = "vertical", pad = 0.02,
    #                 aspect = 35, cbar_label_y=None)
    cbar = fig.colorbar(contourf, ax=ax, label = "m/s")
    cbar.set_label("m/s", fontsize=fontsize+2)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(which='major', length=1, labelsize=fontsize+2)
    #cbar.ax.tick_params(labelsize=14)
        
    #below upper_lim, vertically only every second wind barb for better visibility
    upper_lim = 200
    
    
    #add wind barbs
    if barbs and barb_unit == "knots":
        
        #barbs for winds >= 5 kn
        mask1 = (data["VEL_knots"] >= 5) 
        U_knots_mod = data["U_knots"].where(mask1)
        V_knots_mod = data["V_knots"].where(mask1)
        
        key_length = 7
    
        ax.barbs(x[::barb_step_time,::barb_step_alt], y[::barb_step_time,::barb_step_alt], 
                 U_knots_mod[::barb_step_time,::barb_step_alt], 
                 V_knots_mod[::barb_step_time,::barb_step_alt], 
                 color="k", length=key_length, sizes=dict(emptybarb=0.05), 
                 linewidth = 0.8,
                 fill_empty=True, zorder=4)
        
        #barbs for winds 1 - 5 kn
        mask2 = (data["VEL_knots"] >= 1) & (data["VEL_knots"] < 5)
    
        # U_knots und V_knots nur an diesen Stellen auswählen
        U_small = data["U_knots"].where(mask2)
        V_small = data["V_knots"].where(mask2)
    # =============================================================================
        ax.barbs(x[::barb_step_time,::barb_step_alt], y[::barb_step_time,::barb_step_alt], 
                 U_small[::barb_step_time,::barb_step_alt], 
                 V_small[::barb_step_time,::barb_step_alt], 
                  color = "k",
                  length = key_length, 
                  barb_increments = dict(half=1, full=5, flag=100),
                  sizes = dict(flag=0, full=0, half=0, emptybarb=0.04, height = 0),
                  fill_empty = True, zorder = 4)
    # =============================================================================
        
        
        #empty barbs for dots at every point
        data["stakeholder_wind"] = xr.DataArray(np.zeros(data["U_knots"].shape), 
                                                dims = data["U_knots"].dims, 
                                                coords = data["U_knots"].coords)
        #u_zero = space_variable(data, "stakeholder_wind", upper_lim)
    
        ax.barbs(x[::barb_step_time,::barb_step_alt], y[::barb_step_time,::barb_step_alt], 
                 data["stakeholder_wind"][::barb_step_time,::barb_step_alt], 
                 data["stakeholder_wind"][::barb_step_time,::barb_step_alt], 
                 color = "k",
                 length = key_length, sizes = dict(emptybarb = 0.04),
                 fill_empty = True, zorder = 4)
        
        #add barbkey
        barbkey = add_barbkey(ax, fontsize = fontsize, barb_unit=barb_unit,
                              key_length = key_length)
        
    #add wind barbs
    if barbs and barb_unit == "m/s":
        key_length = 7
    
        ax.barbs(x[::barb_step_time,::barb_step_alt], y[::barb_step_time,::barb_step_alt], 
                 data["u"][::barb_step_time,::barb_step_alt], 
                 data["v"][::barb_step_time,::barb_step_alt], 
                 color="k", length=key_length, sizes=dict(emptybarb=0.05), 
                 barb_increments = dict(half=1, full=2, flag=5),
                 linewidth = 0.8,
                 fill_empty=True, zorder=4)
        
        '''
        #barbs for winds 0-1 m/s
        mask2 = (data["spd"] > 0) & (data["spd"] < 1)
    
        # U_knots und V_knots nur an diesen Stellen auswählen
        U_small = data["u"].where(mask2)
        V_small = data["v"].where(mask2)
    
        ax.barbs(x, y, U_small, V_small, 
                  color = "k",
                  length = key_length, 
                  #barb_increments = dict(half=1, full=2, flag=100),
                  sizes = dict(flag=0, full=0, half=0, emptybarb=0.04, height = 0),
                  fill_empty = True, zorder = 4)
    
        '''
        
        #empty barbs for dots at every point
        data["stakeholder_wind"] = xr.DataArray(np.zeros(data["u"].shape), 
                                                dims = data["u"].dims, 
                                                coords = data["u"].coords)
        #u_zero = space_variable(data, "stakeholder_wind", upper_lim)
    
        ax.barbs(x[::barb_step_time,::barb_step_alt], y[::barb_step_time,::barb_step_alt], 
                 data["stakeholder_wind"][::barb_step_time, ::barb_step_alt], 
                 data["stakeholder_wind"][::barb_step_time, ::barb_step_alt], 
                 color = "k",
                 length = key_length, sizes = dict(emptybarb = 0.04),
                 fill_empty = True, zorder = 4)
        
        #add barbkey
        barbkey = add_barbkey(ax, fontsize = fontsize, barb_unit=barb_unit,
                              key_length = key_length)
        
    #axes limits
    #ax.set_ylim(0, max_height)
    
    
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    #axes labels
    ax.set_xlabel("Time (UTC)",fontsize = fontsize+2)
    ax.set_ylabel("Altitude (m AMSL)",fontsize = fontsize+2)
    
    #axes ticks
    #x hourly ticks
    #ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m. %H'))
    
    #y
    #yticks = np.arange(0, max_height+20, 25)
    #ax.set_yticks(yticks)
    #ax.set_yticklabels([f'{tick}' for tick in yticks])
    
    #fontsize
    ax.tick_params(axis = "both", labelsize = fontsize+1)
    #ax.tick_params(axis="x", labelrotation=45)
    
    
    # ---additional info: streamline - windranger + crestheight
    if additional_info:
        ax.axhline(2920, linestyle = "dashed", color = "darkred",
                   linewidth = 3)
        ax.text(-0.01, 2940, r"$\uparrow$ SL", ha = "right", va = "center",
                color = "darkred", fontsize = fontsize-2,
                transform=ax.get_yaxis_transform())
        ax.text(-0.01, 2900, r"$\downarrow$ WR", ha = "right", va = "center",
                color = "darkred", fontsize = fontsize-2,
                transform=ax.get_yaxis_transform())
        #crest height
        ax.axhline(3500, linestyle = "dashed", color = "darkred",
                   linewidth = 3)
        ax.text(-0.01, 3500, "CH", ha = "right", va = "center", 
                color = "darkred", fontsize = fontsize-2,
                transform=ax.get_yaxis_transform())
        
        
    time_str = str(data.time.values[0])[0:13]
    print(f"Save plot for {time_str}")
    
    fig.tight_layout()
    #fig.tight_layout(rect=[0, 0, 1, 0.95])
    if plot_output:
        
        savestr = plot_output / f"{filled_contour_var}_{time_str}.{file_format}"
        plt.savefig(savestr, bbox_inches = "tight")
        plt.close()
    
    else:
        plt.show()
    
    return fig


#%%main call
'''
days = ["07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", 
        "18", "19", "20", "21", "22", "23", "24"]
for date in days:
    ds = load_data(date)
    
    #ds = resampling(ds)
    
    #calculations
    ds["U_knots"] = ms_to_knots(ds["U"])
    ds["V_knots"] = ms_to_knots(ds["V"])
    ds["W_knots"] = ms_to_knots(ds["W"])
    ds["VEL_knots"] = ms_to_knots(ds["VEL"])
    
    #plot parameter
    filled_contour_var = "VEL" #W
    height_var = "height"
    filled_contour_kwargs = {"levels":[0, 20], "spacing":1, 
                             "cbar_spacing":[0, 20], "label_spacing":[0, 20],
                             "extend":"max", "calc_norm":"yes", "cmap":cmap_horwind#coolwarm
                             }

    plot_kwargs = {"fontsize": 15, "figsize":(25,12)}
    
    #do the plot
    fig = plot_timeheight(
            data = ds, height_var = height_var, 
            filled_contour_var = filled_contour_var, 
            plot_kwargs = plot_kwargs,
            contourf_kwargs = filled_contour_kwargs
            
    )
    
'''