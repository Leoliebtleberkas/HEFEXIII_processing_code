# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:16:01 2026

@author: leopo

"""

#import
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.contour as mcontour
import matplotlib.cbook as cbook
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
import textwrap
import cartopy.crs as ccrs
import os
from os.path import join
import sys
from scipy.constants import g
import metpy
import textwrap
from collections.abc import Mapping, Sequence

#%% functions

def get_norm(
        extend = "max", contourf_levels = np.arange(0, 21), n_colors = None
):
    
    if extend == "max":
        boundaries = np.hstack([contourf_levels, contourf_levels[-1]+1])

    elif extend == "min":
        boundaries = np.hstack([contourf_levels[0]-1, contourf_levels])

    elif extend == "both":
        boundaries = np.hstack([contourf_levels[0]-1,
                                contourf_levels,
                                contourf_levels[-1]+1])
        
    if n_colors is None:
        n_colors = len(boundaries)

    norm = BoundaryNorm(boundaries=boundaries, 
                        ncolors=n_colors - 1, clip=True)
    return norm

def get_levels(
        data, levels, spacing = 1, cbar_spacing = 5, zlevel = None, name = None
):
    
    #get levels from data
    if levels == "minmax":
        level_min = cbar_spacing*(data.min().values // cbar_spacing)
        level_max = cbar_spacing*(np.ceil(data.max().values / cbar_spacing))
        
        #double check because of bug
        level_min = np.min(level_min)
        level_max = np.max(level_max)
        
        levels = np.arange(level_min, level_max, spacing)
            
    #fixed customized levels without regular spacing, no action needed
    elif len(levels) > 2 and isinstance(levels, Sequence):
        levels = levels
        
    #height depending levels with reference value and range
    elif isinstance(levels, Mapping) and ("ref_value" and "delta_value" in levels.keys()):
        #get min value
        ref_value = float(levels["ref_value"])
        z_change = float(levels["z_change_meter"])
        delta = float(levels["delta_value"])
        min_value = ref_value + zlevel*z_change
        #max value
        max_value = min_value + delta
        
        levels = np.arange(int(min_value), int(max_value)+spacing, spacing)
    #levels with regular spacing  
    else:
 
        levels = np.arange(levels[0], levels[1]+1, spacing)
        
        
        
    return levels

def get_labellevels(
        levels, spacing
):
    if isinstance(spacing, int) or (isinstance(spacing, float) and spacing.is_integer()):
        #labellevels = levels[::spacing]
        labellevels = np.arange(levels[0], levels[-1]+1, spacing)
        
    else:
        labellevels = levels
        
    return labellevels

def get_cbarticks(
        levels, spacing
):
    if isinstance(spacing, int) or (isinstance(spacing, float) and spacing.is_integer()):
        cbar_ticks = np.arange(levels[0], levels[-1]+1, spacing)
        
    else:
        cbar_ticks = levels
        
    return cbar_ticks

#filled contour_plot
def filled_contour_plot(
        ax, x_plot, y_plot, filled_contour_data, cmap="Blues", levels=None,
        norm=None, extend="max", alpha = 1, projection = None, zorder=1
):
    
    filled_contour_plot = ax.contourf(x_plot, y_plot, filled_contour_data, 
                                      levels = levels, cmap = cmap, alpha = alpha,
                                      norm = norm, extend=extend, transform=projection,
                                      zorder=zorder)
    
    
    return filled_contour_plot

#contour_plot
def contour_plot(
        ax, x_plot, y_plot, contour_data, color="k", alpha=1, levels=None,
        linestyle="solid",linewidth=1, projection = None, zorder=3
):
    
    contour_plot = ax.contour(x_plot, y_plot, contour_data, 
                              levels = levels, colors = color, alpha=alpha,
                              linestyles=linestyle, linewidths=linewidth, transform=projection,
                              zorder=zorder)
    
    
    return contour_plot

#contourlabels
def add_contourlabels(
        ax, plot_handle, levels, color="grey", labelfloat = 0, zorder = 6,
        manual=False, rightside_up = False, inline_spacing=5, inline = True,
        use_clabeltext=True, fontsize = 10, add_units = False, units = None
):
    
    # Define formatting function
    if labelfloat == -1:
        if add_units:
            def fmt(x):
                rounded_value = round(x / 10)  # Round to nearest 10
                return f"{rounded_value} {units}"
        else:
            def fmt(x):
                return f"{round(x / 10)}"  # Round to nearest 10
    else:
        if add_units:
            def fmt(x):
                return f"{x:.{labelfloat}f} {units}"
        else:
            def fmt(x):
                return f"{x:.{labelfloat}f}"
    
    labels = ax.clabel(plot_handle, levels = levels, colors = color, manual=False,
               rightside_up = rightside_up, inline_spacing = inline_spacing, 
               inline = inline, use_clabeltext = use_clabeltext, 
               fontsize=fontsize, zorder=zorder, fmt = fmt)
    return labels


def add_barbkey(
        ax, fontsize = 10, key_length = 4, barb_unit = "knots"
):
    
    #barbkey settings
    if barb_unit == "m/s":
        barb_u = [0, 1, 2, 5]
        unit_str = "m/s"
        barb_increments = dict(half=1, full=2, flag=5)
    else:
        barb_u = [0, 5, 10, 50]         # U
        unit_str = "kn"
        barb_increments = dict(half=5, full=10, flag=50)
    barb_v = [0, 0, 0, 0]           # V
    
    #positions 0.95
    #key_positions = [(0.05, 1.05), (0.12, 1.05), (0.19, 1.05), (0.26, 1.05)]
    key_positions = [(0.69, 1.05), (0.76, 1.05), (0.83, 1.05), (0.90, 1.05)]
    
    #grey background 0.91
    #box = plt.Rectangle((0.01, 1.01), 0.3, 0.08, color='lightgray',
    box = plt.Rectangle((0.65, 1.01), 0.34, 0.08, color='lightgray',                
                        alpha=1, transform=ax.transAxes, zorder = 6,
                        clip_on = False)
    ax.add_patch(box)
    
    # Barb-Keys plotten
    for i, (x, y) in enumerate(key_positions):
        
        ax.barbs(x, y, [barb_u[i]], [barb_v[i]], color='k', 
                 transform=ax.transAxes, clip_on=False, zorder = 7, length = key_length, 
                 barb_increments = barb_increments,
                 sizes = dict(emptybarb = 0.07), fill_empty = True)
        #points for empty
        ax.barbs(x, y, [barb_v[i]], [barb_v[i]], color='k', 
                 transform=ax.transAxes, clip_on=False, zorder = 7, length = key_length, 
                 barb_increments = barb_increments,
                 sizes = dict(emptybarb = 0.07), fill_empty = True)
        if i == 0:
            string = f"< {barb_u[i+1]}"
        else:
            string = f"{barb_u[i]}"
            
        
        ax.annotate(text = f'{string} {unit_str}', xy = (x,y-0.01), ha='center', va = 'top',
                    xycoords = ax.transAxes, fontsize = fontsize, zorder = 10)
                #fontsize=20, transform=ax.transAxes, zorder = 10)
                #va='center', ha='center', transform=ax.transAxes, zorder = 10)