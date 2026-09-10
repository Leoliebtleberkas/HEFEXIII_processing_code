# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:26:17 2026

@author: leopo
"""

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

#%% class plot variable
class Plot_Variable:
    def __init__(
            self, levels, extend, colormap
            ):
        self.levels = levels
        self.extend = extend
        self.colormap = colormap
    #method for calcuating levels
    def calculate_levels(self, data, spacing = 2):
         
        level_min = 5*(data.min() // 5)
        level_max = 5*(np.ceil(data.max()/ 5))
        levels = np.arange(level_min, level_max+1, spacing)
        self.levels = levels
        return self
        

#%% color maps
#horizontal wind in cross section timeheight (max. 20 m/s)
cmap_horwind = np.array([
    [255, 255, 255],
    [255, 252, 203],
    [224, 243, 139],
    [171, 231, 131],
    [109, 220, 136],
    [0, 208, 149],
    [0, 197, 165],
    [0, 185, 180],
    [0, 173, 193],
    [0, 159, 204],
    [0, 144, 212],
    [55, 127, 216],
    [112, 108, 216],
    [145, 89, 211],
    [168, 69, 201],
    [183, 50, 188],
    [192, 35, 173],
    [195, 48, 93],
    [210, 103, 73],
    [246, 139, 69],
    [255, 204, 79]
])

cmap_horwind = cmap_horwind / 255.0
cmap_horwind = ListedColormap(cmap_horwind)


#vertical velocity Omega
colors = ["blue", "cornflowerblue", "white", "chocolate", "sienna"]
cmap_omega = LinearSegmentedColormap.from_list('omega', colors, N = 9)



#%%
horwind_timeheight = Plot_Variable(levels = np.arange(0, 21, 1),
                                   extend = "max", 
                                   colormap = cmap_horwind)
horwind_timeheight = Plot_Variable(levels = None,
                                   extend = "max", 
                                   colormap = cmap_horwind)
