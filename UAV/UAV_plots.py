# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:05:38 2026

@author: leopo
"""
#import
from matplotlib import pyplot as plt
#%%plot function to evaluate profiles
def plot_uav_profiles(
        ds, var_dict, uav_color_dict, figsize = (8, 12), nrows = 2, ncols = 2,
        fontsize = 14, output_path = None, ncols_legend = None,
        sharey = True, agl = True, xlim_dict = None
):
    #only works for one time step
    n_time = ds.dims.get("time", 0)

    if n_time > 1:
        raise ValueError("only possible to plot one time step")
        
    fig, axs = plt.subplots(nrows=nrows, ncols = ncols, figsize = figsize,
                            sharey = sharey)
    
    #number of plot vars has to agree with number of subplots
    if len(var_dict) != len(axs.flatten()):
        raise ValueError("number of plottet variables has to agree with number of subplots!")
    
    for i,var in enumerate(var_dict.keys()):
        
        ax = axs.flatten()[i]
        
        for location in uav_color_dict.keys():
            
            plot_data = ds[var].sel(location = location).dropna(dim="alt")
            
            #alt. AMSL
            y = plot_data.alt
            
            #plot only if other values than NaN
            if plot_data.size > 0:
                
                #plot above surface
                if agl:
                    y = y - y[0]
                
                #actual plot
                ax.plot(plot_data, y, color = uav_color_dict[location],
                        label = location, linewidth = 2)
                
            #settings
            ax.set_xlabel(var_dict[var], fontsize = fontsize)
            #fixed ticks for wind direction
            if var == "dir":
                ax.set_xticks([90, 180, 270, 360])
                ax.set_xlim(0, 360)
                
            ax.tick_params(axis = "both", labelsize = fontsize-2)
            
    #external legend
    if ncols_legend is None:
        ncols_legend = len(ds.location)
        
    fig.subplots_adjust(top=0.8)
    axs.flatten()[0].legend(fontsize = fontsize, loc = "lower left", ncols = ncols_legend, 
                  bbox_to_anchor=(0, 1.0), bbox_transform = axs.flatten()[0].transAxes, 
                  frameon = False)
    
    #ylabel
    if agl:
        ylabel = r"height [m AGL]"
    else:
        ylabel = r"alt. [m AMSL]"
        
    axs.flatten()[0].set_ylabel(ylabel, fontsize = fontsize)
    
    #axis limits
    if xlim_dict:
        #wind speed
        if xlim_dict["spd"] is not None:
            axs.flatten()[0].set_xlim(0, xlim_dict["spd"])
        #wind direction
        if xlim_dict["dir"] is not None:
            axs.flatten()[1].set_xlim(0, xlim_dict["dir"])
        #temperature
        if xlim_dict["temp"] is not None:
            axs.flatten()[2].set_xlim(0, xlim_dict["temp"])
        #spec. humidity
        if xlim_dict["qv"] is not None:
            axs.flatten()[3].set_xlim(0, xlim_dict["qv"])
        
    #grid on
    for ax in axs.flatten():
        ax.grid(axis = "y")
    
    if output_path is None:
        plt.show()
        
    else:
        plt.savefig(output_path, bbox_inches = "tight")
        plt.close()