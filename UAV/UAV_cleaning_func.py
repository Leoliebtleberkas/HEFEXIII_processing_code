# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:21:34 2026

@author: leopo
"""

#import
import numpy as np
import xarray as xr

#%% simple threshold cleaning
def threshold_control(
        data, threshold_dict, lower_threshold_dict = None
):
    
    #loop over the dict
    for var, threshold in threshold_dict.items():
        
        #print some stats
        print(f"{var}: min.: {data[var].min().values};", 
              f"max.: {data[var].max().values}")
        
        mask = np.abs(data[var]) <= threshold
        data[var] = data[var].where(mask)
        
        #in case of wind: clean also direction
        if var in ["spd", "u", "v"]:
            data["dir"] = data["dir"].where(mask)
    
        #print some stats after cleaning
        print(f"{var}: min.: {data[var].min().values};", 
              f"max.: {data[var].max().values}")
        
    #additional cleaning for variables which can suffer from wrong slightly negative values (=temp)
    if lower_threshold_dict is not None:
        
        #loop over the dict
        for var, threshold in lower_threshold_dict.items():
            
            #print some stats
            print(f"{var}: min.: {data[var].min().values};", 
                  f"max.: {data[var].max().values}")
            
            mask = data[var] > threshold
            data[var] = data[var].where(mask)
            
            #in case of wind: clean also direction
            if var in ["spd", "u", "v"]:
                data["dir"] = data["dir"].where(mask)
                
            #in case of temp and hum: clean also both
            if var == "temp":
                data["qv"] = data["qv"].where(mask)
                data["rh"] = data["rh"].where(mask)
        
            #print some stats after cleaning
            print(f"{var}: min.: {data[var].min().values};", 
                  f"max.: {data[var].max().values}")
        
    
    return data

#%%despiking

def despike(
    ds, n_dict, alt_window=5, increase_per_loop=0.3, max_iter=20,
):
    """
    Iteratives Despiking along altitude axis.

    Parameters
    ----------
    ds : xr.Dataset
    n_dict : dict
        z.B. {"u": 2.5, "v": 2.5, "temp": 3.0}
        variables and starting values für n * std.
    alt_window : int
        window size for rolling mean/std.
    increase_per_loop : float
        enhancement of n per loop.
    max_iter : int

    Returns
    -------
    ds_clean : xr.Dataset
    nan_counter : dict
    """

    ds = ds.copy()

    nan_counter = {}

    for var, n_start in n_dict.items():

        if var not in ds:
            print(f"{var} not found.")
            continue

        n = float(n_start)
        nan_counter[var] = {}

        for i in range(max_iter):

            mean = (
                ds[var]
                .rolling(alt=alt_window, center=True, min_periods=1)
                .mean()
            )

            std = (
                ds[var]
                .rolling(alt=alt_window, center=True, min_periods=1)
                .std()
            )

            mask = np.abs(ds[var] - mean) <= n * std

            before_nan = ds[var].isnull()

            ds[var] = ds[var].where(mask)

            after_nan = ds[var].isnull()

            new_nans = (after_nan & ~before_nan).sum()

            nan_counter[var][n] = int(new_nans)

            print(f"{var}: n={n:.1f}, removed {int(new_nans)} values")

            if new_nans == 0:
                print(f"{var}: finished\n")
                break

            n += increase_per_loop

    return ds, nan_counter



