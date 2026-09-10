# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 16:18:32 2025

@author: leopo
"""

import numpy as np
import xarray as xr
import os

os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\helper_functions")
from postprocess_plots import plot_despiking_corrections


#functions
def threshold_correction(
        ds
):
    
    #threshold adapted from Aubinet et al. 2012
    #|u| < 30 m/s
    #|w| < 5 m/s
    #|T - Tm| < 20 K , T_m monthly mean, here campaign mean
    
    #define masks
    #hor wind
    wspeed = np.sqrt(ds["u"]**2 + ds["v"]**2)
    mask_horwind = (wspeed < 30)
    #vertical wind
    mask_w = np.abs(ds["w"]) < 5
    #temperature
    mask_t = ds["Ts"] < 20
    
    #apply 
    ds["u"] = ds["u"].where(mask_horwind)
    ds["v"] = ds["v"].where(mask_horwind)
    ds["w"] = ds["w"].where(mask_w)
    ds["Ts"] = ds["Ts"].where(mask_t)
    #temperature average
    Tmean = ds["Ts"].mean(dim = "time")
    mask_t1 = np.abs(ds["Ts"] - Tmean) < 20
    ds["Ts"] = ds["Ts"].where(mask_t1)
    
    #specific humidity
    if "q" in ds.data_vars:
        mask_q = (ds["q"] < 0.015) & (ds["q"] > 0)
        ds["q"] = ds["q"].where(mask_q)
    #density moist air
    if "rho_v" in ds.data_vars:
        mask_rhov = (ds["rho_v"] < 8) & (ds["rho_v"] > 0)
        ds["rho_v"] = ds["rho_v"].where(mask_rhov)
    #pressure
    if "p" in ds.data_vars:
        mask_p = (ds["p"] > 700) & (ds["p"] < 800)
        ds["p"] = ds["p"].where(mask_p)
    #air temperature 
    if "Tair" in ds.data_vars:
        mask_Tair = ds["Tair"] < 20
        ds["Tair"] = ds["Tair"].where(mask_Tair)
        #temperature average
        Tmean = ds["Tair"].mean(dim = "time")
        mask_t1 = np.abs(ds["Tair"] - Tmean) < 20
        ds["Tair"] = ds["Tair"].where(mask_t1)
    
    
    return ds


def despiking(ds, window = "15min", 
              n_start = 4, increase_per_loop = 0.3, max_iter = 20, 
              drop_std = False,
              output_path_plot = None):
    
    #make sure ds is a dataset
    if isinstance(ds, xr.DataArray):
        ds = ds.to_Dataset()
        
    #save original variables for later
    vars_orig = list(ds.data_vars)
    
    #empty dict for storing nan counting
    nan_counter = {}
    
    #set initital value for n * std as limit
    n = float(n_start)
    increase_per_loop = float(increase_per_loop)
    
    #iterative despiking
    for i in range(1, max_iter+1):
        
        #set up counter for nans
        nan_counter[n] = {}
        
        #resample to avg period and calculate period mean and std
        ds_mean = ds.resample(time = window).mean()
        ds_std = ds.resample(time = window).std() * n
        
        #rename variables
        ds_mean = ds_mean.rename({var: f"{var}_mean" for var in vars_orig})
        ds_std = ds_std.rename({var: f"{var}_std" for var in vars_orig})
        
        #reindex to original ds
        ds_mean = ds_mean.reindex(time = ds.time, method = "ffill")
        ds_std = ds_std.reindex(time = ds.time, method = "ffill")
        
        #back to old ds
        ds = ds.assign(**ds_mean, **ds_std)
        
        #save raw data for plot --> only if plot is wanted otherwise save RAM
        if output_path_plot is not None:
            ds_raw = ds.copy(deep = True)
        
        #set up masks
        u_mask = (ds["u"] >= ds["u_mean"] - ds["u_std"]) & (ds["u"] <= ds["u_mean"] + ds["u_std"]) 
        v_mask = (ds["v"] >= ds["v_mean"] - ds["v_std"]) & (ds["v"] <= ds["v_mean"] + ds["v_std"]) 
        w_mask = (ds["w"] >= ds["w_mean"] - ds["w_std"]) & (ds["w"] <= ds["w_mean"] + ds["w_std"])
        temp_mask = (ds["Ts"] >= ds["Ts_mean"] - ds["Ts_std"]) & (ds["Ts"] <= ds["Ts_mean"] + ds["Ts_std"])
            
        #stop criterion for loop
        any_new_nans = False
        
        #loop over variables
        var_list = ["u", "v", "w", "Ts"]
        mask_list = [u_mask, v_mask, w_mask, temp_mask]
        additional_mask = {}
        #include specific humidity if measured
        if "q" in ds.data_vars:
            q_mask = ((ds["q"] >= ds["q_mean"] - ds["q_std"]) & 
                      (ds["q"] <= ds["q_mean"] + ds["q_std"]))
            additional_mask["q"] = q_mask
            
        #include density of moist air if measured
        if "rho_v" in ds.data_vars:
            rho_v_mask = ((ds["rho_v"] >= ds["rho_v_mean"] - ds["rho_v_std"]) & 
                      (ds["rho_v"] <= ds["rho_v_mean"] + ds["rho_v_std"]))
            additional_mask["rho_v"] = rho_v_mask
        #include mass mixing ratio of water if measured
        if "wv" in ds.data_vars:
            wv_mask = ((ds["wv"] >= ds["wv_mean"] - ds["wv_std"]) & 
                      (ds["wv"] <= ds["wv_mean"] + ds["wv_std"]))
            
            additional_mask["wv"] = wv_mask
            
        if "p" in ds.data_vars:
            p_mask = ((ds["p"] >= ds["p_mean"] - ds["p_std"]) &
                      (ds["p"] <= ds["p_mean"] + ds["p_std"]))
            
            additional_mask["p"] = p_mask
            
        for var in ["q", "rho_v", "w_v", "p"]:
            if var in ds.data_vars:
                var_list.append(var)
                mask_list.append(additional_mask[var])
        
            
        for var, mask in zip(var_list, mask_list):
            
            before_nan = ds[var].isnull()
            ds[var] = ds[var].where(mask)
            after_nan = ds[var].isnull()
            
            #count produced nans only
            new_nans = (after_nan & ~before_nan).sum(dim="time")
            
            print(f"{var}, std*{n}: {new_nans} produced")
            print(f"{var}, max. values: {ds[var].max(dim = 'time')}")
            
            nan_counter[n][var] = new_nans
            
            #check stop criterion
            if (new_nans > 0).any():
                any_new_nans = True
        
        #optional plots for visual validation
        if output_path_plot:
            ds_raw = ds_raw.sel(time = "2025-08-10")#slice("2025-08-15", "2025-08-16"))
            ds_despiked = ds.sel(time = "2025-08-10")#slice("2025-08-15", "2025-08-16"))
            for h in ds.heights:
                nan_dict_height = {}
                nan_dict_height = {var: nan_counter[n][var].sel(heights = h).item() for var in nan_counter[n]}
                plot_despiking_corrections(
                        ds_raw.sel(heights = h), 
                        ds_despiked = ds_despiked.sel(heights = h), 
                        nan_dict = nan_dict_height, 
                        variables = vars_orig, n_std = n,
                        orig_data_len = len(ds_raw.time),
                        sensor_name = f"metek_{str(h.item()).replace('.', '')}m_{n}std", 
                        output_path = output_path_plot, window = window, 
                        background_step = 5
                )
        
        #drop mean and std variables    
        vars_to_drop = [v for v in ds.data_vars
                        if v.endswith("_mean") or v.endswith("_std")]
        ds = ds.drop_vars(vars_to_drop)
        
        #stop conditions
        if not any_new_nans:
            print(f"no new NaNs produced with {n} * std. Despiking finished!")
            break
        
        #update n
        n =  round(n_start + i * increase_per_loop, 1)
    return ds, nan_counter

def remove_fragmentary_data(
        ds, window = "15min",  remove_threshold = 0.2
):
    
    group_list = []
    
    for label, group in ds.resample(time = window, label = "right", closed = "right"):
        nan_portion = group.isnull().sum(dim = "time") / group.sizes["time"]
        if (nan_portion > remove_threshold).to_array().any():
            group = xr.full_like(group, np.nan)
        else:
            group = group
            
        group_list.append(group)
        
    ds = xr.concat(group_list, dim = "time")
    ds = ds.sortby("time")
    return ds            
