# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:31:51 2026

@author: leopo
"""
#import
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from metpy.calc import wind_direction, wind_speed
from metpy.units import units

#%% simple threshold cleaning
def threshold_control(
        data, threshold_u = None, threshold_v = None, threshold_w = None,
        threshold_spd = None, threshold_spd_low = None, alt_limit = None, 
        threshold_quality = None, 
        threshold_valid = None, threshold_snr = None, threshold_nvrad = None
):
    
    #print some stats
    for plot_var in ["u", "v", "w", "spd"]:
        print(f"{plot_var}: min.: {data[plot_var].min().values};", 
                          f"max.: {data[plot_var].max().values}")
    
    #u
    if threshold_u is not None:
        mask_u = (np.abs(data["u"]) <= threshold_u)
        data = data.where(mask_u)
    #v
    if threshold_v is not None:
        mask_v = (np.abs(data["v"]) <= threshold_v)
        data = data.where(mask_v)
    
    #w
    if threshold_w is not None:
        mask_w = (np.abs(data["w"]) <= threshold_w)
        data = data.where(mask_w)
    
    #valid radial components
    if threshold_valid is not None:
        mask_valid = (data["VALID"] >= threshold_valid)
        data = data.where(mask_valid)
    
    #signal to noise
    if threshold_snr is not None:
        mask_snr = (data["SNR"] >= threshold_snr)
        data = data.where(mask_snr)
    
    #quality binary --> 1 good, 0 bad
    if threshold_quality is not None:
        mask_quality = (data["QUALITY"] < threshold_quality)
        data = data.where(mask_quality)
    
    #number of radial velocities for wind speed calcuations
    if threshold_nvrad is not None:
        mask_nvrad = (data["nvrad"] > threshold_nvrad)
        data = data.where(mask_nvrad)
    
    
    #calculate spd again now an also mask that
    data["spd"] = np.sqrt(data.u**2 + data.v**2)
    
    if threshold_spd is not None:
        mask_speed = (np.abs(data["spd"]) <= threshold_spd)
        data = data.where(mask_speed)
    
    #additional threshold for lower levels
    if (threshold_spd_low is not None) and (alt_limit is not None):
        mask_spd_low = (
            (data.alt >= alt_limit)
            | (np.abs(data["spd"]) <= threshold_spd_low)
        )
        data = data.where(mask_spd_low)
    
    #print some stats after cleaning
    print("AFTER CLEANING:")
    for plot_var in ["u", "v", "w", "spd"]:
        print(f"{plot_var}: min.: {data[plot_var].min().values};", 
                          f"max.: {data[plot_var].max().values}")
    
    return data

#%%despiking
def despiking(ds, window = "1h", 
              n_start = 3.5, increase_per_loop = 0.3, max_iter = 20, 
              alt_despike = False, alt_window = 5, n_height_start = 2, drop_std = False,
              output_path_plot = None):
    
    #make sure ds is a dataset
    if isinstance(ds, xr.DataArray):
        ds = ds.to_Dataset()
        
    #save original variables for later
    vars_orig = ["u", "v", "w", "spd"]
    
    #empty dict for storing nan counting
    nan_counter = {}
    nan_counter_height = {}
    
    #set initital value for n * std as limit
    n = float(n_start)
    n_height = float(n_height_start)
    increase_per_loop = float(increase_per_loop)
    
    #iterative despiking
    for i in range(1, max_iter+1):
        
        #set up counter for nans
        nan_counter[n] = {}
        
        #resample to avg period and calculate period mean and std
        ds_mean = ds[["u", "v", "w", "spd"]].resample(time = window).mean()
        ds_std = ds[["u", "v", "w", "spd"]].resample(time = window).std() * n
        
        #rename variables
        ds_mean = ds_mean.rename({var: f"{var}_mean" for var in vars_orig})
        ds_std = ds_std.rename({var: f"{var}_std" for var in vars_orig})
        
        #reindex to original ds
        ds_mean = ds_mean.reindex(time = ds.time, method = "ffill")
        ds_std = ds_std.reindex(time = ds.time, method = "ffill")
        
        #back to old ds
        ds = ds.assign(**ds_mean, **ds_std)
        
        
        #set up masks
        u_mask = (ds["u"] >= ds["u_mean"] - ds["u_std"]) & (ds["u"] <= ds["u_mean"] + ds["u_std"]) 
        v_mask = (ds["v"] >= ds["v_mean"] - ds["v_std"]) & (ds["v"] <= ds["v_mean"] + ds["v_std"]) 
        w_mask = (ds["w"] >= ds["w_mean"] - ds["w_std"]) & (ds["w"] <= ds["w_mean"] + ds["w_std"])
        spd_mask = (ds["spd"] >= ds["spd_mean"] - ds["spd_std"]) & (ds["spd"] <= ds["spd_mean"] + ds["spd_std"])
            
        #stop criterion for loop
        any_new_nans = False
        
        #loop over variables
        var_list = ["u", "v", "w", "spd"]
        mask_list = [u_mask, v_mask, w_mask, spd_mask]        
            
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
        
        
        #drop mean and std variables    
        vars_to_drop = [v for v in ds.data_vars
                        if v.endswith("_mean") or v.endswith("_std")]
        ds = ds.drop_vars(vars_to_drop)
        
        #stop conditions
        if not any_new_nans:
            print(f"no new NaNs produced with {n} * std. temporal despiking finished!")
            break
        
        #update n
        n =  round(n_start + i * increase_per_loop, 1)
        
        
    # ---- additional despiking over height 
    if alt_despike:
        for j in range(1, max_iter+1):
        
            #set up counter for nans
            nan_counter_height[n] = {}
            #stop criterion for loop
            any_new_nans = False

            ds_alt_mean = (
                ds[list(vars_orig)]
                .rolling(alt=alt_window, center=True, min_periods=1)
                .mean()
            )
        
            ds_alt_std = (
                ds[list(vars_orig)]
                .rolling(alt=alt_window, center=True, min_periods=1)
                .std()
                * n_height
            )
        
            ds_alt_mean = ds_alt_mean.rename(
                {var: f"{var}_alt_mean" for var in vars_orig}
            )
        
            ds_alt_std = ds_alt_std.rename(
                {var: f"{var}_alt_std" for var in vars_orig}
            )
        
            ds = ds.assign(**ds_alt_mean, **ds_alt_std)
        
            for var in vars_orig:
        
                alt_mask = (
                    (ds[var] >= ds[f"{var}_alt_mean"] - ds[f"{var}_alt_std"])
                    & (ds[var] <= ds[f"{var}_alt_mean"] + ds[f"{var}_alt_std"])
                )
        
                before_nan = ds[var].isnull()
        
                ds[var] = ds[var].where(alt_mask)
        
                after_nan = ds[var].isnull()
        
                new_nans = (after_nan & ~before_nan).sum(dim=("time", "alt"))
        
                print(f"{var}, alt despiking std*{n_height}: {new_nans.values} produced")
                
            #check stop criterion
            if (new_nans > 0).any():
                any_new_nans = True
        
        
            #drop mean and std variables    
            vars_to_drop = [v for v in ds.data_vars
                            if v.endswith("_mean") or v.endswith("_std")]
            ds = ds.drop_vars(vars_to_drop)
        
            #stop conditions
            if not any_new_nans:
                print(f"no new NaNs produced with {n_height} * std. spatial despiking finished!")
                break
            
            #update n_height
            n_height =  round(n_height_start + j * increase_per_loop, 1)
        
    return ds, nan_counter

def resample_and_save(
        data, resample_window, output_path, savename
):
    ds_res = data.resample(time = resample_window, closed = "right",label = "right").mean()
    #speed
    u = ds_res["u"]
    v = ds_res["v"]
    ds_res["spd"] = wind_speed(u*units('m/s'), v*units('m/s'))
    #direction
    ds_res["dir"] = wind_direction(u*units('m/s'), v*units('m/s'))
    
    
    ds_res.to_netcdf(output_path / fr"{savename}_{resample_window}.nc")
    
    return ds_res


#interpolate to same vertical resolution
def interpolate_height(
        ds1, ds2, height_coord
):
    #old
    #ds2_interp = ds2.interp(alt=ds1.alt,
    #                        kwargs={"fill_value": "extrapolate"}
     #                       )
     
    #---
    #only to height range which exists in both
    alt_min = max(ds1.alt.min(), ds2.alt.min())
    alt_max = min(ds1.alt.max(), ds2.alt.max())

    ds1 = ds1.sel(alt=slice(None, alt_max))

    ds2_interp = ds2.interp(alt=ds1.alt)

    ds = xr.concat([ds1, ds2_interp], dim = "time")
    ds = ds.sortby("time")
    
    return ds
#%% plotting
def test_plots(data, plot_vars, output_plot, file_ext = None, fontsize = 13):
    
    y, x = np.meshgrid(data.alt, data.time)
    
    for plot_var in plot_vars:
        
        fig, ax = plt.subplots(figsize = (16, 6))
        
        
        c_plot = ax.contourf(x, y, data[plot_var], levels = 20)
        
        #colorbar
        cbar = fig.colorbar(c_plot, ax=ax)
        cbar.set_label(f"{plot_var}")
    
    
        #labels title etc
        ax.set_ylabel("time", fontsize = fontsize)
        ax.set_ylabel("altitude[m]", fontsize = fontsize)  
        ax.set_title(plot_var.capitalize(), fontsize = fontsize + 3)
        #save
        if file_ext:
            plt.savefig(output_plot / fr"{plot_var}_{file_ext}.png")
        else:
            plt.savefig(output_plot / fr"{plot_var}.png")
        plt.close()
        
        #histogram
        data_vec = data[plot_var].values.flatten()
        fig, ax = plt.subplots()
        ax.hist(data_vec, bins = 20)
        
        if file_ext:
            plt.savefig(output_plot / fr"{plot_var}_hist_{file_ext}.png")
        else:
            plt.savefig(output_plot / fr"{plot_var}_hist.png")
        plt.close()
