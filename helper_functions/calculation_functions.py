# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 15:59:05 2026

@author: leopo
"""

#import packages
import metpy.calc as mcalc
from metpy.units import units
import xarray as xr
#%%constants 
Rd = 287.15 # J/(kg*K)
Rv = 461.5  # J/(kg*K)

#%% functions
# ----- ATTENTION: FUNCTIONS PARTLY USE FIXED DEFAULT UNITS -----
def water_vapour_pressure_from_moistdensity(
        moist_density, temperature, inkg = True, inC = True
):
    if inkg:
        moist_density = moist_density / 1000
    if inC:
        temperature = temperature + 273.15
    
    Rv = 461.5 #J / (kg/K)
    e = moist_density * Rv * temperature
    
    return e

def mixing_ratio_from_watervapour(
        watervapourpressure, totalpressure, unit1 = "Pa", unit2 = "hPa"
):
    wv = mcalc.mixing_ratio(watervapourpressure *units(unit1), 
                            totalpressure*units(unit2))
    
    return wv

def specific_humidity_from_mixing_ratio(
        mixingratio, unit = "kg/kg"
):
    q = mcalc.specific_humidity_from_mixing_ratio(mixingratio*units(unit))
    
    return q

def dry_air_density(
        total_pressure, watervapourpressure, temperature, inC = True
):
    if inC:
        temperature = temperature + 273.15
    Rd = 287.15 #J/(kg * K)
    rho_d = ((total_pressure*100 - (watervapourpressure)) /   #dry air pressure
             (Rd * temperature))        #dry air constant and temp
    
    return rho_d

def calc_backward_gradient(
        ds, var = "meanU", var_type = "wind", gradient_name = None 
):
    
    #check temp unit
    unit = None
    if var_type == "temp":
        if ds[var].mean() > 100:
            unit = "K"
    
    #add sfc layer
    new_layer = ds[var].isel(heights=0).copy() * 0 
    new_layer = new_layer.expand_dims(heights=[0]).to_dataset(name=var)
    ds_sfc = xr.concat([new_layer, ds], dim="heights").sortby("heights")
    
    #check order
    ds_sfc = ds_sfc.transpose("time", "heights")
    
    if unit == "K":
        ds_sfc[var].loc[dict(heights=0)] = (
        ds_sfc[var].sel(heights=0) + 273.15
        )
    
    #gradient calculation
    
    grad = ds_sfc[var].diff("heights") / ds_sfc.heights.diff("heights")
    
    # to dataset
    if gradient_name is not None:
        grad_name = gradient_name
    else:
        grad_name = f"grad_{var}"
    
    ds[grad_name] = grad
        
    return ds

    
# =============================================================================
# #%%test
# from pathlib import Path
# import xarray as xr
# folder = Path(r"D:\HEFEXIII\Tower\L2")
# file = r"smartflux_L2_20Hz_orig.nc"
# 
# ds = xr.open_dataset(folder / file)
# 
# #%%
# ds["e"] = water_vapour_pressure_from_moistdensity(ds["rho_v"]/1000, ds["Tair"])    
# 
# wv = mixing_ratio_from_watervapour(ds["e"], ds["p"])
# 
# q = specific_humidity_from_mixing_ratio(wv)
# 
# rho_d = dry_air_density(ds["p"], ds["e"], ds["Tair"])
# =============================================================================
