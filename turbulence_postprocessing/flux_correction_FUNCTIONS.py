# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:27:01 2026

@author: leopo
"""

#import packages
import xarray as xr
import numpy as np
from pathlib import Path

#%%constants
Lv = 2.5e6     #J/(kg*K)
cp = 1004.67   #J/kg
Rd = 287.15    #J(kg*K)
Rv = 461.5     #J/(kg*K)  

#%%flux correction functions
#sensible heat flux
def SND_correction(
        meanT, moisture_flux, inC = True
):
    
    if inC:
        meanT = meanT+273.15
        
    corr = -0.51 * meanT * moisture_flux
    
    return corr


#latent heat flux
def WPL_correction(
        rho_v_mean, rho_d_mean, meanT, wT, wrho_v, mu = 1.6077, inC = True
):
    
    if inC:
        meanT = meanT+273.15
        
    sigma = rho_v_mean / rho_d_mean
    
    corr = (rho_v_mean/meanT)*wT + mu*sigma*wrho_v + mu*sigma*(rho_v_mean/meanT)*wT
    
    return corr

#gas constant of moist air with specific humidity in kg/kg
def calc_gas_constant(
        q
):
    '''
    calculates gas constant of moist air

    Parameters
    ----------
    q : float
        specific humidity in kg/kg.

    Returns
    -------
    R : float
        calculated gas constant [J/(kg*K)] of moist air.

    '''
    R = (1 - q) * Rd + q * Rv
    
    return R

#specific heat of moist air
def calc_spec_heat(
        q
):
    '''
    
    calculates specific heat of moist air
    Parameters
    ----------
    q : float
        specific humidity in kg/kg.

    Returns
    -------
    cp_moist : float
        specific heat [J/kg] of moist air.

    '''
    cp_moist = cp * (1 + 0.84 * q)
    
    return cp_moist

#temperature dependend vaporization enthalpie 
def calc_Lv(
        T 
):
    '''
    

    Parameters
    ----------
    T : float
        temperature in °C or Kelvin.

    Returns
    -------
    Lv_t : float
        temperature dependent vaporization enthalpie [J/(kg*K)].

    '''
    #convert to Kelvin if needed
    if T.mean() < 200:
        T = T +  273.15
        
    Lv_t = Lv + (1850 - 4200) * (T - 273.15)
    
    return Lv_t


#get addtional data
def get_additional_data(
        folder = r"D:\HEFEXIII\Tower\L3\no_sectorwise",
        file = r"smartflux_L3_20Hz_30min.nc"
):
    
    ds_orig = xr.open_dataset(Path(folder) / file)
    print("Additional data for flux corrections loaded...")
    
    return ds_orig

#dynamic sensible heat flux
def calc_dynamic_sensible_heat(
        wT, air_density, cp = 1004
):
    H = air_density * cp * wT
    
    return H

#dynamic latent heat flux from w'q' in m/s * g/kg or m/s * kg/kg
def calc_dynamic_latent_heat(
        wq, air_density, Lv = 2.5e6
):
    LE = air_density * Lv * wq
    
    return LE

#main flux correction function
def correct_fluxes(
        ds_flux, var_moisture_flux = "wq", var_temperature_flux = "wT",
        var_WPL_moisture = "wrho_v"
        ):
    
    ds_orig = get_additional_data()
    
    ds_orig = ds_orig.sel(time = slice(ds_flux.time[0], ds_flux.time[-1]))
    
    #unit check
    if ds_orig["rho_v"].mean() > 1:
        ds_orig["rho_v"] = ds_orig["rho_v"] / 1000
    
    #calculate needed mean values
    #ds_orig["rho"] = ds_orig["p"]*100 / ((ds_orig["Tair"]+273.15) * Rd)
    ds_orig["rho"] = ds_orig["rho_v"] + ds_orig["rho_d"] 
    rho_v_mean = ds_orig["rho_v"].resample(time = "30min", label = "right", 
                                           closed = "right").mean() 
    rho_d_mean = ds_orig["rho_d"].resample(time = "30min", label = "right", 
                                           closed = "right").mean()
    rho_mean = ds_orig["rho"].resample(time = "30min", label = "right", 
                                       closed = "right").mean()
    
    
    #calculate flux corrections
    #do it twice 
    #1. SND
    corr_wT = SND_correction(ds_flux["meanT"], ds_flux[var_moisture_flux])
    ds_flux["wT_corr"] = ds_flux[var_temperature_flux] + corr_wT
    #1. WPL
    corr_wq = WPL_correction(rho_v_mean, rho_d_mean, 
                             ds_flux["meanT"], ds_flux["wT_corr"], 
                             ds_flux[var_WPL_moisture]/1000)

    ds_flux["wq_corr"] = ds_flux["wq"] + corr_wq/rho_d_mean
    
    #2. SND
    corr_wT2 = SND_correction(ds_flux.meanT, ds_flux["wq_corr"])
    ds_flux["wT_corr2"] = ds_flux["wT_corr"] + corr_wT2
    #2. WPL
    corr_wq2 = WPL_correction(rho_v_mean, rho_d_mean, 
                              ds_flux["meanT"], ds_flux["wT_corr2"], 
                              ds_flux["wq_corr"]*rho_d_mean)
    ds_flux["wq_corr2"] = ds_flux["wq_corr"] + corr_wq2/rho_d_mean
    
    #in dynamic fluxes
    ds_flux["H"] = calc_dynamic_sensible_heat(wT = ds_flux[var_temperature_flux],
                                              air_density = rho_mean)
    ds_flux["LE"] = calc_dynamic_latent_heat(wq = ds_flux[var_moisture_flux],
                                             air_density = rho_mean)
    
    ds_flux["H_corr"] = calc_dynamic_sensible_heat(wT = ds_flux["wT_corr2"],
                                                   air_density = rho_mean)
    ds_flux["LE_corr"] = calc_dynamic_latent_heat(wq = ds_flux["wq_corr2"],
                                                  air_density = rho_mean)
    
    return ds_flux

#helper function which adds dynamic sensible heat calculated with 1m data to other levels
def add_dynamic_sensible_heat(
        wT, smartflux_datapath, window
):
    
    ds_smart = xr.open_dataset(smartflux_datapath)
    
    #check density units
    if ds_smart["rho_v"].mean() > 2:
        ds_smart["rho_v"] = ds_smart["rho_v"] / 1000
        
    if ds_smart["rho_d"].mean() > 2:
        ds_smart["rho_d"] = ds_smart["rho_d"] / 1000
        
    #check spec. humidity units
    if ds_smart["q"].mean() > 1:
        ds_smart["q"] = ds_smart["q"] / 1000
    
    #air_density
    rho = ds_smart["rho_v"] + ds_smart["rho_d"]
    
    #specific heat
    cp = calc_spec_heat(ds_smart["q"])
    
    #resample to window
    rho_avg = rho.resample(time = window, label = "right", closed = "right").mean()
    rho_avg = rho_avg.reindex(time = wT.time).bfill(dim = "time").ffill(dim = "time")
    cp_avg = cp.resample(time = window, label = "right", closed = "right").mean()
    cp_avg = cp_avg.reindex(time = wT.time).bfill(dim = "time").ffill(dim = "time")
    
    #calculate dynamic heat flux
    H = wT * rho_avg.squeeze() * cp_avg.squeeze()
    
    return H