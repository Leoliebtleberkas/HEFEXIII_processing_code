# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 11:45:27 2025

@author: leopo
"""

#set to right working directory
#check working directory
import os 
print(os.getcwd())

#set to right working directory
os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\turbulence_postprocessing")
#%%import packages
import xarray as xr
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import json
from postprocess_helpers import (fill_gaps, double_rotation, 
                         calc_cov, make_time_regular,
                         fluxes_calculation, stationarity,
                         stationarity_new, stationarity_new_mean)
from functions import get_fluctuations
from MRD_functions import multiresolution
from spectral_analysis import spectra_eps
from structure_functions import structure_functions_epsilon
from autocorrelation import autocorrelation
print(os.getcwd())
from flux_correction_FUNCTIONS import correct_fluxes, add_dynamic_sensible_heat

#helper functions
os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\helper_functions")
from planarfit_xr_functions import execute_planar_fit
from postprocess_plots import plot_timeseries_multiple, plot_freq_spectra_timecolors

#%% turbulence postprocessing workflow
# count nans
def nan_counter(ds):
    for h in ds.heights:
        ds_h = ds.sel(heights = h)
        print(f"height: {h.values}")
        for var in ds_h.data_vars:
            nansum = ds_h[var].isnull().sum().values
            print(var)
            print(f"sum NaN:{nansum}")
            print(f"NaN percentage: {100*nansum / len(ds_h[var])}")
            
def inf_counter(ds):
    for h in ds.heights:
        ds_h = ds.sel(heights = h)
        print(f"height: {h.values}")
        for var in ds_h.data_vars:
            da = ds_h[var]
            
            # Inf-Maske sicher erstellen (funktioniert immer, auch wenn .isinf() mal fehlt)
            try:
                inf_mask = da.isinf()  # DataArray-Methode
            except AttributeError:
                inf_mask = xr.apply_ufunc(np.isinf, da)
                
            infsum = inf_mask.sum().item()
            print(var)
            print(f"sum NaN:{infsum}")
            print(f"NaN percentage: {100*infsum / len(ds_h[var])}")

def main_postprocess(
        ds, config
):
    """Function to postprocess the data from sonic anemometers.
    originally written by Samuele Mosso, University of Innsbruck, 
    adapted by Leopold Schlagbauer, University of Innsbruck
        INPUT:
            ds:  to be an xarray dataset, with the variables u,v,w and tc, and dimensions time and heights.
            If only one height is present, provide a dimension heights with length one, since the height is needed for
            spectral cutoff determination.
            config: configuration dictionary with following entries:
                'window': window size for averaging, in the form 'nmin' with n integer
                'avg_method': 'detrend' or 'block' for linear detrending or block averaging
                'gap_filling': gap filling method, only 'interp' supported for now
                'spectra': boolean, compute the spectra
                'strfun': boolean, compute the 2nd order structure functions
                'autocorr': boolean, compute the autocorrelation functions
                'MRD': boolean, compute the multiresolution flux decompositions

        OUTPUT: a dictionary with the following keys:
            'stats': dataset with the statistics like reynolds stress tensor ecc.
            'spectra': dataset with the spectra
            'strfun': dataset with the second order structure functions
            'autocorr': dataset with the autocorrelation functions
            'MRD': dataset with the multiresolution flux decompositions
    """
  
    #print("NaNs raw data: ")
    #nan_counter(ds)
    #add artificial height dimension if not present
    if "heights" not in ds.coords:
        ds = ds.assign_coords(heights=[1])
        warnings.warn(
            "Dimension heights was not present, added with fake value, \
                this will cause an imprecise"
            " determination of the cutoff in spectral frequency"
        )
            
    # time always first dimension
    if list(ds.dims)[0] == 'time':
        ds = ds
    else:
        # rearrange
        new_order = ['time'] + [d for d in ds.dims if d != 'time']
        ds = ds.transpose(*new_order)
        warnings.warn(f"time was not first Dimension. \
                      Rearranged to {new_order}.")

            
    #rename variables if necessary
    if config["var_rename"] != "no":
        ds = ds.rename(config["var_rename"])
            
    results = {}
    
    #check for and add missing timesteps --> will increase number of nans if 
    ds = make_time_regular(ds)
    
    # gap_filling
    ds, nan_perc = fill_gaps(ds, config, )
    #print("NaNs after gap filling: ")
    #nan_counter(ds)

    #rotation method
    # double rotation
    if config["rotation"] == "double_rotation":
        ds, rotation = double_rotation(ds, config)
    # planar fit
    elif config["rotation"] == "planar_fit":
        rotate_vars = ["u", "v", "w", "Ts"]
        if "q" in ds.data_vars:
            rotate_vars = ["u", "v", "w", "Ts", "q"]
        uvw_raw = ds[rotate_vars].copy()
        #ds_planeonly is without y-rotation, ds is with all 3 rotations
        #rotation is dictionary with all 3 rotation angles
        ds_planeonly, ds, rotation = execute_planar_fit(uvw_raw, 
                                       averaging_intervall = config["window"], 
                                       plot_path = config["plot_output"])
    # in case no rotation needed anymore
    elif config["rotation"] == False:
        print("Proceed without rotation, only valid if data already rotated!")
        ds = ds
        rotation = None

    #print("NaNs after rotation: ")
    #nan_counter(ds)
    
    # fluxes calculation
    fluxes, fluctuations = fluxes_calculation(ds, config)
    #results["fluxes"] = fluxes
    #results["fluxes_orig"] = fluxes_orig
    
    output_fluctuations = config.get("output_fluctuations", None)
    if output_fluctuations:
        data_vars = ["u", "v", "w", "tc"]
        units = ["m s^-1", "m s^-1", "m s^-1", "°C"]
        if "q" in ds.data_vars:
            data_vars = ["u", "v", "w", "tc", "q"]
            units = ["m s^-1", "m s^-1", "m s^-1", "°C", "kg*kg^-1"]
        savepath = Path(output_fluctuations) / f"fluctuations_{config['window']}.png"
        plot_timeseries_multiple(
                fluctuations, data_vars, units, figsize = (12, 6), colors = [None], 
                labels = [None], savepath = savepath
        )
        #compare to total observation
        savepath_obs = Path(output_fluctuations) / f"totobs_{config['window']}.png"
        plot_timeseries_multiple(
                ds, data_vars, units, figsize = (12, 6), colors = [None], 
                labels = [None], savepath = savepath_obs
        )

    # stationarity
    stat = stationarity(ds, config)
    # merge
    statistics_list = [fluxes, rotation, stat, nan_perc]
    #only merge if not None
    statistics = xr.merge([ds for ds in statistics_list if ds is not None])

    # spectra
    if config["spectra"]:
        spectra, epsilon, slopes = spectra_eps(ds, config, fluxes.meanU)
        statistics = xr.merge([statistics, epsilon, slopes])
        results["spectra"] = spectra

    # structure functions
    if config["strfun"]:
        strfun, epsilon = structure_functions_epsilon(ds, config, fluxes.meanU)
        statistics = xr.merge([statistics, epsilon])
        results["strfun"] = strfun

    # autocorrelation
    if config["autocorr"]:
        autocorr, intlen = autocorrelation(ds, config, fluxes.meanU)
        statistics = xr.merge([statistics, intlen])
        results["autocorr"] = autocorr

    # MRDs
    if config["MRD"]:
        mrd = multiresolution(ds, config)
        results["MRD"] = mrd

    # put to nan the zeros in empty data
    statistics = statistics.where(statistics.meanU > 0)
    fluxes = fluxes.where(fluxes.meanU > 0)
    
    #flux corrections
    if any(v in ds.data_vars for v in ["q", "rho_v", "wv"]): 
        print("Do flux corrections for H and LE...")
        fluxes = correct_fluxes(fluxes)
    
    results["fluxes"] = fluxes

    # create results dictionary
    results["stats"] = statistics
   
    return results

#results = main_postprocess(ds_test, config)
#%% call for all data and every averaging intervall
folder = Path(r"D:\HEFEXIII\Tower\L3\no_sectorwise")

windows = ["30min"]
#windows = ["2h", "30min", "5min", "1min"]

for w in windows:
    os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\turbulence_postprocessing")
    c_file = f"config_{w}_smartflux.txt"
    #c_file = f"config_{w}_smartflux_simpledetrend.txt"
    with open(c_file, "r") as f:
        config = json.load(f)
        
    # --- old...Aubinet says 30min planar fit in general ---
    #d_file_20 = fr"metek_L3_20Hz_{w}.nc"
    #d_file_30 = fr"metek_L3_30Hz_{w}.nc"
    #d_file_smart = fr"smartflux_L3_20Hz_{w}.nc"
    # ---
    #load 30 min planar fit files
    d_file_20 = fr"metek_L3_20Hz.nc"
    d_file_smart = fr"metek_L3_20Hz_smartflux.nc"
    
    #ds_20 = xr.open_dataset(folder / d_file_20)
    #ds_30 = xr.open_dataset(folder / d_file_30)
    #ds_smart = xr.open_dataset(folder / d_file_smart)
    
    #ds_20["time"] = ds_20.time.dt.round("1ms")
    #ds_30["time"] = ds_30.time.dt.round("1ms")
    #ds_smart["time"] = ds_smart.time.dt.round("1ms")
    
    #rename q_calc to q if only q_calc existing
    #if "q_calc" in ds_smart.data_vars and "q" not in ds_smart.data_vars:
    #    ds_smart = ds_smart.rename({"q_calc": "q"})
    
    #loop over datasets
    #for ds, freq in zip([ds_20, ds_30], ["20Hz", "30Hz"]):
    #for ds, freq in zip([ds_20, ds_smart], ["20Hz", "20Hz"]):
    for file, freq in zip([d_file_20], ["20Hz"]):
    
        #read data
        ds = xr.open_dataset(folder / file)
        
        #round time
        ds["time"] = ds.time.dt.round("1ms")
        
        #rename q_calc to q if only q_calc existing
        if "q_calc" in ds.data_vars and "q" not in ds.data_vars:
            ds = ds.rename({"q_calc": "q"})
            
        #---- test time only ----
        #ds = ds.sel(time = slice("2025-08-15 11:00", "2025-08-15 12:00"))
        #ds = ds.sel(time = "2025-08-15")
    
        #empty storage lists
        spectra_list = []
        fluxes_list = []
        stationarity_list = []
        
        #loop over days, otherwise crash
        for label, ds_day in ds.resample(time="1D", label = "right", closed = "right"):
            
            #get rid of days where total data is less than the chosen window
            duration = (ds_day.time.max() - ds_day.time.min()).values
            if duration < np.timedelta64(pd.to_timedelta(config["window"]), "ns"):
                print(f"Skipping {label}...less data when avg. window available")
                continue

            results = main_postprocess(ds_day, config)
            
            #save data
            fluxes = results["stats"]
            spectra = results["spectra"]
            
            #to list
            fluxes_list.append(fluxes)
            spectra_list.append(spectra)
            
        #free some storage, ve variables first
        data_vars = ds.data_vars
        del ds
            
        ds_fluxes = xr.concat(fluxes_list, dim = "time")
        ds_spectra = xr.concat(spectra_list, dim = "time")
        
        #smartflux_path = Path(fr"D:\HEFEXIII\Tower\L3\no_sectorwise\smartflux_L3_20Hz_{w}.nc")
        smartflux_path = Path(fr"D:\HEFEXIII\Tower\L3\no_sectorwise\metek_L3_20Hz_smartflux.nc")
        #if Path.exists(smartflux_path):
        #    ds_fluxes["H"] = add_dynamic_sensible_heat(ds_fluxes["wT"], 
        #                                               smartflux_path, window=w)
            
        #do SND and WPL correction if neccessary moisture variables are avilabel
        if ("wq" in ds_fluxes.data_vars) and ("wrho_v" in ds_fluxes.data_vars):
            
            print("applying SND and WPL correction")
            ds_fluxes = correct_fluxes(ds_fluxes)
            
        else:
            if Path.exists(smartflux_path):
                print("no moisture variables found...",
                      "dynamic sensible heat calculated with 1m density for all levels")
                ds_fluxes["H"] = add_dynamic_sensible_heat(ds_fluxes["wT"], 
                                                           smartflux_path, 
                                                           window=w)
            
            
        
        savepath = Path(r"D:\HEFEXIII\Tower\turbulence_processed")

        filename_fluxes = f"fluxes_{w}_gauss.nc"
        filename_spectra = f"spectra_{w}_gauss.nc"
        filename_stationarity = f"stationarity_{w}_gauss.nc"
        if "q" in data_vars:
            filename_fluxes = f"fluxes_{w}_smart_gauss.nc"
            filename_spectra = f"spectra_{w}_smart_gauss.nc"
            filename_stationarity = f"stationarity_{w}_smart_gauss.nc"

        ds_fluxes.to_netcdf(savepath / fr"{w}_avg" / filename_fluxes)
        ds_spectra.to_netcdf(savepath / fr"{w}_avg" / filename_spectra)

        