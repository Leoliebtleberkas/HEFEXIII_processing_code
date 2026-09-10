# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 09:30:25 2026

@author: leopo

The script can be divided in two parts:
1. rotate wind data of meteks and gill in a carthesian coordinate system 
   using tilt sensors and compasses and save that uncleaned. This data will be
   uses for planar fit rotation and turbulence postprocessing, where quality 
   control is done in a more accurate way (after the planar fit rotation)
   
2. apply quality control to copies of the rotated data and save that in high res.,
   10min, 30min and 1h averaged versions for general analysis and comparison to 
   model outputs. In the end of this script, one dataset is created, containing 
   all important data of all sensors, which is
       - Metek u-sonics: u,v,w,Ts
       - Metek u-sonic combined with smartflux: u,v,w,Ts,Tair,p,q,RH,wv,rho_v
       - gill 2-D sonic: u,v
       - Hygrovue: AirTC, RH
   
"""



from pathlib import Path
import numpy as np
import xarray as xr
import os
os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\tower")
from coordinate_rotation_xr_functions import *

# ----1. Coordinate Rotation Wind Data -----
#%%read data
filepath = Path(r"D:\HEFEXIII\Tower\CR3000_l1")
metek_20Hz = r"metek_L1_20Hz_orig_biascorrected.nc"
#metek_30Hz = r"metek_L1_30Hz_orig.nc"
metek_20Hz_smartflux = r"Smartflux_20Hz_l0_red.nc"
gill_1s = r"gill_L1_1s_orig.nc"

metek_20Hz = xr.open_dataset(filepath / metek_20Hz)
metek_20Hz_smartflux = xr.open_dataset(filepath / metek_20Hz_smartflux)

#metek_30Hz = xr.open_dataset(filepath / metek_30Hz)
gill_1s = xr.open_dataset(filepath / gill_1s)

#%% 30 Hz data get Pitch and Roll from 0.5 m metek
metek_1m_interp = metek_20Hz[["Pitch", "Roll"]].sel(heights = 0.5).interp(
    time = metek_20Hz_smartflux.time)
metek_20Hz_smartflux[["Pitch", "Roll"]] = (
    metek_1m_interp
    .expand_dims({"heights": metek_20Hz_smartflux.heights.values})
)

#%% check pitch and roll angles
#10° threshold
for h in metek_20Hz.heights:
        #ds = metek_20Hz.sel(heights = h)
        false_ratio(metek_20Hz, h)
        
#for h in metek_30Hz.heights:
#        ds = metek_30Hz.sel(heights = h)
#        false_ratio(metek_30Hz, h)
    
#5° threshold
for h in metek_20Hz.heights:
        #ds = metek_20Hz.sel(heights = h)
        false_ratio(metek_20Hz, h, threshold_pitch=5, threshold_roll=5)

#%% correct pitch and roll
metek_20Hz = correct_roll_pitch(metek_20Hz)
metek_20Hz_smartflux = correct_roll_pitch(metek_20Hz_smartflux)
#%%gill compass cleaning and mean calculation
comp_mean = clean_compass_return_mean(gill_1s)

#add mean compass to metek datasets
compass_20Hz = comp_mean.interp(time = metek_20Hz.time)
#compass_30Hz = comp_mean.interp(time = metek_30Hz.time)
compass_20Hz_smartflux = comp_mean.interp(time = metek_20Hz_smartflux.time)


metek_20Hz["Compass"] = compass_20Hz
metek_20Hz_smartflux["Compass"] = compass_20Hz_smartflux
#metek_30Hz["Compass"] = compass_30Hz

#%% ROTATION AFTER AUBINET 2012
# ORDER MATTERS BECAUSE MATRICE MULTIPLICATION IS NOT KOMMUTATIVE!!!!
# FOR ROTATION FROM SONIC TO REFERENCE COORDINATE SYSTEM: 
    #1. Rotation around z
    #2. Rotation around y
    #3. Rotation around x    
 
#20Hz
metek_20Hz["u"], metek_20Hz["v"] = correct_zaxis(metek_20Hz["u"], metek_20Hz["v"], 
                                                 metek_20Hz["Compass"])
metek_20Hz["u"], metek_20Hz["w"] = correct_yaxis(metek_20Hz["u"], metek_20Hz["w"], 
                                                 metek_20Hz["Roll"])
metek_20Hz["v"], metek_20Hz["w"] = correct_xaxis(metek_20Hz["v"], metek_20Hz["w"], 
                                                 metek_20Hz["Pitch"])

#30Hz
#metek_30Hz["u"], metek_30Hz["v"] = correct_zaxis(metek_30Hz["u"], metek_30Hz["v"], 
#                                                 metek_30Hz["Compass"])
#metek_30Hz["u"], metek_30Hz["w"] = correct_yaxis(metek_30Hz["u"], metek_30Hz["w"], 
#                                                 metek_30Hz["Roll"])
#metek_30Hz["v"], metek_30Hz["w"] = correct_xaxis(metek_30Hz["v"], metek_30Hz["w"], 
#                                                 metek_30Hz["Pitch"])      

#%%20Hz smartflux data
#20Hz
metek_20Hz_smartflux["u"], metek_20Hz_smartflux["v"] = correct_zaxis(
                            metek_20Hz_smartflux["u"], metek_20Hz_smartflux["v"], 
                                                 metek_20Hz_smartflux["Compass"])
metek_20Hz_smartflux["u"], metek_20Hz_smartflux["w"] = correct_yaxis(
                            metek_20Hz_smartflux["u"], metek_20Hz_smartflux["w"], 
                                                 metek_20Hz_smartflux["Roll"])
metek_20Hz_smartflux["v"], metek_20Hz_smartflux["w"] = correct_xaxis(
                            metek_20Hz_smartflux["v"], metek_20Hz_smartflux["w"], 
                                                 metek_20Hz_smartflux["Pitch"])                                              

#%% Metek calculate raw winddirection
metek_20Hz = calc_wdir(metek_20Hz)
metek_20Hz = calc_wspeed(metek_20Hz)
#metek_30Hz = calc_wdir(metek_30Hz)
#metek_30Hz = calc_wspeed(metek_30Hz)
metek_20Hz_smartflux = calc_wdir(metek_20Hz_smartflux)
metek_20Hz_smartflux = calc_wspeed(metek_20Hz_smartflux)


#%% import data (remove later)
#folder = Path(r"D:\HEFEXIII\Tower\L2")

#metek_20Hz_cleaned = xr.open_dataset(folder / r"metek_L2_20Hz_cleaned.nc")
#smartflux_20Hz_cleaned = xr.open_dataset(folder / r"smartflux_L2_20Hz_cleaned.nc")

#%% ----- 2. Quality Control and data saving
# ATTENTION, this code block needs some time when run with 1 month of 20 Hz data at 5 levels (~1 hour)

os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\helper_functions")
from quality_control import *

#threshold control
#metek_20Hz_cleaned = threshold_correction(metek_20Hz.copy())
smartflux_20Hz_cleaned = threshold_correction(metek_20Hz_smartflux.copy())

#despiking
#metek_20Hz_cleaned, nan_counter = despiking(metek_20Hz_cleaned)
smartflux_20Hz_cleaned, nan_counter_smart = despiking(smartflux_20Hz_cleaned)

#%% The threshold control and despiking fails in case of the humidity variables
# from the smartflux (some periods with a lot of error values --> std despiking fails
# therefore, additional cleaning based on closest hygrovue RH 
# We exploit that the smartflux was closer to the ice and should always be more humid
from metpy.calc import (mixing_ratio, specific_humidity_from_mixing_ratio, 
                        relative_humidity_from_specific_humidity)
Rv = 461.5   #J/(kg*K)
Rd = 287.15  #J/(kg*K)

smartflux_20Hz_cleaned["e"] = (smartflux_20Hz_cleaned["rho_v"] * 0.001 * Rv * 
                              (smartflux_20Hz_cleaned["Tair"]+273.15))
#water vapour mixing ratio
smartflux_20Hz_cleaned["wv"] = mixing_ratio(smartflux_20Hz_cleaned["e"]*units.Pa, 
                                            smartflux_20Hz_cleaned["p"]*units.hPa)
#specific humidity
smartflux_20Hz_cleaned["q"] = specific_humidity_from_mixing_ratio(
                              smartflux_20Hz_cleaned["wv"]*units("kg/kg"))
#relative humidity
smartflux_20Hz_cleaned["RH"] = relative_humidity_from_specific_humidity( 
                   smartflux_20Hz_cleaned["p"]*units("hPa"), 
                   smartflux_20Hz_cleaned["Tair"]*units("degC"),
                   smartflux_20Hz_cleaned["q"]) * 100
#dry air density
numerator = (smartflux_20Hz_cleaned["p"]*100 - (smartflux_20Hz_cleaned["e"]))
denominator = (Rd * (smartflux_20Hz_cleaned["Tair"]+273.15))
smartflux_20Hz_cleaned["rho_d"] =  numerator/denominator  

# dequantify
smartflux_20Hz_cleaned = smartflux_20Hz_cleaned.metpy.dequantify()

# drop the unneccessary variables
smartflux_20Hz_cleaned = smartflux_20Hz_cleaned.drop_vars(["e", "wv"])

#%% additional cleaning of humidity variables with hygrovue data because filters fail
# in case of larger data gaps or long periods with wrong values
filepath = Path(r"D:\HEFEXIII\Tower\CR3000_l1")
file_hyg = r"hygrovue_L1_5s_orig.nc"

ds_hyg_5s = xr.open_dataset(filepath / file_hyg)

#add to smartflux
smartflux_20Hz_cleaned["RH_hyg"] = ds_hyg_5s["RH"].sel(heights = 2).reindex(
                                   time = smartflux_20Hz_cleaned.time, method = "ffill" )

#define a cleaning mask --> set all obs to nan, where
smartflux_20Hz_cleaned["RH_std"] = 4*smartflux_20Hz_cleaned["RH"].rolling(
                                   time = 100, center = True).std()

mask = (smartflux_20Hz_cleaned["RH_hyg"] -
        smartflux_20Hz_cleaned["RH"]) <= smartflux_20Hz_cleaned["RH_std"]

corr_vars = ["RH", "q", "rho_v"]
smartflux_20Hz_cleaned[corr_vars] = smartflux_20Hz_cleaned[corr_vars].where(mask)

#remove RH_hyg and std again
smartflux_20Hz_cleaned = smartflux_20Hz_cleaned.drop_vars(["RH_hyg", "RH_std"])
#%%test plots
for h in metek_20Hz.heights:
    plot_wind_distribution(metek_20Hz.sel(heights = h, method = "nearest"), 
                      label = str(h.values).replace(".", ""), 
                      title = "Direction Distribution all_new")
#%% save L2 dataset
datapath = Path("D:\HEFEXIII\Tower\L2")
filename = "metek_L2_20Hz_orig.nc"
filename_smart = "smartflux_L2_20Hz_orig.nc"

#rotation only
#metek_20Hz.to_netcdf(datapath / filename)
#metek_20Hz_smartflux.to_netcdf(datapath / filename_smart)

#rotated and cleaned data
#metek_20Hz_cleaned.to_netcdf(datapath / r"metek_L2_20Hz_cleaned.nc")
smartflux_20Hz_cleaned.to_netcdf(datapath / r"smartflux_L2_20Hz_cleaned.nc")


#filename_30hz = "metek_L2_30Hz_orig.nc"
#metek_30Hz.to_netcdf(datapath / filename_30hz)


#%% 10 min resample  
#needed variables meteks and campbell loggers
cr_cols_copy = ["u", "v", "w", "Ts", "Roll", "Pitch", "Compass"]
#needed variables metek/smartflux
sf_cols_copy = ["u", "v", "w", "Ts", "p", "wv", "q", "RH", "rho_v", "Tair", "Roll", "Pitch", "Compass"]

#resample metek - cr3000
metek_20Hz_10min = metek_20Hz[cr_cols_copy].resample(time = "10min", 
                                                    closed = "right", 
                                                    label = "right").mean(dim = "time").copy()
#delete original, uncleaned dataset to save some storage
del metek_20Hz

#resample metek smartflux
metek_20Hz_smartflux_10min = metek_20Hz_smartflux[sf_cols_copy].resample(
    time = "10min", closed = "right", label = "right").mean(dim = "time").copy()
#delete original, uncleaned dataset to save some storage
del metek_20Hz_smartflux

#%%--- CLEANED data

#CAN BE REMOVED:load data for 1min resample
folder = Path(r"D:\HEFEXIII\Tower\L2")
metek_20Hz_cleaned = xr.open_dataset(folder / r"metek_L2_20Hz_cleaned.nc")
smartflux_20Hz_cleaned = xr.open_dataset(folder / r"smartflux_L2_20Hz_cleaned.nc")

#%%

cr_cols_copy = ["u", "v", "w", "Ts"]
#needed variables metek/smartflux
sf_cols_copy = ["u", "v", "w", "Ts", "p", "q", "RH", "rho_v", "Tair"]
#resample metek - cr3000
metek_20Hz_cleaned_10min = metek_20Hz_cleaned[cr_cols_copy].resample(time = "10min", 
                                                    closed = "right", 
                                                    label = "right").mean(dim = "time").copy()
#resample metek smartflux
smartflux_20Hz_cleaned_10min = smartflux_20Hz_cleaned[sf_cols_copy].resample(
    time = "10min", closed = "right", label = "right").mean(dim = "time").copy()


#%%calc mean wind direction and speed
#metek - campbell logger
metek_20Hz_10min = calc_wdir(metek_20Hz_10min)
metek_20Hz_10min = calc_wspeed(metek_20Hz_10min)

#metek - smartflux
metek_20Hz_smartflux_10min = calc_wdir(metek_20Hz_smartflux_10min)
metek_20Hz_smartflux_10min = calc_wspeed(metek_20Hz_smartflux_10min)
#metek_30Hz_10min = calc_wdir(metek_30Hz_10min)
#metek_30Hz_10min = calc_wspeed(metek_30Hz_10min)

#%% ---- CLEANED data
#metek - campbell logger
metek_20Hz_cleaned_10min = calc_wdir(metek_20Hz_cleaned_10min)
metek_20Hz_cleaned_10min = calc_wspeed(metek_20Hz_cleaned_10min)

#metek - smartflux
smartflux_20Hz_cleaned_10min = calc_wdir(smartflux_20Hz_cleaned_10min)
smartflux_20Hz_cleaned_10min = calc_wspeed(smartflux_20Hz_cleaned_10min)


#%%save metek 10min
datapath = Path("D:\HEFEXIII\Tower\L2")
filename_cr3000 = "metek_L2_20Hz_10min_res.nc"
filename_smart = "metek_L2_30Hz_10min_res.nc"
filename_smart_20Hz = "smartflux_L2_20Hz_10min_res.nc"
#cr3000
metek_20Hz_10min.to_netcdf(datapath / filename_cr3000)
#smartflux
#metek_30Hz_10min.to_netcdf(datapath / filename_smart)
#smartflux 20Hz
metek_20Hz_smartflux_10min.to_netcdf(datapath / filename_smart_20Hz)


#%% ----cleaned data
datapath = Path("D:\HEFEXIII\Tower\L2")
metek_20Hz_cleaned_10min.to_netcdf(datapath / r"metek_L2_20Hz_10min_res_cleaned.nc")
smartflux_20Hz_cleaned_10min.to_netcdf(datapath / r"smartflux_L2_20Hz_10min_res_cleaned.nc")


#%% ----- GILL CORRECTIONS -----
#no vertical component, only correct u and v with x- and y-tilt
#-->assumption that w_mean = 0

#fill compass gaps
gill_1s["Compass"] = gill_1s["Compass"].ffill(dim = "time")
gill_1s["Compass"] = gill_1s["Compass"].bfill(dim = "time")


#add w = 0 to dataframes 
gill_1s["w"] = xr.DataArray(dims = gill_1s.dims, 
                            data = np.zeros(gill_1s["u"].shape))
    
gill_1s["u"], gill_1s["w"] = correct_yaxis(gill_1s["u"], gill_1s["w"], gill_1s["Ytilt"])
gill_1s["v"], gill_1s["w"] = correct_xaxis(gill_1s["v"], gill_1s["w"], gill_1s["Xtilt"])
gill_1s["u"], gill_1s["v"] = correct_zaxis(gill_1s["u"], gill_1s["v"], gill_1s["Compass"])

#wspd
gill_1s["Wspd"] = wind_speed(gill_1s["u"]*units("m/s"),
                            gill_1s["v"]*units("m/s"))

#wdir
gill_1s["Wdir"] = wind_direction(gill_1s["u"]*units("m/s"),
                                 gill_1s["v"]*units("m/s"))

#drop unneccessary cols
gill_1s = gill_1s.drop_vars("w")
    
#%%resample to 10 min
gil_cols = ["u", "v", "Compass", "Xtilt", "Ytilt", "Zorient", "Status"]
gill_1s_10min = gill_1s[gil_cols].resample(time = "10min", closed = "right", label = "right").mean().copy()

    
#wspd
gill_1s_10min["Wspd"] = wind_speed(gill_1s_10min["u"]*units("m/s"),
                                   gill_1s_10min["v"]*units("m/s"))

#wdir
gill_1s_10min["Wdir"] = wind_direction(gill_1s_10min["u"]*units("m/s"),
                                       gill_1s_10min["v"]*units("m/s"))
    
    
#%%save L2 Gill datasets
datapath = Path("D:\HEFEXIII\Tower\L2")
filename_1s = "gill_L2_1s_orig.nc"
filename_10min = "gill_L2_1s_10min_res.nc"
#save
#gill_1s.to_netcdf(datapath / filename_1s)
gill_1s_10min.to_netcdf(datapath / filename_10min)


#%%load hygrovues and create one large dataset with 10min data of 
#wind, temperature and RH
filepath = Path(r"D:\HEFEXIII\Tower\CR3000_l1")
file_hyg = r"hygrovue_L1_5s_10min_res.nc"
#file_hyg = r"hygrovue_L1_5s_orig.nc"

ds_hyg = xr.open_dataset(filepath / file_hyg)
#ds_hyg = ds_hyg.resample(time = "1min", closed = "right", label = "right").mean().copy()

#common time axis
all_times = xr.concat([
        metek_20Hz_cleaned_10min.time, smartflux_20Hz_cleaned_10min.time,
        gill_1s_10min.time,ds_hyg.time,],
    dim="time").drop_duplicates(dim = "time").sortby("time")

#reindex to that
metek = metek_20Hz_cleaned_10min.reindex(time=all_times)
smartflux = smartflux_20Hz_cleaned_10min.reindex(time=all_times)
gill = gill_1s_10min.reindex(time=all_times)
hyg = ds_hyg.reindex(time=all_times)

#merge
ds_all = xr.merge([metek, smartflux, gill, hyg],join="outer")

#delete unneccessary
ds_all = ds_all.drop_vars(["Compass", "Xtilt", "Ytilt", "Zorient", "Status"])

#%%resample to 30 min and 1h
ds_all_30min = ds_all.resample(time = "30min", closed = "right",
                               label = "right").mean()
ds_all_1h = ds_all.resample(time = "1h", closed = "right",
                            label = "right").mean()

#%%calc wind speed and dir again
ds_all_30min["Wspd"] =  wind_speed(ds_all_30min["u"]*units("m/s"),
                           ds_all_30min["v"]*units("m/s"))
ds_all_30min["Wdir"] = wind_direction(ds_all_30min["u"]*units("m/s"),
                              ds_all_30min["v"]*units("m/s"))
ds_all_1h["Wspd"] =  wind_speed(ds_all_1h["u"]*units("m/s"),
                           ds_all_1h["v"]*units("m/s"))
ds_all_1h["Wdir"] = wind_direction(ds_all_1h["u"]*units("m/s"),
                           ds_all_1h["v"]*units("m/s"))

#%%save
folder = Path(r"D:\HEFEXIII\Tower\L2")

#10min
ds_all.to_netcdf(folder / "tower_10min.nc")
#30min
ds_all_30min.to_netcdf(folder / "tower_30min.nc")
#1h
ds_all_1h.to_netcdf(folder / "tower_1h.nc")
