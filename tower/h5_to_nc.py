# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 16:24:38 2025
convert the old .h5 files to .nc befor the further HEFEXIII data postprocessing
@author: leopo
"""

import pandas as pd
import xarray as xr
from pathlib import Path
#%% as function
def h5_to_netcdf(
        inputpath, outputpath, filename, keys, heights
):
    '''
    

    Parameters
    ----------
    filename : str or Path
        path to the .h5 file
    keys : LIST of str
        contains the group keys of the .h5 file, e.g. "height_05m" etc. 
    heights : LIST of Float
        values of the single heights in m

    Returns
    -------
    None. Only saves the created .nc files in the same directory as the .h5 files

    '''
    
    #data into a xarray
    data_array = []
    for key, h in zip(keys, heights):
    
        df = pd.read_hdf(inputpath / filename, key = key)
        
        if (df.index.name == None) and ("TIMESTAMP" in df.columns):
            df.set_index("TIMESTAMP", inplace = True)
        df.index.name = "time"
        
        #convert numeric columns to float32
        # convert only numeric columns
        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].astype("float32")
    
        #get rid of timezone, otherwise error with concatenate
        if df.index.tz is not None:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        
        ds = xr.Dataset.from_dataframe(df)
        ds = ds.expand_dims(heights = [h])
        
        #add to ds list
        data_array.append(ds)
    
    #concatenate all dataframes
    ds_nc = xr.concat(data_array, dim = "heights", join = "outer")
    ds_nc = ds_nc.assign_coords(heights = ("heights", heights))
    
    nc_file = filename.replace(".h5", ".nc")
    output_path = outputpath / nc_file
    ds_nc.to_netcdf(output_path)
    
    
#%%files
file_20Hz = r'metek_L1_20Hz_orig.h5'
file_20Hz_10min = r'metek_L1_20Hz_10min_res.h5'
file_gill_1s = r"gill_L1_1s_orig.h5"
file_gill_10min = r"gill_L1_1s_10min_res.h5"
file_metek_intercomparison = r'metek_L0_20Hz_SENSOR_COMPARISON.h5'

#datapath
datapath_l2 = Path(r"D:\HEFEXIII\Tower\L2")
datapath_l1 = Path(r"D:\HEFEXIII\Tower\CR3000_l1")
datapath_l0 = Path(r"D:\HEFEXIII\Tower\CR3000_l0")

heights_gill = [2, 4, 7]
keys_gill = [f"height_{h}m" for h in heights_gill]
heights_20Hz = [0.5, 3, 5, 9]
keys_20Hz = ["height_05m", "height_3m", "height_5m", "height_9m"]

heights_intercomparison = [1,3,5,7]
keys_intercomparison = [f"height_{h}m" for h in heights_intercomparison]

#%% for metek intercomparison data (NO FIELD DATA, WAS DONE AFTER THE CAMPAIGN)
h5_to_netcdf(datapath_l0, datapath_l0, file_metek_intercomparison, 
             keys_intercomparison, heights_intercomparison)

#%% call function
h5_to_netcdf(datapath_l1, datapath_l1, file_20Hz, keys_20Hz, heights_20Hz)
#%%
h5_to_netcdf(datapath_l1, datapath_l1, file_20Hz_10min, keys_20Hz, heights_20Hz)
h5_to_netcdf(datapath_l1, datapath_l1, file_gill_1s, keys_gill, heights_gill)
h5_to_netcdf(datapath_l1, datapath_l1, file_gill_10min, keys_gill, heights_gill)
