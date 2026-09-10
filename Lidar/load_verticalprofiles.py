# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:41:36 2026

@author: leopo
"""

#import packages
import xarray as xr
import pandas as pd
import re

#%% uav
def load_uav_data(
        folder_uav
):
    # storage dict and list
    datasets_uav = {}
    subfolders = []
    
    # loop over subfolder
    for subfolder in folder_uav.iterdir():
        if not subfolder.is_dir():
            continue
    
        print(f"Processing folder: {subfolder.name}")
    
        dataarrays = []
        subfolders.append(subfolder.name)
    
        # loop over csv files
        for csv_file in subfolder.glob("*.csv"):
    
            # read csv
            df = pd.read_csv(csv_file,sep=",",na_values="NA")
    
            # height as index
            df = df.set_index("alt")
    
            # time from column names
            df.columns = pd.to_datetime(df.columns, utc = True).tz_convert(None)
    
            # In long-format
            df_long = df.stack(dropna=False).reset_index()
            df_long.columns = ["alt", "time", "value"]
    
            # In xarray DataArray 
            da = df_long.set_index(["time", "alt"]).to_xarray()["value"]
    
            # variable names
            stem = csv_file.stem
            if "_imet_" in stem:
                var_part = stem.split("_imet_")[1]
            elif "_up_" in stem:
                var_part = stem.split("_up_")[1]
            else:
                var_part = stem
                
            varname = var_part.rsplit("_", 1)[0]
            da.name = varname
    
            dataarrays.append(da)
    
        # skip if no data
        if not dataarrays:
            continue
    
        # concat all subfolder data
        ds_uav = xr.Dataset({da.name: da for da in dataarrays})
    
        # naming
        #ds_uav = ds_uav.assign_coords(variable=[da.name for da in dataarrays])
    
        # to xr.Dataset
        #ds_uav = ds_uav.to_dataset(dim="variable")
    
        # save datastes in dict
        datasets_uav[subfolder.name] = ds_uav
        
        return datasets_uav
    
    
#%% read wind ranger and streamline data
#windranger
def load_windranger_data(
        folder_windranger, averaged_data = False
):
    
    #sorting function
    def preprocess(ds):
        return ds.sortby("time").drop_duplicates("time")

    #file pattern
    if averaged_data:
        pattern = re.compile(r"\d{8}_\d{2}0000_averaged.nc")
    else:
        pattern = re.compile(r"\d{8}_\d{2}0000\.nc$")
    folder_pattern = re.compile(r"^\d{8}$")
    subfolders = sorted([f for f in folder_windranger.iterdir() if 
                         f.is_dir() and folder_pattern.match(f.name)])
    #subfolders = [folder_windranger / f"202508{day}" for day in ["07", "08", "09", "10", "18", "19"]]
    
    #store data
    datasets_windranger = {}
    
    for folder in subfolders:
        print(f"Processing folder: {folder.name}")
        
        files = [f for f in folder.glob("*.nc") if pattern.search(f.name)]
        files = sorted(files)
        
        
        if files:
            '''
            ds_wind = xr.open_mfdataset(
                files,
                combine="nested",
                concat_dim = "time",
                preprocess = preprocess,
                engine="h5netcdf",
                parallel = False, chunks = None,
                data_vars="all"
            )
            datasets = [xr.open_dataset(fil)]
            datasets_windranger[folder.name] = ds_wind
            '''
            datasets = [xr.open_dataset(f) for f in files]
            ds_wind = xr.concat(datasets, dim = "time")
            datasets_windranger[folder.name] = ds_wind
            
    data_list = list(datasets_windranger.values())
    ds_windranger = xr.concat(data_list, dim = "time")
    
    return ds_windranger

#%%streamline
def load_streamline_data(
        folder_streamline, 
        pattern_18m = rf"VAD149_User3_RGL18_SNR0_CNS60v04_202508\d{2}\.nc",
        pattern_36m = rf"VAD149_User3_RGL32_SNR0_CNS60v04_202508\d{2}\.nc"
):
    #days1 = ["07", "08", "09", "10"]
    #days2 =  ["18", "19"]
    #days_pattern1 = "|".join(days1)
    #days_pattern2 = "|".join(days2)
    
    #pattern = re.compile(r"\d{8}\.nc$")
    #pattern = r"\d{2}"
    
    #pattern1 = re.compile(rf"VAD149_User3_RGL18_SNR0_CNS60v04_202508({pattern})\.nc")
    #pattern2 = re.compile(rf"VAD149_User3_RGL32_SNR0_CNS60v04_202508({pattern})\.nc")
    pattern1 = re.compile(pattern_18m)
    pattern2 = re.compile(pattern_36m)
    
    
    files1 = [f for f in folder_streamline.glob("*.nc") if pattern1.search(f.name)]
    files1 = sorted(files1)
    files2 = [f for f in folder_streamline.glob("*.nc") if pattern2.search(f.name)]
    files2 = sorted(files2)
    list_streamline18 = [xr.open_dataset(f) for f in files1]
    ds_streamline18 = xr.concat(list_streamline18, dim = "time")
    list_streamline32 = [xr.open_dataset(f) for f in files2]
    ds_streamline32 = xr.concat(list_streamline32, dim = "time")
    #ds_streamline18 = xr.open_mfdataset(files1, combine = "by_coords", engine = "h5netcdf", 
    #                                  parallel = False, chunks = None, data_vars = "all")
    #ds_streamline32 = xr.open_mfdataset(files2, combine = "by_coords", engine = "h5netcdf", 
    #                                  parallel = False, chunks = None, data_vars = "all")
    return ds_streamline18, ds_streamline32

#%% main function to load the all

def load_vertical_profiles(
        folder_streamline = None, 
        folder_windranger = None,
        folder_uav = None, 
        averaged_data = False,
        file_pattern_streamline18 = rf"VAD149_User3_RGL18_SNR0_CNS60v04_202508\d{2}\.nc",
        file_pattern_streamline36 = rf"VAD149_User3_RGL18_SNR0_CNS60v04_202508\d{2}\.nc"
):
    
    return_dict = {}
    
    #streamline
    if folder_streamline is not None:
        print("---- load streamline data ----")
        ds_streamline18, ds_streamline32 = load_streamline_data(folder_streamline,
                                                    pattern_18m=file_pattern_streamline18,
                                                    pattern_36m=file_pattern_streamline36)
        
        #rename
        if all([var in ds_streamline18.data_vars for var in ["wspeed", "wdir"]]): 
            vars_streamline = {"wspeed":"spd", "wdir":"dir"}
            ds_streamline18 = ds_streamline18.rename(vars_streamline)
            ds_streamline32 = ds_streamline32.rename(vars_streamline)
        
        # total altitude as coordinate
        if "heights" in ds_streamline18.coords:
            ds_streamline18 = ds_streamline18.assign_coords(
                alt = ds_streamline18["heights"] + 2719)
            ds_streamline18 = ds_streamline18.swap_dims({"heights": "alt"})
    
            ds_streamline32 = ds_streamline32.assign_coords(
                alt = ds_streamline32["heights"] + 2719)
            ds_streamline32 = ds_streamline32.swap_dims({"heights": "alt"})
            
        elif "height" in ds_streamline18.coords:
            ds_streamline18 = ds_streamline18.assign_coords(
                alt = ds_streamline18["height"] + 2719)
            ds_streamline18 = ds_streamline18.swap_dims({"height": "alt"})
    
            ds_streamline32 = ds_streamline32.assign_coords(
                alt = ds_streamline32["height"] + 2719)
            ds_streamline32 = ds_streamline32.swap_dims({"height": "alt"})
            
        return_dict["streamline18"] = ds_streamline18
        return_dict["streamline32"] = ds_streamline32
        
    #windranger
    if folder_windranger is not None:
        print("---- load windranger data ----")
        ds_windranger = load_windranger_data(folder_windranger, averaged_data)
        
        # rename
        vars_windranger = {"VEL":"spd", "DIR":"dir", 
                           "U":"u", "V":"v", "W":"w", "alt":"alt_instrument"}
        ds_windranger = ds_windranger.rename(vars_windranger)
        
        #windranger true measurement height
        ds_windranger["MDT"] = ds_windranger["MDT"] + ds_windranger["alt_instrument"].mean()
        ds_windranger = ds_windranger.assign_coords(
            alt = ds_windranger["height"] + ds_windranger["alt_instrument"].mean())
        ds_windranger = ds_windranger.swap_dims({"height":"alt"})
        
        return_dict["windranger"] = ds_windranger
        
    #UAV
    if folder_uav is not None:
        print("---- load uav data ----")
        datasets_uav = load_uav_data(folder_uav)
        
        #rename
        vars_uav = {"wind_speed":"Wspd", "wind_dir":"Wdir", "temp":"Ts", "rh":"RH"}
        for key in datasets_uav.keys():
            datasets_uav[key] = datasets_uav[key].rename(vars_uav)
            
        return_dict["uav"] = datasets_uav
            
    return return_dict







