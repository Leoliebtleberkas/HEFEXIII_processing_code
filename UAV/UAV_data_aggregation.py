# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:32:09 2026

@author: leopo
"""

#import packages
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from metpy.calc import wind_components, wind_speed, wind_direction
from metpy.units import units

#%% load uav data function
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
        
        #add agl as time dependent variable
        ds_uav["agl"] = ds_uav["alt"] - (ds_uav["alt"].min() - 1)
        ds_uav["agl"] = ds_uav["agl"].broadcast_like(ds_uav["temp"])        
       
        # save datastes in dict
        datasets_uav[subfolder.name] = ds_uav
    
    return datasets_uav

    
#%% call function
folder_uav = Path(r"D:\HEFEXIII\UAV\l4")

ds_uav_dict = load_uav_data(folder_uav)

#to one ds
ds_uav = xr.concat(ds_uav_dict.values(), dim = "location", join = "outer").assign_coords(
    location = list(ds_uav_dict.keys()))

#calculate wind sped and wind direction
ds_uav["spd"] = wind_speed(ds_uav["u"]*units("m/s"), ds_uav["v"]*units("m/s"))
ds_uav["dir"] = wind_direction(ds_uav["u"]*units("m/s"), ds_uav["v"]*units("m/s"))

#get rid of units
ds_uav = ds_uav.metpy.dequantify()

#and save 
ds_uav.to_netcdf(folder_uav / r"quadcopter_combined.nc")
#%% not included until now: fixed wing erlangen and augsburg, quadcopter erlangen
folder_erlangen_quad = Path(r"D:\HEFEXIII\UAV\l1\HEFEX IOP1 Q8 Andre")

#read all csv files
df_list = []
ds_list = []
for csv_file in folder_erlangen_quad.glob("*.csv"):

        # read csv
        df = pd.read_csv(csv_file,sep=",",na_values=["NA", "", "NAN"])
        
        #rename
        df = df.rename(columns = {"UTC":"time", "altitude(meters)":"alt",
                                  "wind_speed(m/s)":"spd", "wind_dir(deg)":"dir"})
        #time
        df["time"] = pd.to_datetime(df["time"]).dt.round("h")# - pd.Timedelta(2, unit = "h")
        
        
        #calculate u and v
        
        spd = df["spd"].to_numpy() * units("m/s")
        direction = df["dir"].to_numpy() * units.degrees
        u, v = wind_components(spd, direction)
        
        df["u"] = u
        df["v"] = v
        
        # separate ascend and descend and set a flag for ascend
        df["dalt"] = df["alt"].diff()
        df["ascending"] = df["dalt"] > 0
        ascent = df.loc[df["ascending"]].copy()
        
        #average the heights for mean
        ascent["alt"] = np.round(ascent["alt"], decimals = 0).astype(int)
        
        profile = ascent.groupby("alt").mean(numeric_only=True)
        
        
        #to dataset
        ds = profile.to_xarray()
        t0 = pd.to_datetime(df["time"].iloc[0]).tz_convert(None)
        ds = ds.expand_dims(time = [t0])
        
        df_list.append(df)
        ds_list.append(ds)
        #time from column names
        #df.columns = pd.to_datetime(df.columns, utc = True).tz_convert(None)
        
# attention hardcode: there is one duplicate timestep in the data after rounding --> kick out
del df_list[1]
del ds_list[1]

#%% combine datasets and save
ds_quad = xr.concat(ds_list, dim="time", join = "outer")

#dro duplicates
ds_quad = ds_quad.drop_duplicates(dim = "time", keep = "first")

#make agl a coordinate
agl = ds_quad["height_above_takeoff(meters)"].values
ds_quad = ds_quad.rename({"height_above_takeoff(meters)":"agl"})
#ds_quad = ds_quad.assign_coords(agl=("alt", agl.values[:len(ds_fw.alt)])

#get rid of units
ds_quad = ds_quad.metpy.dequantify()

ds_quad.to_netcdf(folder_uav / r"erlangen_quadcopter.nc")


#%% Erlangen fixed wing data
folder_uav = Path(r"D:\HEFEXIII\UAV\l4")
df_fw = pd.read_csv(folder_uav / r"HEFEXIII_fixedwing-data_FAU.csv",
                    sep = ",", na_values = ["NA", "", "NAN"])

#separate columns --> elevatino info uniform
meta_cols = ["ele", "agl"]
#the others contain differnt times but same variables
value_cols = [c for c in df_fw.columns if c not in meta_cols]

#in long format
df_long = df_fw.melt(id_vars=meta_cols,value_vars=value_cols,
                  var_name="time_var",value_name="value")

#extract time and varibles
df_long[["time_str", "variable"]] = df_long["time_var"].str.rsplit("_", n=1, expand=True)

#parse time
df_long["time"] = pd.to_datetime(df_long["time_str"], format = "%d-%m-%Y_%H%M")

#to UTC
df_long["time"] = (df_long["time"].dt.tz_localize(None) - pd.Timedelta(hours = 2))

#rename ele to alt
df_long = df_long.rename(columns={"ele": "alt"})

#save agl in between
agl = df_long["agl"]

#use pivot to go into dataset
df_pivot = df_long.pivot_table(
    index=["time", "alt"],
    columns="variable",
    values="value",
    aggfunc="mean"
)

ds_fw = df_pivot.to_xarray()

#add agl as coordinate
agl_ar = xr.DataArray(data = agl.values[:len(ds_fw.alt)], 
                      dims = "alt", coords = {"alt": ds_fw["alt"]})
#ds_fw["agl"] = agl.values[:len(ds_fw.alt)]
ds_fw["agl"] = agl_ar.broadcast_like(ds_fw["wspd"])

#%%calculations for fixed wing
from metpy.calc import potential_temperature

theta = potential_temperature(ds_fw["pres"]*units("Pa"), 
                              ds_fw["shttemp"]*units("degC"))
ds_fw["theta"] = theta

u,v = wind_components(ds_fw["wspd"]*units("m/s"), ds_fw["wdir"]*units.deg)

ds_fw["u"] = u
ds_fw["v"] = v

#convert some units
ds_fw["pres"] = ds_fw["pres"] / 100 #to hPa

#get rid of units
ds_fw = ds_fw.metpy.dequantify()

#subset and renaming for comparison to other quadcopter uav
rename_dict = {"shttemp":"temp", "shtrhum":"rh", "shtshum":"qv", 
               "pres":"p", "wspd":"spd", "wdir": "dir"}

ds_fw_sub = ds_fw[list(rename_dict.keys()) + ["u", "v", "agl"]].copy()
ds_fw_sub = ds_fw_sub.rename(rename_dict)


#save 
ds_fw.to_netcdf(folder_uav / r"erlangen_fixedwing.nc")
ds_fw_sub.to_netcdf(folder_uav / r"erlangen_fixedwing_subset.nc")

#%% merge in one UAV
ds_all = ds_uav.copy()

#fixed wing
fw_loc = "F309"
ds_fw_sub = ds_fw_sub.expand_dims(location = [fw_loc])
ds_all = xr.concat([ds_all, ds_fw_sub], dim = "location", join = "outer")

#quadcopter
qc_loc = "Q309"
ds_quad = ds_quad.expand_dims(location = [qc_loc])
#only relevants vars
ds_quad_sub = ds_quad[["spd", "dir"]]
ds_all = xr.concat([ds_all, ds_quad_sub], dim = "location", join = "outer")

#save 
ds_all.to_netcdf(folder_uav / r"UAV_all_locations_raw.nc")
