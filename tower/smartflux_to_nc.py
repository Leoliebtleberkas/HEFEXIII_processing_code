# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 11:54:22 2026

@author: leopo
"""

#packages
import os
import zipfile
import pandas as pd
import re
from pathlib import Path 
import xarray as xr


#%% ----- smartflux 20 Hz data shaping -----
#%% one testfile
folder = Path(r"D:\HEFEXIII\Tower\Licor\raw")
testfile = r"2025-08-10T023000_smart3-00360.ghg" 
with zipfile.ZipFile(folder / testfile, 'r') as z:
    # Suche nach der .data Datei
    data_files = [f for f in z.namelist() if f.endswith(".data")]
    if not data_files:
        raise ValueError("Keine .data Datei im Archiv gefunden")

    # Öffnen als Textstream
    with z.open(data_files[0]) as f:
        lines = f.read().decode('utf-8', errors='ignore').splitlines()

# ---- Header (DATAH) finden ----
header_line = next(l for l in lines if l.startswith("DATAH"))
columns = header_line.strip().split("\t")[1:]  # erstes Feld = DATAH

# ---- DATA-Zeilen extrahieren ----
data_lines = [l.strip().split("\t")[1:] for l in lines if l.startswith("DATA")]
#kick out first line if its only the columnnames
if data_lines[0] == columns:
    data_lines = data_lines[1:]

# ---- DataFrame bauen ----
df = pd.DataFrame(data_lines, columns=columns)

# ---- Numerische Spalten konvertieren ----
df = df.apply(pd.to_numeric, errors="ignore")

# Optional: Zeitindex aus Seconds + Nanoseconds
df["timestamp"] = pd.to_datetime(df["Seconds"], unit='s') + pd.to_timedelta(df["Nanoseconds"], unit='ns')
df = df.set_index("timestamp")

#get rid off empty columns
df = df.loc[:, df.columns != '---']

print(df.head())

#%% multifile

folder = Path(r"D:\HEFEXIII\Tower\Licor\raw")

dfs = []

for ghg_file in sorted(folder.glob("*.ghg")):
    print(f"Lese: {ghg_file.name}")
    
    try:
        with zipfile.ZipFile(ghg_file, 'r') as z:
            data_files = [f for f in z.namelist() if f.endswith(".data")]
            if not data_files:
                print("  Keine .data Datei gefunden")
                continue

            with z.open(data_files[0]) as f:
                lines = f.read().decode('utf-8', errors='ignore').splitlines()

        # ---- Header finden ----
        header_line = next(l for l in lines if l.startswith("DATAH"))
        columns = header_line.strip().split("\t")[1:]

        # ---- DATA-Zeilen extrahieren ----
        data_lines = [l.strip().split("\t")[1:] 
                      for l in lines if l.startswith("DATA")]

        # doppelte Header-Zeile entfernen
        if data_lines and data_lines[0] == columns:
            data_lines = data_lines[1:]

        if not data_lines:
            continue

        # ---- DataFrame bauen ----
        df = pd.DataFrame(data_lines, columns=columns)
        df = df.apply(pd.to_numeric, errors="coerce")

        # Zeitindex
        df["timestamp"] = (
            pd.to_datetime(df["Seconds"], unit='s') +
            pd.to_timedelta(df["Nanoseconds"], unit='ns')
        )
        df = df.set_index("timestamp")
        
        # only needed variables --> uncomment if you want to have all!
        # ------
        vars_needed = ["Pressure (kPa)", "Temperature (C)", "Dew Point (C)",
                       "H2O (mmol/mol)", "H2O (g/m^3)", "H2O (mmol/m^3)",
                       "U (m/s)", "V (m/s)", "W (m/s)", "T (C)"]
        # ------
        
        df = df[vars_needed]

        # leere --- Spalten entfernen
        df = df.loc[:, df.columns != '---']

        dfs.append(df)

    except Exception as e:
        print(f"  Fehler in {ghg_file.name}: {e}")

#%% ---- Zusammenführen ----
if dfs:
    combined_df = pd.concat(dfs, ignore_index = False)
    
    # chronologisch sortieren
    #combined_df = combined_df.sort_index()
    
    print("Fertig.")
    print("Gesamtgröße:", combined_df.shape)
    print(combined_df.head())
else:
    print("Keine gültigen Dateien gefunden.")

#%% get mean period
dt = dfs[1180].index.to_series().diff().dropna()
print(1 / dt.mean().total_seconds())

#%% to dataset
h = 1

ds = xr.Dataset(combined_df)
ds = ds.assign_coords(heights = 1)
ds = ds.expand_dims(heights = 1)
ds = ds.rename(timestamp = "time")
ds = ds.transpose()

#correct naming
renaming_dict = {"U (m/s)": "u", 
                 "V (m/s)": "v",
                 "W (m/s)": "w", 
                 "T (C)": "Ts",
                 "H2O (mmol/mol)": "H2O (mmol mol^-1)", 
                 "H2O (mmol/m^3)": "H2O (mmol m^-3)",
                 "H2O (g/m^3)": "H2O (g m^-3)"}

ds = ds.rename(renaming_dict)

#save
#savepath = Path(r"D:\HEFEXIII\Tower\Licor")
#ds.to_netcdf(savepath / "Smartflux_20Hz_orig.nc")

#%% unit conversions and calculations
import metpy.calc as mcalc
from metpy.units import units

#constants 
Rd = 287.15 # J/(kg*K)
Rv = 461.5  # J/(kg*K)

ds = ds.rename({"Pressure (kPa)":"Pressure (hPa)"})
#pressure in hPa
ds["Pressure (hPa)"] = ds["Pressure (hPa)"] * 10

#water vapour pressure 
ds["e"] = ds["H2O (g m^-3)"] * 0.001 * Rv * (ds["Temperature (C)"]+273.15) #in Pa!!
#water vapour mixing ratio
ds["wv"] = mcalc.mixing_ratio(ds["e"]*units.Pa, ds["Pressure (hPa)"]*units.hPa)
#specific humidity
ds["q"] = mcalc.specific_humidity_from_mixing_ratio(ds["wv"]*units("kg/kg"))
#dry air density
ds["rho_d"] = ((ds["Pressure (hPa)"]*100 - (ds["e"])) /  #dry air pressure
              (Rd * (ds["Temperature (C)"]+273.15)))        #dry air constant and temp

#ds = ds.drop_vars("Dew Point (C)")

#%%save with additonal variables and changed names

savepath = Path(r"D:\HEFEXIII\Tower\Licor")
ds.to_netcdf(savepath / "Smartflux_20Hz_additionalvariables.nc")

#%%save with variables for post processing/flux calculation only
rename_dict2 = {
               "Temperature (C)": "Tair",
               "H2O (g m^-3)": "rho_v",
               "Pressure (hPa)": "p"
               }

ds_subset = ds.rename(rename_dict2)
ds_subset = ds_subset[["u", "v", "w", "Ts", "Tair", "p", "rho_v"]]

savepath = Path(r"D:\HEFEXIII\Tower\CR3000_l1")
ds_subset.to_netcdf(savepath / "Smartflux_20Hz_l0_red1.nc")




#%% ---- 30min default flux data from smartflux -----

#%%read data
base_path = r"D:\HEFEXIII\Tower\Licor\results"

all_dfs = []

for file in os.listdir(base_path):
    if file.endswith(".zip"):
        zip_path = os.path.join(base_path, file)

        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                
                # nur gewünschte Datei
                if (
                    name.startswith("output/")
                #----- FILENAMES -----
                    #eddypro_exp_full_output FOR FULL OUTPUT FILES
                    #eddypro_exp_fluxnet FOR OTHERS
                    and "eddypro_exp_fluxnet" in name
                    and name.endswith(".csv")
                ):
                    
                    with z.open(name) as f:
                        
                        # --- Header manuell einlesen ---
                        lines = f.read().decode("utf-8").splitlines()
                        
                        if "eddypro_exp_full_output" in name:
                            colnames = lines[1].split(",")   # 2. Zeile
                            units    = lines[2].split(",")   # 3. Zeile
                            data     = lines[3].split(",")   # 4. Zeile
                        elif "eddypro_exp_fluxnet" in name:
                            colnames = lines[0].split(",")   # 1. Zeile
                            data     = lines[1].split(",")   # 2. Zeile
                            units = None
                        df = pd.DataFrame([data], columns=colnames)
                        
                        # optional: Units als Attribut speichern
                        if units:
                            df.attrs["units"] = dict(zip(colnames, units))
                        
                        # optional: Quelle speichern
                        df["source_zip"] = file
                        
                        all_dfs.append(df)

# Alle zusammenführen
combined_df = pd.concat(all_dfs, ignore_index=True)
if "TIMESTAMP_START" in combined_df.columns:
    combined_df["datetime"] = pd.to_datetime(combined_df["TIMESTAMP_START"])
else:
    combined_df["datetime"] = pd.to_datetime(combined_df["date"] + " " + combined_df["time"])
    
combined_df = combined_df.set_index("datetime")
#%%columns to numeric
for c in combined_df.select_dtypes(include=["object"]).columns:
    combined_df[c] = pd.to_numeric(combined_df[c], errors = "coerce")

df_smart = combined_df.dropna(axis = 1, how = "all")

#%% simple plot
from matplotlib import pyplot as plt
plot_vars = ["sonic_temperature", "H", "RH", "LE"]
fig, axs = plt.subplots(nrows = len(plot_vars))
for i,var in enumerate(plot_vars):
    ax = axs[i]
    
    ax.plot(df_smart[var])
    
fig.tight_layout(pad = 0.3)
plt.show()
#%% basic plot
fig, axs = plt.subplots(nrows = 4)
ax0 = axs[0]
ax0.plot(df_smart["air_pressure"] / 100)

ax1 = axs[1]
ax1.plot(df_smart["air_density"])

ax2 = axs[2]
ax2.plot(df_smart["RH"])

ax3 = axs[3]
ax3.plot(df_smart["co2_mixing_ratio"])

fig.tight_layout(pad = 0.8)
plt.show()

#%% to xarray
import xarray as xr
from pathlib import Path

h = 1

ds_smart = xr.Dataset(df_smart)
ds_smart = ds_smart.assign_coords(heights = 1)
ds_smart = ds_smart.expand_dims(heights = 1)
ds_smart = ds_smart.rename(datetime = "time")
ds_smart = ds_smart.transpose()

#correct naming
renaming_dict = {var: var.replace("/", "") for var in ds_smart.data_vars
                 if "/" in var}
ds_smart = ds_smart.rename(renaming_dict)

savepath = Path(r"D:\HEFEXIII\Tower\Licor")
filename = r"Smartflux_fluxnet_30min.nc"

ds_smart.to_netcdf(savepath / filename)
