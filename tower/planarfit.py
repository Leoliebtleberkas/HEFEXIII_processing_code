# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 10:43:21 2026

@author: leopo
"""

#import packages
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from metpy.calc import wind_direction
from metpy.units import units
import os
os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\helper_functions")
#from helper_functions import despiking
from quality_control import threshold_correction, despiking, remove_fragmentary_data
from calculation_functions import (water_vapour_pressure_from_moistdensity,
                                   specific_humidity_from_mixing_ratio,
                                   dry_air_density,
                                   mixing_ratio_from_watervapour)

#%% plot functions for 3d-plane
# Define an event handler function. It takes an event object. This function will 
# be called whenever the event occurs.
def on_move(event):   
    # event ... only argument of the function
    
    # Checking whether the event occurs within the axes (ax) of the plot.
    if event.inaxes == ax:   
        # Updates the viewing angle of the 3D plot (ax) to its current elevation 
        # (elev) and azimuth (azim). This effectively maintains the current viewing 
        # perspective when the plot is being interactively rotated.
        ax.view_init(elev=ax.elev, azim=ax.azim)   
        
        # This line triggers the canvas to redraw itself. It ensures that any 
        # changes made to the plot (such as updating the viewing angle) are reflected 
        # in the displayed plot.
        fig.canvas.draw_idle()


# Function for plotting the the planar fit plane and all the values that 
# contributed to the computation of the plane. It will open a number of  
# interactive figures equal to the number of sonics present at the station  
# under investigation. Once the interactive figures are opened, they can be 
# easily saved as png files.
def plot_pf_plane(ds_unrot_wind, b0, b1, b2, output_path):
    # df_unrot_wind ... pandas data frame containing all the unrotated mean wind components in a sequence of u, v, w
    # b0, b1, b2 ... b coefficients

    # Linear regression plane
    u = ds_unrot_wind["u"].values
    v = ds_unrot_wind["v"].values
    w = ds_unrot_wind["w"].values
    u_plane = np.linspace(u.min(), u.max(), len(u))
    v_plane = np.linspace(v.min(), v.max(), len(v))
    u_plane, v_plane = np.meshgrid(u_plane, v_plane)
    w_plane = b0 + b1 * u_plane + b2 * v_plane
            
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
             
    # Plot the scatter plot
    ax.scatter(u, v, w, c='b', marker='o', label='Data Points', s = 1)
        
    # Plot the regression plane
    ax.plot_surface(u_plane, v_plane, w_plane, alpha=0.4, cmap='viridis', label='Regression Plane')
        
    # Set the initial elevation and azimut angles
    ax.view_init(elev=10, azim=110)
        
    # Add the points to the plane
    for i in range(len(ds_unrot_wind.time)):
        u_point, v_point, w_point = u[i], v[i], w[i]
        ax.plot([u_point, u_point], [v_point, v_point],
                [w_point, b0 + b1 * u_point + b2 * v_point], 
                color='red', linewidth=0.5)
            
    # Get the limits for each axis
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()
        
    # Determine the maximum range among all axes
    max_range = max([
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0]
    ])
        
    # Set the same scale for all three axes
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
        
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
    # This line connects the on_move function to the 'motion_notify_event' event 
    # of the figure canvas (fig.canvas). This means that whenever the mouse is 
    # moved over the plot, the on_move function will be called, allowing for 
    # interactive manipulation of the plot.
    fig.canvas.mpl_connect('motion_notify_event', on_move)
        
    # Add labels with overlines and units
    ax.set_xlabel(r'$\overline{u}$ [m s$^{-1}$]')
    ax.set_ylabel(r'$\overline{v}$ [m s$^{-1}$]')
    ax.set_zlabel(r'$\overline{w}$ [m s$^{-1}$]')
                
    #make directory and save
    #output_path = Path(output_path)
    #output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
        
#%% PLANAR FIT AFTER WILZCAK 2001

#1. DEFINITIONS:
#   - Rotation angle around y = alpha
#   - Rotation angle around x = beta
#   - Rotation angle around z = gamma  

#2. BASIC PRINCIPLE:
#   rotate the 3-D wind vector in a "mean streamline coordinate system" 
#   = into a plane which is parallel to the mean streamlines/sfc parallel
#   u_p = P*(u_m - c) with 
#   - u_p = wind vector in mean streamline coordinate system, not rotated in mean wind
#   - u_m = measured wind vector
#   - P = partial rotation matrix, which rotates u_m so that new z-axis perpendicular to streamlines
#   - c = mean instrument offset error

# ---> vertical wind should average to zero over long period, need to find P for that
#      leads to linear regression equation: 
#      w_m = c3 - p31/p33 * u_m - p32/p33 * v_m = b0 + b1 * u_m + b2 * v_m
# ---> need to find b0,b1,b2 to get p31, p32, p33 which give beta and gamma, e.g.:
#      tan(beta) = -p32/p33, sin(alpha) = p31 (see Wilzcak 2001 for equations) 

#      use alpha, beta to rotate wind data in mean streamline coordinate system
#      for single averaging intervalls: rotate into intervall mean wind
#      ---> alpha = arctan(v_p/u_p)   

#3. REQUIREMENTS:
#   Method only applicable to data when Anemometer is not moved. 
#   ---> Solved that by creating the L2 dataset
#   For determination of b0, b1, b2: 
#   neglect small wind speeds < 1 m/s ---> can lead to unrealistically large pitch


#%%

def planar_fit_coefficients(uvw0):
    """ Function to determine regression coeff_tempicients b for calculating 
        rotation angles using planar fit method (Wilczak et al. 2001)

        INPUT:
        uvw0m ... avg wind components (xarray dataset)
                 u - 1st column, v - 2nd column, w - 3rd column

        OUTPUT:
        bb ... b coefficients from solving set of equations (eq. 48)
    """


    # b coefficients (solve eq. 48)
    if "time" in uvw0.coords:
        nn = len(uvw0.time)
    else:
        nn = 1
    su  = uvw0["u"].sum().values
    sv  = uvw0["v"].sum().values
    sw  = uvw0["w"].sum().values
    suv = (uvw0["u"] * uvw0["v"]).sum().values
    suw = (uvw0["u"] * uvw0["w"]).sum().values
    svw = (uvw0["v"] * uvw0["w"]).sum().values
    su2 = (uvw0["u"]**2).sum().values
    sv2 = (uvw0["v"]**2).sum().values

    lhs = np.array([[nn, su, sv],
                    [su, su2, suv],
                    [sv, suv, sv2]])
    rhs = np.array([sw, suw, svw])
    bb = np.linalg.solve(lhs, rhs)

    return bb


def get_rotation_matrix(
        bb
):
    '''
    Parameters
    ----------
    bb : XARRAY DATASET with dimension "heights"
        the coefficients (from eq.48 Wilzcak et al. 2001)

    Returns
    -------
    P : XARRAY DATASET with dimension "heights"
        The partial rotation matrices for every height.
    alpha : FLOAT
        rotation angle around y.
    beta : FLOAT
        rotation angle around x.
    '''
    
    
    b0, b1, b2 = bb[0], bb[1], bb[2]

    #get coefficients of partial rotation matrix P
    denom = np.sqrt(b1**2 + b2**2 + 1)
    p31 = -1 * b1 / denom
    p32 = -1 * b2 / denom
    p33 = 1 / denom

    #angles
    tanbeta = -1 * p32 / p33
    sinbeta = -1 * p32 / np.sqrt(p32**2 + p33**2)
    cosbeta = p33 / np.sqrt(p32**2 + p33**2)
    sinalpha = p31
    cosalpha = np.sqrt(p32**2 + p33**2)

    #angles in return arrays
    alpha = np.rad2deg(np.arcsin(sinalpha))
    print(f"rotation angle around y, alpha: {alpha}°")
    beta = np.rad2deg(np.arcsin(sinbeta))
    print(f"rotation angle around x, beta: {beta}°")
    
    # p coefficients (eq. 42)
    denom = np.sqrt(bb[1]**2 + bb[2]**2 + 1)
    p31 = -bb[1] / denom
    p32 = -bb[2] / denom
    p33 = 1 / denom
    pp = np.array([p31, p32, p33])

    #rotation matrix C (around x)
    C = np.array([[1,       0,            0],
                  [0, cosbeta, -1 * sinbeta],
                  [0, sinbeta,      cosbeta]])
    #rotation matrix D (around y)
    D = np.array([[     cosalpha, 0, sinalpha],
                  [            0, 1,        0],
                  [-1 * sinalpha, 0, cosalpha]])
    
    #partial rotation matrix P
    P = np.matmul(np.transpose(D), np.transpose(C))
    
    return P, alpha, beta

def get_gamma_rotation_M(
        u_mean, v_mean
):
    
    #rotation angle around z (gamma)
    gamma = np.arctan2(v_mean, u_mean)
    
    #rotation matrix around z 
    if np.isscalar(gamma):
        M = np.zeros((1, 3, 3))
        M[0, 0, 0] = np.cos(gamma)
        M[0, 0, 1] = np.sin(gamma)
        M[0, 1, 0] = -np.sin(gamma)
        M[0, 1, 1] = np.cos(gamma)
        M[0, 2, 2] = 1
        
    else:
        gamma = np.asarray(gamma)
        M = np.zeros((len(gamma), 3, 3))
        M[:, 0, 0] = np.cos(gamma)
        M[:, 0, 1] = np.sin(gamma)
        M[:, 1, 0] = np.sin(gamma) * (-1)
        M[:, 1, 1] = np.cos(gamma)
        M[:, 2, 2] = 1
    
    return gamma, M


#%% main function for planar fit

def planar_fit_main(
        uvw_raw, averaging_intervall, height, output_plot = None
):
    '''
    

    Parameters
    ----------
    uvw_raw : XARRAY DATAARRAY
        contains u,v,w components, dimensions time, heights even if 1-d
    averaging_intervall : STR
        averaging intervall for the rotation
    output_plot : STR / Path
        output path for optional analysis plots. default = None
    Returns
    -------
    None.

    '''
    #store time for later
    times = uvw_raw.time.values
    if averaging_intervall:
        uvw_raw_mean = uvw_raw.copy().resample(time = averaging_intervall).mean()
        edges = uvw_raw_mean.time.values
    else:
        uvw_raw_mean = uvw_raw.copy().mean(dim = "time")
    
    
    #only wind speed > 1 m/s for calculation of coefficients
    spd = np.sqrt(uvw_raw["u"]**2 + uvw_raw["v"]**2 + uvw_raw["w"]**2)
    uvw0 = uvw_raw.where(spd > 1, drop = True).copy()
    #uvw0 = neglect_weak_wind(uvw_raw.copy())

    #intervall mean for b coefficients
    if averaging_intervall:
        uvw0m = uvw0.resample(time = averaging_intervall).mean()
    else:
        #uvw0m = uvw0.mean()
        uvw0m = uvw0
    
    #original dataset to DataArray for matrix calculations
    uvw_raw = uvw_raw.to_array().values
    
    #get planar fit coefficients b0, b1, b2
    bb = planar_fit_coefficients(uvw0m)
    b0, b1, b2 = bb[0], bb[1], bb[2]
    
    #substract systematic offset from mean values directly
    uvw_raw_mean["w"] -= b0

    
    #plot the plane through unrotated data, use a sample of 5000 points
    if output_plot:
        output_plot = output_plot / f"planarfit_{str(height).replace('.', '')}_xr.png"
        
        #select random points
        random_idx = np.random.choice(len(uvw0.time), size=500, replace=False)
        ds_sample = uvw0.isel(time=random_idx)
        plot_pf_plane(ds_sample, 
                      b0, b1, b2, output_plot)
    
            
    #free RAM
    del uvw0, uvw0m
    
    #get partial rotation matrix P (x and y axis rotation)
    P, alpha, beta = get_rotation_matrix(bb)
    
    # substract systematic offset from raw values
    uvw_raw[2] = uvw_raw[2] - b0
    
    #rotation around x and y --> wind data in streamline plane
    uvw2 = np.matmul(P, uvw_raw)

# ----- first step done -----

    #rotation angle gamma
    #2 APPROACHES FOR GLACIER SPACE: 
        
    #1. AVERAGING OVER SINGLE AVERAGING INTERVALLS (e.g. 30min)
    #   ALWAYS ROTATE INTO THE MEAN WIND OF THE AVERAGING INTERVALL
    if averaging_intervall:
        
        #mean values to array for matrix multiplication
        uvw_raw_mean = uvw_raw_mean.to_array().values
        #averaged partially rotated dataset for Gamma
        uvw2_mean = np.matmul(P, uvw_raw_mean)
        #get rotation angle Gamma
        u_mean = uvw2_mean[0]
        v_mean = uvw2_mean[1]
        gamma, M = get_gamma_rotation_M(u_mean, v_mean)
        
        #Finally rotation around z
        #target array
        uvw3 = np.full_like(uvw2, np.nan)
        
        #get intervall indices 
        idx = np.searchsorted(times, edges)
        
        #for loop for rotation
        for ii in range(len(idx) - 1):
            s, e = idx[ii], idx[ii+1]
            MM = M[ii]
            block = uvw2[:, s:e]
            uvw3[:, s:e] = np.matmul(MM, block)
        
    #2. AVERAGING OVER WHOLE PERIOD --> ONE SINGLE ROTATION AROUND Z
    #   TRY TO GET V_MEAN TO ZERO OVER WHOLE DATA  
    else:
        
        #mean values to array for matrix multiplication
        uvw_raw_mean = uvw_raw_mean.to_array().values
        uvw2_mean = np.matmul(P, uvw_raw_mean)
        
        #get gamma (rotation angle around z) and rotation Matrix M
        u_mean = uvw2_mean[0]
        v_mean = uvw2_mean[1]
        gamma, M = get_gamma_rotation_M(u_mean, v_mean) 
    
        #rotation around z axis
        uvw3 = np.matmul(M[0], uvw2)
    
    return uvw3

#%% helpers
def merge_to_dataset_2sets(
        uvw2, uvw3, uvw_raw
):
        
    ds_uvw2 = xr.Dataset(
        {"u": (("time",), uvw2[0, :]),
         "v": (("time",), uvw2[1, :]),
         "w": (("time",), uvw2[2, :])
         },
        coords={"time": uvw_raw.time}
        )
    
    ds_uvw3 = xr.Dataset(
        {"u": (("time",), uvw3[0, :]),
         "v": (("time",), uvw3[1, :]),
         "w": (("time",), uvw3[2, :])
         },
        coords={"time": uvw_raw.time}
        )
    
    return ds_uvw2, ds_uvw3

def merge_to_dataset(
        uvw3, uvw_raw
):
    
    ds_uvw3 = xr.Dataset(
        {"u": (("time",), uvw3[0, :]),
         "v": (("time",), uvw3[1, :]),
         "w": (("time",), uvw3[2, :])
         },
        coords={"time": uvw_raw.time}
        )
    
    return ds_uvw3

#%% select wind sector
def select_sector(ds, sector):
    start, end = sector[0], sector[1]
    
    if start == end:
        raise ValueError("sector has zero width!!!")
    
    #calculate wind direction
    wdir = (np.degrees(np.arctan2(ds["u"], ds["v"])) + 360) % 360
 
    #select data - first limit included, second excluded
    if start < end:
        cond = (wdir >= start) & (wdir < end)
    else:
        cond = (wdir >= start) | (wdir < end)

    ds_sel = ds.where(cond, drop = True)
    
    
    return ds_sel

#%% execute planar fit
#averaging_intervall = "30min"
#for averaging_intervall in ["1min", "5min", "30min", "2h"]:
#    print(f"planar fit rotation for avg. intervall {averaging_intervall}")
def execute_planar_fit(ds, averaging_intervall = "1min", sectors = None,
                       output_plot = None, datapath = None, ret = None,
                       calculate_moisture_variables = False):

        
    #if averaging over whole data
    if averaging_intervall == "whole_data":
        averaging_intervall = None
        
    #basic cleaning
    ds = threshold_correction(ds)
    
    #storage lists
    uvw3_list_height = []
    heights_list = list(ds.heights.values)
    for h in ds.heights:
        ds_h = ds[["u", "v", "w"]].sel(heights = h)
        #select sector
        if sectors:
            sec_h = sectors[str(h.values)]
            print(f"sectorwise planar fit for height {h.values}\n" 
                  f" sectors are: {sec_h}")
            
            #sector lists
            uvw3_sec_list = []

            for sec in sec_h:
                
                print(f"execute planar fit for sector {sec}")
                uvw_raw = select_sector(ds_h, sec)
                print("sector selection done, execute planar fit")
                
                #check if there is any data at all
                if uvw_raw.time.size == 0:
                    print(f"No data from sector {sec} --> skip")
                    continue
                #call planar fit 
                uvw3 = planar_fit_main(uvw_raw, 
                                         averaging_intervall = averaging_intervall,
                                         height=h.values,
                                         output_plot = output_plot)
                
                print(f" planar fit for height {h.values} and sector {sec} done")
                ds_uvw3_sec = merge_to_dataset(uvw3, uvw_raw)
                #save to storage list
                uvw3_sec_list.append(ds_uvw3_sec)
                
            #to one dataset for this height
            ds_uvw3_height = xr.concat(uvw3_sec_list, dim = "time")
            
            #save to storage list for the datasets per height
            uvw3_list_height.append(ds_uvw3_height)
            
                
        else: 
            uvw_raw = ds_h
            uvw3 = planar_fit_main(uvw_raw, 
                                     averaging_intervall = averaging_intervall,
                                     height=h.values,
                                     output_plot = output_plot)
            
            ds_uvw3_height = merge_to_dataset(uvw3, uvw_raw)

            #save to storage list
            uvw3_list_height.append(ds_uvw3_height)
        
    #concatenate lists to dataset again
    #uvw2_all = xr.concat(uvw2_list_height, dim=pd.Index(heights_list, name="heights"))
    uvw3_all = xr.concat(uvw3_list_height, dim=pd.Index(heights_list, name="heights"))
    
    #reindex to original time
    #uvw2_all = uvw2_all.reindex(time = ds.time)
    uvw3_all = uvw3_all.reindex(time = ds.time)
    
    #add temperature if it existed in original dataset
    if "Ts" in ds.data_vars:
        uvw3_all["Ts"] = ds["Ts"]
    #additional moisture variables
    additional_variables = ["q", "rho_v", "p", "wv", "Tair"]
    for var in additional_variables:
        if var in ds.data_vars:
            uvw3_all[var] = ds[var]
        
    #ADDITONAL THRESHOLD CORRECTION FOR 0.5m W COMPONENT --
    #--> too many errors to be cleaned by despiking in some periods
    if 0.5 in uvw3_all.heights.values:
        mask = np.abs(uvw3_all["w"].sel(heights=0.5)) < 2.5
        uvw3_all["w"].loc[dict(heights=0.5)] = (
                                                uvw3_all["w"].sel(heights=0.5).where(mask)
        )
        
    #despiking with default input arguments --> AFTER ROTATION!
    uvw3_all, nan_dict = despiking(uvw3_all)#, 
                             #output_path_plot = Path(r"D:\HEFEXIII\plots\tower\despiking"))
                             
    #intervalls with less than 80% data --> NaN
    uvw3_all = remove_fragmentary_data(uvw3_all)                       

    #get frequency for saving 
    seconds = ds.time.diff("time").median().values / np.timedelta64(1, "s")
    freq_str = f"{round(1/seconds)}Hz"
    
    #calculate additonal moisture variables after despiking
    #---- ATTENTION: FUNCTIONS USE FIXED DEFAULT UNITS. ERROR IF DATA HAS DIFFERENT
    if all(v in uvw3_all.data_vars for v in ["p", "rho_v", "Tair"]):
        
        uvw3_all["e"] = water_vapour_pressure_from_moistdensity(uvw3_all["rho_v"], 
                                                          uvw3_all["Tair"])
        uvw3_all["wv"] = mixing_ratio_from_watervapour(uvw3_all["e"],
                                                       uvw3_all["p"])
        uvw3_all["q"] = specific_humidity_from_mixing_ratio(uvw3_all["wv"])
        uvw3_all["rho_d"] = dry_air_density(uvw3_all["p"], 
                                            uvw3_all["e"], uvw3_all["Tair"])
     

    #save data
    if datapath:
        os.makedirs(datapath, exist_ok = True)
        if averaging_intervall:
            filename = f"metek_L3_{freq_str}_{averaging_intervall}.nc"
            if any(v in uvw3_all.data_vars for v in additional_variables):
                filename = f"smartflux_L3_{freq_str}_{averaging_intervall}.nc"
        else:
            filename = "metek_L3_{freq_str}_fixedsystem.nc"
            if any(v in uvw3_all.data_vars for v in additional_variables):
                filename = f"smartflux_L3_{freq_str}_fixedsystem.nc"          
            
        uvw3_all.to_netcdf(datapath /filename)
        
        print(f"saving of {filename} done!")
        
        #partial rotation (=without z-axis rotation)
        #filename = "metek_L3_partial_30Hz_orig.nc"
        #filepath = datapath / filename
        
        #if not filepath.exists():
        #    uvw2_all.to_netcdf(filepath)
        #else: 
        #    print(f"xy-rotated data already existing --> skip saving")
            
    if ret:
        return uvw3_all
    else:
        del uvw3_all
                
    #return uvw3_all



#%% execute planar fit
filepath = Path("D:\HEFEXIII\Tower\L2")
smartflux_20Hz = r"metek_L2_20Hz_smartflux.nc"
metek_20Hz = r"metek_L2_20Hz.nc"

#windows = ["1min", "5min", "2h", "whole_data"]
windows = ["30min"]

for file in [smartflux_20Hz, metek_20Hz]:
    data = xr.open_dataset(filepath / file)
    #data = data.drop_vars("q")
    variables = ["u", "v", "w", "Ts"]
    additional_variables = ["Tair", "p", "rho_v", "q", "wv", "e"]
    
    for v in additional_variables:
        if v in data.data_vars:
            variables.append(v)
    
    data = data[variables]#.astype("float32")
    for averaging_intervall in windows:
        ds = data.copy()
        execute_planar_fit(ds, averaging_intervall = averaging_intervall, 
                              datapath = Path(r"D:\HEFEXIII\Tower\L3\no_sectorwise"))
