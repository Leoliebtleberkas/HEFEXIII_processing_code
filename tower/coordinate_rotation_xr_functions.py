# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 08:52:07 2026

@author: leopo

functions needed for 3D wind component coordinate rotations
"""

#import packages
from pathlib import Path
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from metpy.units import units
from metpy.calc import wind_direction, wind_speed
#%%helpers
def false_ratio(
        ds, h, threshold_pitch = 10, threshold_roll = 10
):
    '''
    function for identifying error values of pitch and roll angles based on thresholds.
    thresholds have to be identified, e.g. by plotting and visually identifying!
    prints the percentage of identified error values

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing "Pitch" and "Roll" angles and a heights coordinate.
        Function wont work if one is missing
    threshold_pitch : float or int, optional
        threshold in degrees for pitch angle error values. The default is 10.
    threshold_roll : float or int, optional
        threshold in degrees for roll angle error values. The default is 10.

    Returns
    -------
    None. Prints the percentage of error values only

    '''
    for h in ds.heights:
        dsh = ds.sel(heights = h)
        outliers_pitch = dsh["Pitch"].where(
            np.abs(dsh["Pitch"]) >= threshold_pitch).dropna(dim = "time")
        outliers_roll = dsh["Roll"].where(
            np.abs(dsh["Roll"]) >= threshold_roll).dropna(dim = "time")
        perc_pitch = len(outliers_pitch) / len(dsh["Pitch"])
        perc_roll = len(outliers_roll) / len(dsh["Roll"])
        print(f"height {h.values}: false ratio Pitch, {threshold_pitch}° threshold: {perc_pitch}")
        print(f"height {h.values}: false ratio roll, {threshold_roll}° threshold: {perc_roll}")


def correct_roll_pitch(
        ds, threshold_pitch = 10, threshold_roll = 10
):
    '''
    threshold based cleaning of pitch and roll angles of met. sensors. 
    was used for metek-uSonics from HEFEXIII 

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing variables "Pitch" and "Roll".
    threshold_pitch : float or int, optional
        threshold in degrees for pitch angle error values. These values
        will be removed and filled with forward and backward fill. The default is 10.
    threshold_roll : float or int, optional
        threshold in degrees for roll angle error values. These values
        will be removed and filled with forward and backward fill. The default is 10.

    Returns
    -------
    ds : xr.Dataset
        dataset with cleaned pitch and roll angles

    '''
    
    ds["Pitch"] = ds["Pitch"].where(np.abs(ds["Pitch"]) <= threshold_pitch, np.nan)
    ds["Roll"] = ds["Roll"].where(np.abs(ds["Roll"]) <= threshold_roll, np.nan)
    
    #backwards fill
    ds["Pitch"] = ds["Pitch"].bfill(dim = "time")
    ds["Pitch"] = ds["Pitch"].ffill(dim = "time")
    ds["Roll"] = ds["Roll"].bfill(dim = "time")
    ds["Roll"] = ds["Roll"].ffill(dim = "time")
    return ds

def clean_compass_return_mean(
        ds, threshold_low = 90, threshold_upper = 120
):
    '''
    function for cleaning compass data and returning the average over height, 
    in case of multiple observation heights. Was used for gill compasses of HEFEXIII

    Parameters
    ----------
    ds : xr.Dataset
        dataset containing "Compass" variable with heights coordinate.
    threshold_low : float or int
        lower boundary for valid values. Default is 90
    threshold_upper : float or int
        upper boundary for valid values. Default is 120

    Returns
    -------
    comp_mean : xr.Dataset
        cleaned compass data, averaged over all observation heights.

    '''
    ds["Compass"] = ds["Compass"].where((ds["Compass"] >= threshold_low) & 
                                        (ds["Compass"] <= threshold_upper), np.nan) 

    #calculate mean compass
    comp_mean = ds["Compass"].mean(dim = "heights")
    
    return comp_mean

def calc_wdir(
        ds
):
    '''
    function for calculating wind direction

    Parameters
    ----------
    ds : xr.Dataset
        dataset containing u and v wind component in m/s.

    Returns
    -------
    ds : xr.Dataset
        dataset with new variable "Wdir" containing wind direction in °.

    '''
    ds["Wdir"] = wind_direction(ds["u"] * units("m/s"), 
                           ds["v"] * units("m/s"))
    
    return ds

def calc_wspeed(
        ds
):
    '''
    function for calculating wind speed.    

    Parameters
    ----------
    ds : xr.Dataset
        dataset with new variable "Wspd" containing wind speed in m/s.

    Returns
    -------
    ds : TYPE
        DESCRIPTION.

    '''
    ds["Wspd"] = wind_speed(ds["u"] * units("m/s"), 
                           ds["v"] * units("m/s"))
    
    return ds

def plot_wind_distribution(
        ds, label, bins = 36, title = "Direction Distribution", 
        output_path = r"D:\HEFEXIII\plots\tower\winddirhist_check"
):
    '''
    Function for plotting a histogramm of wind direction 

    Parameters
    ----------
    ds : xr.Dataset
        dataset containing wind direction variable "Wdir".
    label : str
        label for plot legend.
    bins : int, optional
        bins forthe plotted wind direction histogram. The default is 36.
    title : str, optional
        plot title. The default is "Direction Distribution".
    output_path: str, optional
        path were plot should be saved

    Returns
    -------
    None. saves plot as .png under output path

    '''
    fig, ax = plt.subplots(figsize = (10, 3))
    
    ax.hist(ds["Wdir"], bins = bins,
               label = label) 
    
    
    plt.legend()
    plt.title(title)
    
    #save
    plt.tight_layout()
    
    plt.savefig(Path(output_path) / 
                f"{title.replace(" ", "")}_{label}_xr.png")
    plt.close()


#%%coordinate rotation
#three distinct functions, less confusion, all after Aubinet 2012 
#1. rotation around z-axis, correct offset from north-south (=yaw angle, R01)
def correct_zaxis(
        u, v, angle, degree = True
):
    '''
    
    Parameters
    ----------
    u : array-like
        raw u component wind data.
    v : array-like
        raw v component wind data.
    angle : int or float
        angle for rotation around z-axis, can be in radian or degree.
    degree : Boolean, optional
        if True treat the given angle as value in degree, otherwise as radian.
        The default is True.

    Returns
    -------
    u_rot : array-like
        rotated u-component.
    v_rot : array-like
        rotated v-component.

    '''
    if degree:
        #in radian and negative because counter-clockwise rotation
        angle = np.deg2rad(angle)
    #calculation
    u_rot = u * np.cos(angle) + v * np.sin(angle)
    v_rot = -1 * u * np.sin(angle) + v * np.cos(angle)
    return u_rot, v_rot
   
   
#2. rotation around y-axis, offset from vertical in y-dir 
def correct_yaxis(
        u, w, angle, degree = True
):
    '''
    
    Parameters
    ----------
    u : array-like
        raw u component wind data.
    w : array-like
        raw w component wind data.
    angle : int or float
        angle for rotation around y-axis, can be in radian or degree.
    degree : Boolean, optional
        if True treat the given angle as value in degree, otherwise as radian.
        The default is True.

    Returns
    -------
    u_rot : array-like
        rotated u-component.
    w_rot : array-like
        rotated w-component.

    '''
    if degree:
        #in radian
        angle = np.deg2rad(angle)
    #calculation
    u_rot = u * np.cos(angle) + w * np.sin(angle)
    w_rot = -1 * u * np.sin(angle) + w * np.cos(angle)
    return u_rot, w_rot

#3. rotation around x-axis, offset from vertical in x-dir
def correct_xaxis(
        v, w, angle, degree = True
):
    '''
    
    Parameters
    ----------
    v : array-like
        raw v component wind data.
    w : array-like
        raw w component wind data.
    angle : int or float
        angle for rotation around x-axis, can be in radian or degree.
    degree : Boolean, optional
        if True treat the given angle as value in degree, otherwise as radian.
        The default is True.

    Returns
    -------
    v_rot : array-like
        rotated v-component.
    w_rot : array-like
        rotated w-component.

    '''
    if degree:
        #in radian
        angle = np.deg2rad(angle)
    #calculation
    v_rot = v * np.cos(angle) + w * np.sin(angle)
    w_rot = -1 * v * np.sin(angle) + w * np.cos(angle)
    return v_rot, w_rot