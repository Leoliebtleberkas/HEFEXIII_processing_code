
#%%import packages
import numpy as np
import xarray as xr
import pandas as pd
import os 
from pathlib import Path
os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\turbulence_postprocessing")
from functions import get_fluctuations, detrend_runmean
from spectral_analysis import spectra_eps
from structure_functions import structure_functions_epsilon
from autocorrelation import autocorrelation
from MRD_functions import multiresolution
os.chdir(r"C:\Users\leopo\PhD\glacier_space\HEFEXIII_processing_code\helper_functions")
from planarfit_xr_functions import execute_planar_fit
#from postprocess_plots import plot_despiking_corrections
import warnings

#%%


# Run routine
def postprocess(ds, config):
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

    # handle single sonic
    if "heights" not in ds.coords:
        ds = ds.assign_coords(heights=[1])
        warnings.warn(
            "Dimension heights was not present, added with fake value, this will cause an imprecise"
            " determination of the cutoff in spectral frequency"
        )

    results = {}
    # gap_filling
    ds, nan_perc = fill_gaps(ds, config)

    #rotation method
    # double rotation
    if config["rotation"] == "double_rotation":
        ds, rotation = double_rotation(ds, config)
    elif config["rotation"] == "planar_fit":
        uvw_raw = ds[["u", "v", "w", "Ts"]].copy()
        #ds_planeonly is without y-rotation, ds is with all 3 rotations
        #rotation is dictionary with all 3 rotation angles
        ds = execute_planar_fit(uvw_raw, averaging_intervall = config["window"], 
                             ret = True)
    elif config["rotation"] == "no":
        warnings.warn("Proceed without rotation, only valid if dataset \
                      contains already rotated data!")
        ds = ds

    # fluxes calculation - also includes detrending (or block averaging)
    fluxes = fluxes_calculation(ds, config)

    # stationarity
    stat = stationarity(ds, config)
    # merge
    statistics = xr.merge([fluxes, rotation, stat, nan_perc])

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

    # create results dictionary
    results["stats"] = statistics
    return results


# --------------------
# Statistics functions
# --------------------


def make_time_regular(
        ds
):
    #original time
    time = ds.time.values.astype("datetime64[ns]")
    dt = np.median(np.diff(time))

    t0 = time[0]
    #whole number indizes for calculating new and regular time
    k1 = np.round((time - t0) / dt).astype(np.int64)
    #correct k
    k = k1.copy()
    for i in range(1, len(k)):
        if k[i] <= k[i-1]:
            k[i] = k[i-1] + 1
       
    #artificial time in regular steps mapped to existing time
    time_mapped = t0 + k*dt

    #replace original time by mapped time --> df has regular time, gaps still exist
    ds = ds.assign_coords(time = time_mapped)
    
    #detect gaps
    gap_idx = np.where(np.diff(k) > 1)[0]

    #fill gaps in k
    missing_k = []
    for i in gap_idx:
        missing_k.extend(range(k[i]+1, k[i+1]))
    missing_k = np.array(missing_k, dtype=np.int64)

    #calculate missing timesteps 
    time_fill = t0 + missing_k * dt

    #final filled and regular timedata
    time_final = np.concat([time_mapped, time_fill])
    time_final = np.sort(time_final)

    #reindex the data --> has no regular time, gaps filled with nan
    ds = ds.reindex(time = time_final)
    
    return ds
    

def fill_gaps(ds, config, count_nans=True, add_missing_timesteps=False):
    # config
    window = config["window"]
    method = config["gap_filling"]

    # check method
    if method != "interp":
        warnings.warn(
            "Gap filling method {} not recognized, will use interpolation".format(
                method
            )
        )
        method = "interp"
        
    #reindex ds to account for entirely missing timesteps
    if add_missing_timesteps:
        ds = make_time_regular(ds)

    # count nans per period
    var_list = list(ds.data_vars)
    ds_groups = ds.resample(time=window, label = "right", 
                            closed = "right")
    nans = []
    gap_fill = []
    nan_warned = False

    # main loop
    for label, group in ds_groups:
        # count nans
        nan_array = np.invert(np.isfinite(group[var_list[0]]))
        for var in var_list[1:]:
            nan_array = nan_array | np.invert(np.isfinite(group[var]))
        nans.append(nan_array)
        
        # FILL GAPS
        # remove infs
        for var in var_list:
            group[var] = group[var].where(np.isfinite(group[var]), other=np.nan)

        # gap filling
        if method == "interp":
            group = group.interpolate_na(dim="time", limit=10) #ORIGINAL 10
            #bfill and ffill for data edges
            
            group = group.bfill(dim = "time", limit = 5)
            group = group.ffill(dim = "time", limit = 5)

        # take care of residual nans
        for var in var_list:
            if (np.isnan(group[var])).sum() > 0:
                # put to mean what is still nan after interpolate
                group[var] = group[var].where(
                    np.isfinite(group[var]), other=group[var].mean(dim = "time")
                )
                # if nans are surviving it means the period was empty, put surviving nans to zero
                group[var] = group[var].where(np.isfinite(group[var]), other=0)
                # surviving nans warning, only warn once per period
                if not nan_warned:
                    warnings.warn(
                        "Nans are surviving gap filling in time period {}, set to mean value. Will only warn once.".format(
                            group.time[0].data
                        )
                    )
                    #nan_warned = True
        gap_fill.append(group)

    # reconcatenate
    ds = xr.concat(gap_fill, dim="time")
    nans = xr.concat(nans, dim="time").resample(time=window, label = "right", 
                                                closed = "right")
    nan_perc = (nans.sum() / nans.count()).rename("QCnan")

    if count_nans:
        return ds, nan_perc
    else:
        return ds


def double_rotation(ds, config, return_rotation=True):
    """applies double rotation to raw data and returns rotated data and rotation angles"""

    # config
    window = config["window"]

    # means
    ds_mean = ds.resample(time=window, label = "right", closed = "right").mean()

    # rotation angles
    theta = np.arctan2(ds_mean.v, ds_mean.u)
    phi = np.arctan2(ds_mean.w, np.sqrt(ds_mean.u**2 + ds_mean.v**2))
    rotation = xr.Dataset(
        data_vars=dict(dir=(270 - theta * 180 / np.pi) % 360, theta=theta, phi=phi)
    )

    # sines and cosines
    ct = np.cos(theta).reindex(time=ds.time).ffill(dim="time")
    st = np.sin(theta).reindex(time=ds.time).ffill(dim="time")
    cp = np.cos(phi).reindex(time=ds.time).ffill(dim="time")
    sp = np.sin(phi).reindex(time=ds.time).ffill(dim="time")

    # rotate
    u_rot = ct * cp * ds.u + st * cp * ds.v + sp * ds.w
    v_rot = -st * ds.u + ct * ds.v
    w_rot = -ct * sp * ds.u - st * sp * ds.v + cp * ds.w
    ds = ds.assign(
        u=u_rot,
        v=v_rot,
        w=w_rot,
    )

    if return_rotation:
        return ds, rotation
    else:
        return ds


def fluxes_calculation(ds, config, third="all", fourth="main"):
    """Calculates statistics by specified orders at the chosen time average,
    method: block, detrend to be passed to the get_fluctuations function
    third (fourth): all= all possible combinations of u,v,w,t;
                    main = just uuu(u), vvv(v) ecc;
                    False= don't include the third (fourth) order statistics
    fourth order 'all' not implemented"""

    # config
    window = config["window"]

    ds_fluct = get_fluctuations(ds, config)
    ds_fluct_orig = ds_fluct.copy()

    # Second order and means
    fluxes = xr.Dataset(
        data_vars=dict(
            meanU=ds.u,
            meanT=ds.tc,
            uu=ds_fluct.u * ds_fluct.u,
            vv=ds_fluct.v * ds_fluct.v,
            ww=ds_fluct.w * ds_fluct.w,
            uv=ds_fluct.u * ds_fluct.v,
            uw=ds_fluct.u * ds_fluct.w,
            vw=ds_fluct.v * ds_fluct.w,
            TT=ds_fluct.tc * ds_fluct.tc,
            uT=ds_fluct.u * ds_fluct.tc,
            vT=ds_fluct.v * ds_fluct.tc,
            wT=ds_fluct.w * ds_fluct.tc,
            sdir=np.arctan2(ds_fluct.v, ds_fluct.u) ** 2 * 180 / np.pi,
        )
    )
    fluxes = fluxes.assign(tke=0.5 * (fluxes.uu + fluxes.vv + fluxes.ww))
    if "q" in ds.data_vars:
        fluxes = fluxes.assign(qq = ds_fluct.q * ds_fluct.q,
                               wq = ds_fluct.w * ds_fluct.q)
    if "rho_v" in ds.data_vars:
        fluxes = fluxes.assign(rho_v2 = ds_fluct.rho_v * ds_fluct.rho_v,
                               wrho_v = ds_fluct.w * ds_fluct.rho_v)
    if "wv" in ds.data_vars:
        fluxes = fluxes.assign(wv2 = ds_fluct.wv * ds_fluct.wv,
                               wwv = ds_fluct.w * ds_fluct.wv)

    # Third order
    if third is not False:
        fluxes = fluxes.assign(
            uuu=ds_fluct.u * ds_fluct.u * ds_fluct.u,
            vvv=ds_fluct.v * ds_fluct.v * ds_fluct.v,
            www=ds_fluct.w * ds_fluct.w * ds_fluct.w,
            TTT=ds_fluct.tc * ds_fluct.tc * ds_fluct.tc,
        )
        if "q" in ds.data_vars:
            fluxes = fluxes.assign(qqq = ds_fluct.q * ds_fluct.q * ds_fluct.q)
            
        if "wv" in ds.data_vars:
            fluxes = fluxes.assign(wv3 = ds_fluct.wv * ds_fluct.wv * ds_fluct.wv)
            
        if "rho_v" in ds.data_vars:
            fluxes = fluxes.assign(rho_v3 = ds_fluct.rho_v*ds_fluct.rho_v*ds_fluct.rho_v)

    if third == "all":
        fluxes = fluxes.assign(
            uuv=ds_fluct.u * ds_fluct.u * ds_fluct.v,
            uuw=ds_fluct.u * ds_fluct.u * ds_fluct.w,
            uvw=ds_fluct.u * ds_fluct.v * ds_fluct.w,
            uvv=ds_fluct.u * ds_fluct.v * ds_fluct.v,
            uww=ds_fluct.u * ds_fluct.w * ds_fluct.w,
            vvw=ds_fluct.v * ds_fluct.v * ds_fluct.w,
            vww=ds_fluct.v * ds_fluct.w * ds_fluct.w,
            utke=fluxes.tke * ds_fluct.u,
            vtke=fluxes.tke * ds_fluct.v,
            wtke=fluxes.tke * ds_fluct.w,
            uuT=ds_fluct.u * ds_fluct.u * ds_fluct.tc,
            vvT=ds_fluct.v * ds_fluct.v * ds_fluct.tc,
            wwT=ds_fluct.w * ds_fluct.w * ds_fluct.tc,
            uvT=ds_fluct.u * ds_fluct.v * ds_fluct.tc,
            uwT=ds_fluct.u * ds_fluct.w * ds_fluct.tc,
            vwT=ds_fluct.v * ds_fluct.w * ds_fluct.tc,
            uTT=ds_fluct.u * ds_fluct.tc * ds_fluct.tc,
            vTT=ds_fluct.v * ds_fluct.tc * ds_fluct.tc,
            wTT=ds_fluct.w * ds_fluct.tc * ds_fluct.tc,
        )
        if "q" in ds.data_vars:
            fluxes = fluxes.assign(
                        wwq = ds_fluct.w * ds_fluct.w * ds_fluct.q
                        )
        if "wv" in ds.data_vars:
            fluxes = fluxes.assign(
                        wwwv = ds_fluct.w * ds_fluct.w * ds_fluct.wv
                        )
        if "rho_v" in ds.data_vars:
            fluxes = fluxes.assign(
                        wwrho_v = ds_fluct.w * ds_fluct.w * ds_fluct.rho_v
                        )

    # fourth order
    if fourth is not False:
        fluxes = fluxes.assign(
            uuuu=ds_fluct.u * ds_fluct.u * ds_fluct.u * ds_fluct.u,
            vvvv=ds_fluct.v * ds_fluct.v * ds_fluct.v * ds_fluct.v,
            wwww=ds_fluct.w * ds_fluct.w * ds_fluct.w * ds_fluct.w,
            TTTT=ds_fluct.tc * ds_fluct.tc * ds_fluct.tc * ds_fluct.tc,
        )
        if "q" in ds.data_vars:
            fluxes = fluxes.assign(
                        qqqq = ds_fluct.q * ds_fluct.q * ds_fluct.q * ds_fluct.q)
        if "wv" in ds.data_vars:
            fluxes = fluxes.assign(
                        wv4 = ds_fluct.wv **4)
        if "rho_v" in ds.data_vars:
            fluxes = fluxes.assign(
                        rho_v4 = ds_fluct.rho_v **4)
        
    #high frequency fluxes
    fluxes_orig = fluxes.copy()

    # average
    fluxes = fluxes.resample(time=window, label = "right", 
                             closed = "right").mean(dim = "time")

    # ustar
    fluxes = fluxes.assign(ustar=(fluxes.uw**2 + fluxes.vw**2) ** 0.25)

    return fluxes, ds_fluct_orig


def stationarity(ds, config):
    """Calculates starionarity on sub-intervals of a sixth of the length of the original window"""

    # config
    window = config["window"]

    # define window of sub intervals
    sub_window = window.split("m")[0] + "0s"

    # take mean out and divide in time blocks
    ds_fluct = get_fluctuations(ds, config)
    ds_groups = ds_fluct.resample(time=window, label = "right", 
                                  closed = "right")
    stat = []

    for label, group in ds_groups:
        # block covariance
        group_cov = calc_cov(group).assign_coords(time=label)
        # divide in sub-samples
        sub_groups = group.resample(time=sub_window, label = "right", 
                                    closed = "right")
        sub_group_cov = []
        # stats for each subsample
        for lab, sub_group in sub_groups:
            sub_group_cov.append(calc_cov(sub_group))
        # mean between subsamples
        sub_group_cov = (
            xr.concat(sub_group_cov, dim="time")
            .mean(dim="time")
            .assign_coords(time=label)
        )
        # stationarity tests
        stat.append(100 * np.abs((group_cov - sub_group_cov) / group_cov))

    return xr.concat(stat, dim="time").rename(dict(U="statU", uw="statUW", wT="statWT"))


# Helpers
# ----------
def calc_cov(ds):
    """method used by Postprocess_whole.stationarity to calculate the subgroup covariances"""
    ds_cov = xr.Dataset(
        data_vars=dict(
            U=(ds.u * ds.u).mean(dim="time")  #DEFAULT:SAMU USED SQRT of whole term (with 2nd line)
                - ds.u.mean(dim="time") * ds.u.mean(dim="time"),
            uw=(ds.u * ds.w).mean(dim="time")
            - ds.u.mean(dim="time") * ds.w.mean(dim="time"),
            wT=(ds.w * ds.tc).mean(dim="time")
            - ds.w.mean(dim="time") * ds.tc.mean(dim="time"),
        )
    )
    return ds_cov

# Helpers
# ----------
def calc_cov_new_mean(ds):
    """method used by Postprocess_whole.stationarity to calculate the subgroup covariances"""
    ds_cov = xr.Dataset(
        data_vars=dict(
            U=(ds.u * ds.u).mean(dim="time")
                - ds.u.mean(dim="time") * ds.u.mean(dim="time"),
            uw=(ds.u * ds.w).mean(dim="time")
            - ds.u.mean(dim="time") * ds.w.mean(dim="time"),
            wT=(ds.w * ds.tc).mean(dim="time")
            - ds.w.mean(dim="time") * ds.tc.mean(dim="time"),
        )
    )
    return ds_cov



def stationarity_new_mean(ds, config):
    """Calculates starionarity on sub-intervals of a sixth of the length of the original window"""

    # config
    window = config["window"]

    # define window of sub intervals
    sub_window = pd.to_timedelta(window) / 6
    
    if config["avg_method"] == "detrend_2step":
        ds_pre_detrend, ds_runmean = detrend_runmean(ds, config)
    else:
        ds_pre_detrend = ds
        
    #resample to actual avg. window    
    ds_groups = ds_pre_detrend.resample(time = window, closed = "right", label = "right")
    stat = []

    for label, group in ds_groups:
        
        # block covariance
        group_cov = calc_cov_new_mean(group).assign_coords(time=label)
        # divide in sub-samples
        sub_groups = group.resample(time=sub_window, label = "right", 
                                    closed = "right")
        sub_group_cov = []
        # stats for each subsample
        for lab, sub_group in sub_groups:
            sub_group_cov.append(calc_cov_new_mean(sub_group))
        # mean between subsamples
        sub_group_cov = (
            xr.concat(sub_group_cov, dim="time")
            .mean(dim="time")
            .assign_coords(time=label)
        )
        # stationarity tests
        #stat.append(100 * (group_cov - sub_group_cov) / group_cov)
        stat.append(100 * np.abs((group_cov - sub_group_cov) / group_cov))

    return xr.concat(stat, dim="time").rename(dict(U="statU", uw="statUW", wT="statWT"))



def stationarity_new(ds, config):
    """Calculates starionarity on sub-intervals of a sixth of the length of the original window"""

    # config
    window = config["window"]

    # define window of sub intervals
    sub_window = pd.to_timedelta(window) / 6
        
    #resample to actual avg. window    
    ds_groups = ds.resample(time = window, closed = "right", label = "right")
    stat = []

    for label, group in ds_groups:
        
        # block covariance
        group_cov = calc_cov_new_mean(group).assign_coords(time=label)
        # divide in sub-samples
        sub_groups = group.resample(time=sub_window, label = "right", 
                                    closed = "right")
        sub_group_cov = []
        # stats for each subsample
        for lab, sub_group in sub_groups:
            sub_group_cov.append(calc_cov_new_mean(sub_group))
        # mean between subsamples
        sub_group_cov = (
            xr.concat(sub_group_cov, dim="time")
            .mean(dim="time")
            .assign_coords(time=label)
        )
        # stationarity tests
        #stat.append(100 * (group_cov - sub_group_cov) / group_cov)
        stat.append(100 * np.abs((group_cov - sub_group_cov) / group_cov))

    return xr.concat(stat, dim="time").rename(dict(U="statU", uw="statUW", wT="statWT"))



# Helpers
# ----------
def calc_cov_new(ds):
    """method used by Postprocess_whole.stationarity to calculate the subgroup covariances"""
    ds_cov = xr.Dataset(
        data_vars=dict(
            U=(ds.u * ds.u).mean(dim="time"),
            uw=(ds.u * ds.w).mean(dim="time"),
            wT=(ds.w * ds.tc).mean(dim="time"),
        )
    )
    return ds_cov
