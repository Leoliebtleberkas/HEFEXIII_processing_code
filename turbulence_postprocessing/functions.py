import xarray as xr
import numpy as np
from scipy import signal
import warnings

def get_fluctuations(ds, config, window_spec = None, method_spec = None):
    # detrends each block and returns fluctuations
    method = config['avg_method']
    window = config['window']
    
    if window_spec is not None:
        window = window_spec
    if method_spec is not None:
        method = method_spec

    # avoid loss of heights dimension for single sonic
    heights = ds.heights

    # Compute fluctuations according to method
    if method == 'block':
        ds_mean = ds.resample(time=window, label = "right", 
                              closed = "right").mean().reindex(time=ds.time).bfill(dim='time')
        ds_fluct = ds - ds_mean
    elif method == 'detrend':
        ds_fluct = detrend(ds, window)
    elif method == 'detrend_2step':
        ds_detrend_runmean, ds_runmean = detrend_runmean(ds, config)
        ds_fluct = detrend(ds_detrend_runmean, window)
    else:
        warnings.warn('avg_method {} not recognized, detrend will be used'.format(method))
        ds_fluct = detrend(ds, window)

    return ds_fluct.assign_coords(heights=heights)

def detrend_runmean(ds, config):
    '''
    Function for detrending data at selected heights with a running mean
    Parameters
    ----------
    ds : xr.Dataset
        input data which should be detrended with a running mean. 
    config : dict
        contains a list of heights where the data should be detrended (runmean_heights)
        and a list of the window/sample sizes for every height. Has to be in correct order!

    Raises
    ------
    ValueError
        if length of the given lists config["runmean_heights"] and 
        config["runmean_nsample"] does not match

    Returns
    -------
    ds_detrend : xr.Dataset
        Contains the detrended data for the heights given in runmean_heights 
        and the original data for all other heights
    ds_runmean : xr.Dataset
        The calculated running means data for given heights, np.nan for 
        all other heights

    '''
    
    #window running mean
    runmean_heights = config['runmean_heights']
    runmean_nsample = config['runmean_nsample']
    
    if len(runmean_heights) != len(runmean_nsample):
        raise ValueError("length of runmean_heights and runmean_windows must be equal")
        
    for h in runmean_heights:
        if h not in ds.heights:
            warnings.warn(f"{h}m not existing in data...will ignore")
        
    runmean_list = []
    detrend_list = []
    for h in ds.heights:
        if h in runmean_heights:
            sample_idx = runmean_heights.index(h)
            window = runmean_nsample[sample_idx]
            ds_h = ds.sel(heights = h)
            
            ds_runmean = ds_h.rolling(time = window, min_periods = 1, 
                                             center = True).mean()
            ds_detrend = ds_h - ds_runmean
            
            detrend_list.append(ds_detrend)
            runmean_list.append(ds_runmean)
        else: 
            warnings.warn(f"skip height {h.values}m...not given in runmean_heights")
            ds_runmean = xr.full_like(ds.sel(heights = h), np.nan)
            ds_detrend = ds.sel(heights = h).copy()
            
            
            detrend_list.append(ds_detrend)
            runmean_list.append(ds_runmean)
        
            
        
    #back to dataset
    ds_detrend = xr.concat(detrend_list, dim = "heights")
    ds_runmean = xr.concat(runmean_list, dim = "heights")

    return ds_detrend, ds_runmean

def detrend_old(ds, window):
    ds_groups = ds.resample(time=window, label = "right", 
                            closed = "right")
    group_list = []

    for label, group in ds_groups:
        axis = list(ds.sizes.keys()).index("time")
        group_list.append(group.map(signal.detrend, type = "linear", 
                                    axis = axis))
        #original
        #group_list.append(group.map(signal.detrend, args=[0])) 


    return xr.concat(group_list, dim='time')

def detrend(ds, window):
    ds_groups = ds.resample(time=window, label="right", closed="right")
    group_list = []

    for label, group in ds_groups:
        detrended_heights = []
        for h in group.heights.values:
            group_h = group.sel(heights=h)
            valid = group_h.dropna(dim="time", how="any")
            
            if len(valid.time) < 2:
                detrended_heights.append(group_h * np.nan)
                continue
            
            detrended = valid.map(signal.detrend, type="linear", axis=0)
            detrended = detrended.reindex(time=group_h.time)
            detrended_heights.append(detrended)
        
        group_list.append(xr.concat(detrended_heights, dim="heights"))

    return xr.concat(group_list, dim='time')

def detrend_finite(ds, window):
    """
    Apply linear detrending on finite values in a Dataset per resample window.
    NaNs are preserved.
    """
    ds_groups = ds.resample(time=window, label = "right", 
                            closed = "right")
    group_list = []

    for label, group in ds_groups:
        def detrend_da(da):
            # Mask finite values
            finite_mask = np.isfinite(da)
            data = da.copy()
            if finite_mask.any():
                finite_vals = data.values[finite_mask.values]
                finite_vals = signal.detrend(finite_vals, type='linear')
                data.values[finite_mask.values] = finite_vals
            return data

        # Apply detrending only on finite values
        group_list.append(group.map(detrend_da))

    return xr.concat(group_list, dim='time')