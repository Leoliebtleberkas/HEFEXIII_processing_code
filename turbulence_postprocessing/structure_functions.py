import numpy as np
import xarray as xr
import pandas as pd
from functions import get_fluctuations


# Main routine
# -------------
def structure_functions_epsilon(ds, config, wspd):
    '''works on already rotated data'''

    # config
    window = config['window']

    # divide in time blocks
    ds_fluct = get_fluctuations(ds, config)

    # structure functions
    strfun = strfun_sonic(ds_fluct, window, wspd)

    # dissipation rate
    epsilon = strfun_eps(strfun)

    return strfun, epsilon


# Subroutines
# ------------
def strfun_sonic(ds, window, wspd):
    ''' if not called by main postprocessing needs wspd as dataset with time index same as the window resampling'''

    # check if multiple heights are present
    if len(np.shape(ds.u)) == 1:
        single_sonic = True
    else:
        single_sonic = False

    # group by window
    ds_groups = ds.resample(time=window, label = "right", closed = "right") #original was only (time=window)
    # group length
    #N = ds_groups._group_indices[0].stop
    # indexes and lag logarithmically space
    # ---- ORIGINAL VERSION ----
    dt_old = (ds.time[1] - ds.time[0]).item() / 1e9
    window_seconds = pd.Timedelta(window)
    N_old = int(window_seconds.total_seconds() / dt_old)
    # --- end of original version ---
    
    #new version
    dt = np.median(np.diff(ds.time.values).astype('timedelta64[ns]').astype(float)) #in ns
    window_ns = pd.to_timedelta(window).to_numpy().astype('timedelta64[ns]').astype(float)
    N = int(round(window_ns / dt))
    # end of new version
    
    
    
    k = np.unique((np.logspace(np.log10(1), np.log10(N/2))).astype(int))

    # structure functions
    # resample and loop
    strfun = []
    if single_sonic:
        for label, group in ds_groups:
            
            #safety ceck
            if group.time.size != N:
                print(f"Skipping window {label}, height {group.heights.values} \n"
                      "Group has missing timesteps!!!")
                continue
            
            spd = wspd.sel(time=label)
            sfu = calc_strfun(group.u, group.u, k)
            sfv = calc_strfun(group.v, group.v, k)
            sfw = calc_strfun(group.w, group.w, k)
            lag = np.tile(k, (1, 1)).T * spd.data * 0.05
            strfun.append(xr.Dataset(coords=dict(k=k, time=label),
                                     data_vars=dict(sfu=(['k'], sfu.flatten()),
                                                    sfv=(['k'], sfv.flatten()),
                                                    sfw=(['k'], sfw.flatten()),
                                                    lag=(['k'], lag.flatten()))))
    else:
        for label, group in ds_groups:
            
            #safety ceck
            if group.time.size != N:
                print(f"Skipping window {label}, height {group.heights.values} \n"
                      "Group has missing timesteps!!!")
                continue
            
            spd = wspd.sel(time=label)
            sfu = calc_strfun(group.u, group.u, k)
            sfv = calc_strfun(group.v, group.v, k)
            sfw = calc_strfun(group.w, group.w, k)
            lag = np.tile(k, (len(spd), 1)).T * spd.data * 0.05
            strfun.append(xr.Dataset(coords=dict(k=k, time=label),
                                     data_vars=dict(sfu=(['k', 'heights'], sfu),
                                                    sfv=(['k', 'heights'], sfv),
                                                    sfw=(['k', 'heights'], sfw),
                                                    lag=(['k', 'heights'], lag))))

    strfun = xr.concat(strfun, dim='time').assign_coords(heights=ds.heights)

    return strfun


def strfun_eps(strfun):

    #constants
    cu = 18  /55 * 1.5
    cvw = cu * 4 / 3
    c2u = 4.017 * cu
    c2vw = 4.017 * cvw

    # isolate inertial subrange from slope
    grad = np.log10(strfun.where(strfun.lag < 20)).diff(dim='k').interp(coords={'k':strfun.k})
    slope = grad.sfu / grad.lag
    subrange = (slope > 2 / 3 - 0.1) & (slope < 2 / 3 + 0.1)
    strfun_in = strfun.where(subrange, drop=True)
    #calculate epsilon
    epsUsf = ((strfun_in.sfu/c2u) ** (3 / 2) / strfun_in.lag).mean(dim='k').rename('epsUsf')
    epsVsf = ((strfun_in.sfu / c2vw) ** (3 / 2) / strfun_in.lag).mean(dim='k').rename('epsVsf')
    epsWsf = ((strfun_in.sfu / c2vw) ** (3 / 2) / strfun_in.lag).mean(dim='k').rename('epsWsf')

    #save
    epsilon = xr.merge([epsUsf, epsVsf, epsWsf])

    return epsilon


# Helpers
# -------
def calc_strfun(var1, var2, k):
    N = len(var1.time)
    k = k[k < N]
    strfun = []
    
    time_axis = var1.get_axis_num("time")
    for i in k:
        
        du1 = (
            var1.isel(time=slice(None, -i)).data - var1.isel(time=slice(i, None)).data
        )

        du2 = (
            var2.isel(time=slice(None, -i)).data - var2.isel(time=slice(i, None)).data
        )

        sf = np.mean(du1 * du2, axis=time_axis)

        strfun.append(sf)
    return np.array(strfun)

def calc_strfun_orig(var1, var2, k):
    N = len(var1.time)
    k = k[k < N]
    strfun = []
    
    #necessary if time is not first dimension
    var1 = var1.transpose("time", "heights")
    var2 = var2.transpose("time", "heights")
    
    for i in k:
        
        sf = ((var1[:N - i].data - var1[i:].data) * (var2[:N - i].data - var2[i:].data)).mean(axis=0)

        strfun.append(sf)
    return np.array(strfun)
