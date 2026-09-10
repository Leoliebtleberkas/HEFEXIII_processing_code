import numpy as np
import xarray as xr
import pandas as pd
from functions import get_fluctuations
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)


# Main routine
# -------------
def autocorrelation(ds, config, wspd):
    """works on already rotated data"""

    # config
    window = config["window"]

    # divide in time blocks
    ds_fluct = get_fluctuations(ds, config)

    # structure functions
    autocorr = autocorr_sonic(ds_fluct, window)

    # integral length scales
    intlen = autocorr_intlen(autocorr, wspd)

    return autocorr, intlen


# Subroutines
# ------------
def autocorr_sonic(ds, window):
    # check if multiple heights are present
    if len(np.shape(ds.u)) == 1:
        single_sonic = True
    else:
        single_sonic = False

    # group by window
    ds_groups = ds.resample(time=window)
    # group length
    #N = ds_groups._group_indices[0].stop
    # indexes and lag logarithmically space
    dt = (ds.time[1] - ds.time[0]).item() / 1e9
    window_seconds = pd.Timedelta(window)
    N = int(window_seconds.total_seconds() / dt)
    k = np.unique((np.logspace(-1, np.log10(N / 2), num=200).astype(int)))
    lag = k * dt

    # structure functions
    # resample and loop
    autocorr = []
    for label, group in ds_groups:
        acu = calc_autocorr(group.u, k)
        acv = calc_autocorr(group.v, k)
        acw = calc_autocorr(group.w, k)

        if single_sonic:
            autocorr.append(
                xr.Dataset(
                    coords=dict(lag=lag, time=label),
                    data_vars=dict(
                        acu=(["lag"], acu), acv=(["lag"], acv), acw=(["lag"], acw)
                    ),
                )
            )
        else:
            autocorr.append(
                xr.Dataset(
                    coords=dict(lag=lag, time=label),
                    data_vars=dict(
                        acu=(["lag", "heights"], acu),
                        acv=(["lag", "heights"], acv),
                        acw=(["lag", "heights"], acw),
                    ),
                )
            )

    autocorr = xr.concat(autocorr, dim="time").assign_coords(heights=ds.heights)

    return autocorr


def autocorr_intlen(autocorr, wspd):
    # find the first time the acf crosses 1/e
    intlen = (
        np.abs(autocorr - np.exp(-1))
        .idxmin(dim="lag")
        .rename(dict(acu="intlenU", acv="intlenV", acw="intlenW"))
        * wspd
    )
    return intlen


# Helpers
# -------
def calc_autocorr(var, k):
    N = len(var)
    autocorr = []
    for i in k:
        ac = (var[: N - i].data * var[i:].data).mean(axis=0) / var.var(axis=0).data
        autocorr.append(ac)
    return np.array(autocorr)
