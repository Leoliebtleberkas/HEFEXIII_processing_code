import numpy as np
import xarray as xr
from functions import get_fluctuations
import pandas as pd
print("MRD_functions.py loaded")

# Main routine
# -------------
def multiresolution(ds, config):
    """ works on already rotated data """

    # config
    window = config['window']

    # divide in time blocks
    ds_fluct = get_fluctuations(ds, config)

    # MRDs
    mrd = MRD_sonic(ds_fluct, window)

    return mrd


# Subroutines
# -------------
def MRD_sonic(ds, window, m_ref = None):
    """use detrended data if calling from outside"""

    # check if multiple heights are present
    if len(np.shape(ds.u)) == 1:
        single_sonic = True
    else:
        single_sonic = False
        
    #check if m was properly given in case of 30Hz
    dt = ds["time"].to_dataframe().index.diff().median().total_seconds()
    
    if not m_ref:
        if ~np.isclose(dt, 0.05):
            raise ValueError("automatic calculation of m only works for 20Hz \n"\
                             "give both m_ref (as int) and window as input with \n"\
                                 "window = dt * 2**m_ref closest to desired window size")

    # group by window
    ds_groups = ds.resample(time=window)
    # group length
    #dt = (ds.time[1] - ds.time[0]).values
    #window
    #window_pd = pd.to_timedelta(window)
    #window_np = window_pd.to_numpy()
    #N = int(np.round(np.timedelta64(window_np) / dt))

    # if only one avg period is present N will be None
    #if N is None:
    #    N = len(ds.time)
    mrd = []

    if single_sonic:
        tau = None
        for label, group in ds_groups:
            df = group.drop('heights').to_dataframe().rename(columns={'tc': 't'})
            var = variance(df, ['u', 'v', 'w', 't'], tau, m_ref)
            covar = covariance(df, ['uv', 'uw', 'vw', 'ut', 'vt', 'wt'], tau, m_ref)
            tau = var['Period']

            mrd.append(xr.Dataset(coords=dict(tau=tau, time=label),
                                         data_vars=dict(CUU=(['tau'], var['u']),
                                                        CVV=(['tau'], var['v']),
                                                        CWW=(['tau'], var['w']),
                                                        CTT=(['tau'], var['t']),
                                                        CUV=(['tau'], covar['u-v']),
                                                        CUW=(['tau'], covar['u-w']),
                                                        CVW=(['tau'], covar['v-w']),
                                                        CUT=(['tau'], covar['u-t']),
                                                        CVT=(['tau'], covar['v-t']),
                                                        CWT=(['tau'], covar['w-t']))))
    else:
        tau = None
        for label, group in ds_groups:
            single_mrd = []
            for h in group.heights:
                grouph = group.sel(heights=h)
                df = grouph.to_dataframe().rename(columns={'tc': 't'})
                var = variance(df, ['u', 'v', 'w', 't'], tau, m_ref)
                covar = covariance(df, ['uv', 'uw', 'vw', 'ut', 'vt', 'wt'], tau, m_ref)
                tau = var['Period']

                single_mrd.append(xr.Dataset(coords=dict(tau=tau, heights=h, time=label),
                                             data_vars=dict(CUU=(['tau'], var['u']),
                                                            CVV=(['tau'], var['v']),
                                                            CWW=(['tau'], var['w']),
                                                            CTT=(['tau'], var['t']),
                                                            CUV=(['tau'], covar['u-v']),
                                                            CUW=(['tau'], covar['u-w']),
                                                            CVW=(['tau'], covar['v-w']),
                                                            CUT=(['tau'], covar['u-t']),
                                                            CVT=(['tau'], covar['v-t']),
                                                            CWT=(['tau'], covar['w-t']))))
            mrd.append(xr.concat(single_mrd, dim='heights'))

    mrd = xr.concat(mrd, dim='time').assign_coords(heights=ds.heights)
    return mrd


# Helpers from ECpy, Manuela Lehner
# ---------------------------------
def variance(data, svars, tau_ref, m_ref = None):
    # frequency and period
    nn = len(data)
    #breakpoint()
    #dt = (data.index[1] - data.index[0]).total_seconds()
    dt = data.index.diff().median().total_seconds()
    
    if m_ref is None:
        m = int(np.log2(nn))
    else:
        m = int(m_ref)
    if tau_ref is None:
        prd = np.array([dt * 2 ** im for im in range(m, 0, -1)])
    else:
        prd = tau_ref
    print(f"variance {prd}")
    frq = 1 / prd

    # create dataframe
    mrd = pd.DataFrame(data=np.vstack((frq, prd)).T, columns=['Frequency', 'Period'])

    # MRD
    data = data[svars].iloc[:2 ** m]
    data = data - data.mean()

    mrdaux = np.empty((len(prd), len(svars))) + np.nan
    for ii, im in enumerate(range(m - 1, -1, -1)):
        mnprd = pd.Timedelta(str(dt * 2 ** im) + 's')
        if float(pd.__version__[:3]) >= 1.1:
            toffset = data.index[0] - pd.to_datetime(data.index[0].date())
            mn = data.resample(mnprd, offset=toffset).mean()
        else:
            toffset = ((data.index[0] - pd.to_datetime(data.index[0].date())) /
                       pd.Timedelta('1' + pd.Timedelta(mnprd).resolution_string))
            mn = data.resample(mnprd, base=toffset).mean()
        mrdaux[ii, :] = (mn ** 2).mean()
        data = data - mn.reindex(data.index).ffill()

    mrd[svars] = pd.DataFrame(data=mrdaux)

    return mrd


def covariance(data, cvars, tau_ref, m_ref = None):
    # frequency and period
    nn = len(data)
    #dt = (data.index[1] - data.index[0]).total_seconds()
    dt = data.index.diff().median().total_seconds()

    if m_ref is None:
        m = int(np.log2(nn))
    else:
        m = int(m_ref)
        
    if tau_ref is None:
        prd = np.array([dt * 2 ** im for im in range(m, 0, -1)])
    else:
        prd = tau_ref

    print(f"co-variance {prd}")
    frq = 1 / prd

    # create dataframe
    mrd = pd.DataFrame(data=np.vstack((frq, prd)).T, columns=['Frequency', 'Period'])

    # MRD
    covars = [cv[0] + '-' + cv[1] for cv in cvars]
    svars = list(set([cv[0] for cv in cvars] + [cv[1] for cv in cvars]))
    data = data[svars].iloc[:2 ** m]
    # data = data - data.mean()

    mrdaux = np.empty((len(prd), len(cvars))) + np.nan 
    #loop over m --> exponent, start at biggest intervall go to smaller
    for ii, im in enumerate(range(m - 1, -1, -1)):
        #calculate the window size in seconds
        mnprd = pd.Timedelta(str(dt * 2 ** im) + 's')
        #offset
        if float(pd.__version__[:3]) >= 1.1:
            toffset = data.index[0] - pd.to_datetime(data.index[0].date())
            #resample to window size
            mn = data.resample(mnprd, offset=toffset).mean()
        else:
            toffset = ((data.index[0] - pd.to_datetime(data.index[0].date())) /
                       pd.Timedelta('1' + pd.Timedelta(mnprd).resolution_string))
            #resample to window size
            mn = data.resample(mnprd, base=toffset).mean()
        #loop over covariance variables    
        for icv, cv in enumerate(cvars):
            #compute the mean of the current window size and add it to the storage
            #dataframe to the current window size
            mrdaux[ii, icv] = (mn[cv[0]] * mn[cv[1]]).mean()
        #forward fill the average and then substract the average from the 
        #data before going into the next iteration
        data = data - mn.reindex(data.index).ffill()

    mrd[covars] = pd.DataFrame(data=mrdaux)

    return mrd
