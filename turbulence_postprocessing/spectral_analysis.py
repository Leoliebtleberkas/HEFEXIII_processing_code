import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from scipy import signal
from functions import get_fluctuations


# Main routine
# -------------
def spectra_eps(ds, config, wspd):
    '''works on already rotated data
    At the moment the dissipation calculation for single sonic should work only if there is a height dimension with 1 value'''

    # config
    window = config['window']

    # divide in time blocks
    ds_fluct = get_fluctuations(ds, config)

    # spectra & co
    spectra = spectra_sonic(ds_fluct, window)

    # slopes high and low freq and dissipation rates
    slopes, epsilon = spectral_slopes_epsilon(spectra, wspd)

    return spectra, epsilon, slopes


# Subroutines
# -------------
def spectra_sonic(ds, window):
    '''calculates spectra and cospectra for sonic anemometer dataset (xarray), by window,
        the data must be already rotated and detrended
        supports multiple heights
        Will have a problem if the data length is not a perfect multiple of the window size
        '''
    # check if multiple heights are present
    if len(ds.dims) <= 1:
        single_sonic = True
    else:
        single_sonic = False
        
    #get N observations per window
    dt = np.median(np.diff(ds.time.values).astype('timedelta64[ns]').astype(float)) #in ns
    window_ns = pd.to_timedelta(window).to_numpy().astype('timedelta64[ns]').astype(float)
    n_obs = int(round(window_ns / dt))

    # resample and loop
    ds_groups = ds.resample(time=window, label = "right", closed = "right")
    spectra = []

    # # one sonic
    if single_sonic:
        for label, group in ds_groups:
            
            #safety ceck
            if group.time.size != n_obs:
                print(f"Skipping window {label}, height {group.heights.values} \n"
                      "Group has missing timesteps!!!")
                continue
            
            freq, su = logbin_spectrum(*calc_spectrum(group.u))
            freq, sv = logbin_spectrum(*calc_spectrum(group.v))
            freq, sw = logbin_spectrum(*calc_spectrum(group.w))
            freq, sT = logbin_spectrum(*calc_spectrum(group.tc))

            # cospectra
            freq, cuw = logbin_spectrum(*calc_cospectrum(group.u, group.w))
            freq, cvw = logbin_spectrum(*calc_cospectrum(group.v, group.w))
            freq, cuv = logbin_spectrum(*calc_cospectrum(group.u, group.v))
            freq, cuT = logbin_spectrum(*calc_cospectrum(group.u, group.tc))
            freq, cvT = logbin_spectrum(*calc_cospectrum(group.v, group.tc))
            freq, cwT = logbin_spectrum(*calc_cospectrum(group.w, group.tc))
            

            spectra.append(xr.Dataset(coords=dict(freq=freq, time=label),
                                      data_vars=dict(su=(['freq'], su),
                                                                sv=(['freq'], sv),
                                                                sw=(['freq'], sw),
                                                                sT=(['freq'], sT),
                                                                cuw=(['freq'], cuw),
                                                                cuv=(['freq'], cuv),
                                                                cvw=(['freq'], cvw),
                                                                cwT=(['freq'], cwT),
                                                                cvT=(['freq'], cvT),
                                                                cuT=(['freq'], cuT),)))
        spectra = xr.concat(spectra, dim='time').assign_coords(heights=ds.heights)

    # multiple sonics
    else:
        for label, group in ds_groups:
            single_spectra = []
# =============================================================================
#             if (group.isnull().mean(dim='time') > 0.2).any():
#                 print(f"Too many NaNs in {group.time}")
#                 continue
#             else:
# =============================================================================
            for h in group.heights:
                grouph = group.sel(heights = h)

                #safety ceck
                if grouph.time.size != n_obs:
                    print(f"Skipping window {label}, height {group.heights.values} \n"
                          "Group has missing timesteps!!!")
                    continue
                # spectra
                freq, su = logbin_spectrum(*calc_spectrum(grouph.u))
                freq, sv = logbin_spectrum(*calc_spectrum(grouph.v))
                freq, sw = logbin_spectrum(*calc_spectrum(grouph.w))
                freq, sT = logbin_spectrum(*calc_spectrum(grouph.tc))

                # cospectra
                freq, cuw = logbin_spectrum(*calc_cospectrum(grouph.u, grouph.w))
                freq, cvw = logbin_spectrum(*calc_cospectrum(grouph.v, grouph.w))
                freq, cuv = logbin_spectrum(*calc_cospectrum(grouph.u, grouph.v))
                freq, cuT = logbin_spectrum(*calc_cospectrum(grouph.u, grouph.tc))
                freq, cvT = logbin_spectrum(*calc_cospectrum(grouph.v, grouph.tc))
                freq, cwT = logbin_spectrum(*calc_cospectrum(grouph.w, grouph.tc))

                # put in dataset
                single_spectra.append(xr.Dataset(coords=dict(freq=freq, time=label, heights=h),
                                                 data_vars=dict(su=(['freq'], su),
                                                                sv=(['freq'], sv),
                                                                sw=(['freq'], sw),
                                                                sT=(['freq'], sT),
                                                                cuw=(['freq'], cuw),
                                                                cuv=(['freq'], cuv),
                                                                cvw=(['freq'], cvw),
                                                                cwT=(['freq'], cwT),
                                                                cvT=(['freq'], cvT),
                                                                cuT=(['freq'], cuT),
                                                                )))

            #to dataset
            if single_spectra:
                spectra.append(xr.concat(single_spectra, dim='heights'))

        spectra = xr.concat(spectra, dim='time')
    return spectra


def spectral_slopes_epsilon(spectra, wspd, h=None):
    # calculate cutoff based on height and speed
    if h is None:
        cutoff = wspd / (2 * np.pi * spectra.heights)
    else:
        cutoff = wspd / (2 * np.pi * h)

    # loose cospectra
    spectra = spectra[['su', 'sv', 'sw', 'sT']]

    # HIGH frequency spectra
    # isolate frequencies higher than cutoff, not all the way down because of aliasing
    spectra_high = spectra.where((spectra.freq > cutoff) & (spectra.freq < spectra.freq[-8]))
    # move left limit to the first maximum
    spectra_high = spectra_high.where(spectra_high.freq > spectra_high.idxmax(dim='freq'))

    # LOW frequency spectra
    # isolate frequencies lower than cutoff
    spectra_low = spectra.where(spectra.freq < cutoff)

    # EPSILON
    # kolmogorov constant
    cu = 18 / 55 * 1.5
    cvw = cu * 4 / 3
    cT = 0.8

    epsU = (2 * np.pi / wspd * (spectra_high.freq ** (5 / 3) * spectra_high.su / cu) ** (3 / 2)).median(
        dim='freq').rename('epsU')
    epsV = (2 * np.pi / wspd * (spectra_high.freq ** (5 / 3) * spectra_high.sv / cvw) ** (3 / 2)).median(
        dim='freq').rename('epsV')
    epsW = (2 * np.pi / wspd * (spectra_high.freq ** (5 / 3) * spectra_high.sw / cvw) ** (3 / 2)).median(
        dim='freq').rename('epsW')
    epsT = ((2 * np.pi / wspd) ** (2 / 3) * spectra_high.freq ** (5 / 3) * spectra_high.sT * epsU ** (
            1 / 3) / cT).median(dim='freq').rename('epsT')

    # SLOPES
    # switch to logarithmic space
    spectra_high = np.log10(spectra_high).assign_coords(freq=np.log10(spectra_high.freq))
    spectra_low = np.log10(spectra_low).assign_coords(freq=np.log10(spectra_low.freq))

    # spectral slopes
    slopes_h = spectra_high.polyfit(dim='freq', deg=1).sel(degree=1).drop_vars('degree').rename(
        dict(su_polyfit_coefficients='slopeHU',
             sv_polyfit_coefficients='slopeHV',
             sw_polyfit_coefficients='slopeHW',
             sT_polyfit_coefficients='slopeHT'))

    slopes_l = spectra_low.polyfit(dim='freq', deg=1).sel(degree=1).drop_vars('degree').rename(
        dict(su_polyfit_coefficients='slopeLU',
             sv_polyfit_coefficients='slopeLV',
             sw_polyfit_coefficients='slopeLW',
             sT_polyfit_coefficients='slopeLT'))

    return xr.merge([slopes_h, slopes_l]), xr.merge([epsU, epsV, epsW, epsT])


# Helpers
# -------------
#comment: errors with bigger averaging intervalls --> try n = min(len(var), 1024)
def calc_spectrum(var, dt=None):
    n = choose_nperseg(len(var))
    #noverlap = n//2
    #n = None
    #n = 1024
    noverlap = n // 2
    
    if dt is None:
        dt = (var.time[1] - var.time[0]).item() / 1e9
    #original
    n = len(var.time)
    noverlap = 0
        
    freq, spectrum = signal.welch(var, fs= 1 / dt,
                                  window='boxcar', detrend=False,
                                  nperseg=n, noverlap=noverlap)
    freq = freq[1:]
    spectrum = spectrum[1:].real

    return freq, spectrum


def calc_cospectrum(var1, var2, dt=None):
    n = choose_nperseg(len(var1))
    #noverlap = n // 2
    #n = 256
    #n = None
    #n = 2048
    noverlap = int(0.75*n)

    if dt is None:
        dt = (var1.time[1] - var1.time[0]).item() / 1e9
    #original
    n = len(var1.time)
    noverlap = 0
    window = "boxcar"

    freq, cospectrum = signal.csd(var1, var2, fs=1 / dt,
                                  window='boxcar', detrend=False, #window='hann'
                                  nperseg=n, noverlap=noverlap)
    freq = freq[1:]
    cospectrum = cospectrum[1:].real

    return freq, cospectrum


def logbin_spectrum(freq, spectrum, N_bins=80, stat='mean'): #N_bins = 80
    if stat == 'mean':
        statistic = np.mean
    elif stat == 'median':
        statistic = np.median
    else:
        statistic = stat

    edges = np.logspace(np.log10(freq[0]), np.log10(freq[-1]), N_bins + 1)
    freq_bin = (edges[1:] + edges[:-1]) / 2
    spec_bin = stats.binned_statistic(freq, spectrum, statistic=statistic,
                                      bins=edges).statistic
    return freq_bin, spec_bin


def choose_nperseg(n_samples):
    # Empfohlene 2er-Potenzen
    candidates = np.array([256, 512, 1024, 2048, 4096, 8192, 16384])

    # Potenz wählen, die 4-12 Segmente ergibt
    seg_perms = n_samples / candidates
    valid = (seg_perms >= 4) & (seg_perms <= 12)

    # Nimm die größte mögliche Potenz (bessere Frequenzauflösung)
    if np.any(valid):
        return int(np.min(candidates[valid]))
    else:
        # fallback: kleinste Potenz mit mindestens 2 Segmenten
        fallback = candidates[n_samples / candidates >= 2][0]
        return int(fallback)
    
def nan_fraction(da, vars_needed):
    """Return fraction of NaNs across all required variables."""
    
    total = da.sizes["time"] * len(vars_needed)
    nan_total = sum(da[v].isnull().sum().item() for v in vars_needed)
    return nan_total / total