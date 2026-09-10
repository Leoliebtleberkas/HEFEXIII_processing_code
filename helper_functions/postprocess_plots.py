# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 16:27:06 2025

@author: leopo
"""

import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from pathlib import Path

#%% plot functions for evaluating post processing
#%% plot spectra functions
def plot_freq_spectra(
        data, spectra_var, ylabel = "m^2/s^2 * Hz^(-1)", spectra_times_freq = False,
        kolmogorovslope = True, ax = None, figsize = (10, 5),
        color = None, label = None, linestyle = "solid", savepath = None
):
    
    #plot only finite values
    data = data.where(np.isfinite(data[spectra_var]), drop = True)
    #data = data.where(data[spectra_var] > 1e-6, drop = True)
    
    if label == None:
        label = spectra_var
    if ax == None:
        fig, ax = plt.subplots(figsize = figsize)
    #premultiplied with frequency
    if spectra_times_freq:
        plt.loglog(data.freq, data.freq*data[spectra_var], label = label,
                   color = color, linestyle = linestyle, linewidth = 2)
        
        #-2/3 slope
        slope = -2/3
        #f_ref = np.array([0.1, 2])  # pick frequency range inside inertial subrange
        #A = 0.01  # arbitrary scaling factor
        #plt.loglog(f_ref, A * f_ref**(-2/3), 'k--', label=r'$f^{-2/3}$')
        
    else:
        plt.loglog(data.freq, data[spectra_var], label = label,
                   color = color, linestyle = linestyle, linewidth = 2)
        
        #Kolmogorov -5/3 slope
        slope = -5/3
        #f_ref = np.array([0.1, 2])  # pick frequency range inside inertial subrange
        #A = 0.01  # arbitrary scaling factor
        #plt.loglog(f_ref, A * f_ref**(-5/3), 'k--', label=r'$f^{-5/3}$')
        
    #Kolmogorov slope as reference
    kolmogorov_slope(slope, ax)
    
        
    #limits
    y_min = 10 ** np.floor(np.log10(data[spectra_var].min()))
    y_max = 10 ** np.ceil(np.log10(data[spectra_var].max()))
    ax.set_ylim(y_min, y_max)
    
    #labels
    ax.set_title(f"{spectra_var}")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(ylabel)
    
    ax.legend(loc = "lower left")
    
    if savepath:
        plt.savefig(savepath)
    
    return slope

def plot_freq_spectra_timecolors(
        data: xr.Dataset, spectra_var: str, 
        height = 3, 
        ylabel="m^2/s^2 * Hz^(-1)", spectra_times_freq=False,
        kolmogorovslope=True, ax=None, figsize=(12,6),
        cmap="viridis", linestyle="solid", savepath=None
):
    """
    Plots frequency spectra aus einem xarray Dataset über die Zeit.
    Resampled auf 1-Stunden-Mittelwerte entlang der time-Dimension.
    Farbcode zeigt die Zeit an.
    """
    # 1-Stunden Resample
    data_resampled = data.resample(time="1h").mean()
    
    # Plot-Setup
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Convert times to pandas.Timestamp for color mapping
    times_pd = pd.to_datetime(data_resampled.time.values)
    norm = plt.Normalize(times_pd.min().value, times_pd.max().value)
    colormap = cm.get_cmap(cmap)
    
    for t_idx, t in enumerate(data_resampled.time.values):
        spectrum = data_resampled[spectra_var].sel(time=t)
        
        # nur finite Werte
        freq = data_resampled.freq
        y = spectrum
        mask = np.isfinite(y)
        if spectra_times_freq:
            y = np.abs(freq.values * y.values)
        else:
            y = np.abs(y.values)
        
        ax.loglog(freq.values[mask], y[mask], 
                  color=colormap(norm(np.datetime64(t).astype(float))),
                  linestyle=linestyle, linewidth=1.5)
    
    # Kolmogorov Referenzlinie
    if kolmogorovslope:
        slope = -5/3 if not spectra_times_freq else -2/3
        kolmogorov_slope(slope, ax)
    
    # Limits automatisch auf nächste Zehnerstelle
    y_all = data_resampled[spectra_var].values.flatten()
    #y_all = y_all[np.isfinite(y_all)]
    y_all = y_all[np.isfinite(y_all) & (y_all > 0)]
    if y_all.size == 0:
        y_min, y_max = 1e-4, 1e0
    else:
        if spectra_times_freq:
            y_all *= freq.values[0]  # falls Frequenzmultiplikation nötig
        if len(y_all) > 0:
            y_min = 10 ** np.floor(np.log10(np.nanmin(y_all)))
            y_max = 10 ** np.ceil(np.log10(np.nanmax(y_all)))
            ax.set_ylim(y_min, y_max)
    
    # Labels
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(ylabel)
    ax.set_title(spectra_var)

    # Format colorbar ticks
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    tick_times = pd.date_range(start=times_pd.min().normalize(),
                           end=times_pd.max(),
                           freq="3h")
    tick_times_int = tick_times.view("int64")
    #tick_times = cbar.get_ticks()
    #tick_times = pd.to_datetime(tick_times)
    cbar.set_ticks(tick_times_int)
    cbar.set_ticklabels([t.strftime("%d.%m %H:%M") for t in tick_times])
    cbar.set_label("Time")
    
    ax.legend(loc = "lower left")
    
    if savepath:
        file = f"{spectra_var}_{height}.png"
        savepath = savepath / file
        plt.savefig(savepath)
        plt.close()
    
    return ax
    
# =============================================================================
# def kolmogorov_slope(
#         slope, ax      
# ):
#     
#     if slope == -2/3:
#         label_slope = r'$f^{-2/3}$'
#     elif slope == -5/3:
#         label_slope = r'$f^{-5/3}$'
#     
#     # Achsenlimits holen
#     x_min, x_max = ax.get_xlim()
#     
#     # Fixe y-Werte, bei denen Linien starten sollen
#     #y_starts = [0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10]  
#     
#     y_starts = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
#     for y0 in y_starts:
#         f_ref = np.array([x_min, x_max])
#         y_ref = y0 * (f_ref / x_min) ** slope
#         ax.loglog(f_ref, y_ref, 'k--', linewidth=1.0)
# 
#     # Dummy für Legende
#     ax.loglog([], [], 'k--', label=label_slope)
# =============================================================================
    
def kolmogorov_slope(
        slope, ax, y_starts = None, swapped_axes = False     
):
    
    if slope == -2/3:
        label_slope = r'$f^{-2/3}$'
    elif slope == -5/3:
        label_slope = r'$f^{-5/3}$'
    
    # Achsenlimits holen
    if swapped_axes:
        x_min, x_max = ax.get_ylim()
    else:
        x_min, x_max = ax.get_xlim()
    
    # Fixe y-Werte, bei denen Linien starten sollen
    #y_starts =  
    if y_starts is None:
        y_starts = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    else:
        y_starts = y_starts

    for y0 in y_starts:
        f_ref = np.array([x_min, x_max])
        y_ref = y0 * (f_ref / x_min) ** slope
        if swapped_axes:
            ax.loglog(y_ref, f_ref, 'r--', linewidth=1.0)
        else:
            ax.loglog(f_ref, y_ref, 'r--', linewidth=1.0)

    # Dummy für Legende
    ax.loglog([], [], 'r--', label=label_slope)
    
    
#%% plot despiking

def plot_despiking_corrections(
        ds_raw, ds_despiked, nan_dict, orig_data_len = None,
        variables=["u", "v", "w", "Ts"], n_std=4,
        sensor_name="metek_05m", output_path=None, window="30min",
        background_step=50
):
    """
    Plot despiking diagnostics:
    - all outliers fully plotted
    - background data downsampled
    """

    nrows = len(variables)
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 15), sharex=True)

    if nrows == 1:
        axs = [axs]

    # time as numpy array (once)
    t = ds_raw.time.values

    for ax, var in zip(axs, variables):

        # --- NumPy arrays (FAST) ---
        y_raw = ds_raw[var].values
        y_dsp = ds_despiked[var].values

        mean = ds_despiked[f"{var}_mean"].values
        std  = ds_despiked[f"{var}_std"].values

        upper = mean + std
        lower = mean - std

        # --- Outlier mask ---
        outlier = (y_raw > upper) | (y_raw < lower)
        
        # --- large value mask ---
        #upper_perc = np.nanpercentile(y_dsp, 99.9)
        #lower_perc = np.nanpercentile(y_dsp, 0.1)
        #large_value = (y_dsp > upper_perc) | (y_dsp < lower_perc)

        # --- Indices ---
        idx_all = np.arange(len(y_raw))
        idx_bg  = idx_all[~outlier][::background_step]   # background
        idx_out = idx_all[outlier]                        # ALL outliers
        #idx_large = idx_all[large_value]


        # --- Background despiked data (downsampled) ---
        ax.scatter(
            t[idx_bg],
            y_dsp[idx_bg],
            s=1,
            alpha=0.25,
            color="r",
            label="despiked (downsampled)",
            rasterized=True
        )

        # --- Outliers (FULL resolution) ---
        if idx_out.size > 0:
            ax.scatter(
                t[idx_out],
                y_raw[idx_out],
                s=14,
                color="orange",
                edgecolor="k",
                linewidth=0.3,
                label="outliers",
                rasterized=True
            )

        # --- Large values (FULL resolution) ---
        #if idx_large.size > 0:
        #    ax.scatter(
        #        t[idx_large],
        #        y_raw[idx_large],
        #        s=10,
        #        color="blue",
        #        edgecolor="k",
        #        linewidth=0.3,
        #        label="large values",
        #        rasterized=True
         #   )

        # --- Mean & std band ---
        ax.plot(t, mean, color="k", linewidth=1, label="period mean")

        band_step = background_step * 2

        idx_band = np.arange(0, len(t), band_step)
        
        ax.fill_between(
            t[idx_band],
            lower[idx_band],
            upper[idx_band],
            color="gray",
            alpha=0.25,
            label=f"±{n_std}σ",
            rasterized=True
        )

        # --- NaN info ---
        freq_nans = np.round(
            100 * nan_dict[var] / orig_data_len, decimals=2
        )
        ax.text(
            0.02, 0.05,
            f"Total Nans produced: {nan_dict[var]} = {freq_nans} %",
            transform=ax.transAxes
        )

        ax.set_title(var)
        ax.legend(loc="upper right")

    plt.tight_layout()

    if output_path:
        plt.savefig(
            output_path / f"{sensor_name}_despiking_{window}.png",
            bbox_inches="tight",
            dpi=150
        )

    #clear memory
    plt.close(fig)
    del fig, axs

    
#%% timeseries of fluctuations fluxes etc.
def plot_timeseries_multiple(
        ds, data_vars, units, figsize = (12, 6), colors = [None], 
        labels = [None], savepath = None
):
    
    if None in colors:
        cmap = plt.get_cmap("tab10")   # oder "tab20", "viridis", "plasma", etc.
        n = len(ds.heights)
        colors = [cmap(i / n) for i in range(n)]
    
    fig, axs = plt.subplots(len(data_vars), figsize = figsize)
        
    if labels is None or len(labels) != len(data_vars):
        labels = data_vars

    for i, var in enumerate(data_vars):
        
        if type(axs) == np.ndarray:
            ax = axs[i]
        else: 
            ax = axs
            
        flux_data = ds[var]
        for j, h in enumerate(flux_data.heights.values):
            flux = flux_data.sel(heights = h)
            ax.plot(flux.time, flux, color = colors[j], label = f"{h} m")
            if i == 0:
                ax.legend()
        
        ax.set_title(labels[i])
        ax.set_ylabel(f"[{units[i]}]")
        #0-line
        ax.axhline(0, color = "k", linestyle = "dashed")
        
        ax.label_outer()
        
    if savepath:
        plt.savefig(savepath)
        plt.close()
        
def plot_timeseries_single(
        ds, data_vars, units, figsize = (12, 6), colors = [None], 
        labels = [None], savepath = None
):
    
    if None in colors:
        cmap = plt.get_cmap("tab10")   # oder "tab20", "viridis", "plasma", etc.
        n = len(ds.heights)
        colors = [cmap(i / n) for i in range(n)]
    
    fig, axs = plt.subplots(len(data_vars), figsize = figsize)
        
    if labels is None or len(labels) != len(data_vars):
        labels = data_vars

    for i, var in enumerate(data_vars):
        
        if type(axs) == np.ndarray:
            ax = axs[i]
        else: 
            ax = axs
            
        flux_data = ds[var]
        ax.plot(flux_data.time, flux_data, color = "k")
        ax.legend()
        
        ax.set_title(labels[i])
        ax.set_ylabel(f"[{units[i]}]")
        #0-line
        ax.axhline(0, color = "k", linestyle = "dashed")
        
        ax.label_outer()
        
    if savepath:
        plt.savefig(savepath)
        plt.close()
        
        
#%% circular plot
def plot_w_circular(
        wdir, w, output_plot    
):
    
    fig, ax = plt.subplots(subplot_kw = {"projection": "polar"})
    ax.scatter(wdir, w, s = 2)
    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(-10, 10)
    
    plt.savefig(output_plot)
    plt.close()
    
#%% wavelet spectrum overview
def plot_wavelet_overview(
        ds, original_signal, reproduced_signal, 
        freq, scale, variance, 
        power_wavelet, significance95, 
        glbl_power, glbl_signif, fft_freqs, fft_theor, fft_power,
        title, label, units, cmap = "turbo", output = None,
        f_mcnider = None
):
    dt_time = np.timedelta64(50, "ms")
    fig = plt.figure(figsize = (11,8))

    #original time series anomaly and inverse wavelet transform

    ax = plt.axes([0.1, 0.75, 0.65, 0.2])
    ax.plot(ds.time, original_signal, "k", linewidth=1.5, label = "original detrended data")
    ax.plot(ds.time, np.real(reproduced_signal), "--", 
            linewidth=1, color="grey", label = "inverse cwt")
    ax.axhline(y = 0, linestyle = "--", color = "grey")
    ax.legend()
    ax.set_title("a) {}".format(title))
    ax.set_ylabel(r"{} [{}]".format(label, units))

    #wavelet power spectrum
    #vmin and vmax for LogNorm/colorbar 
    vmin = np.nanmin(power_wavelet[power_wavelet>0]*variance)
    vmax = np.nanmax(power_wavelet*variance)
    vmin = 1e-4
    norm = LogNorm(vmin=vmin, vmax = vmax)
    bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
    pcm = bx.pcolormesh(ds.time.values, freq, power_wavelet*variance,
                        cmap=cmap, norm = norm)
    #add colorbar
    cbar_ax = fig.add_axes([0.1, 0.3, 0.65, 0.01])
    cbar = fig.colorbar(pcm, cax = cbar_ax, orientation = "horizontal")
    cbar.set_label(r"Power [({})^2]".format(units))
    #cbar = plt.colorbar(pcm, ax=bx, orientation="vertical")
    #cbar.set_label("m/s")

    #significant structures
    extent = [ds.time.min(), ds.time.max(), 0, min(freq)]
    bx.contour(
        ds.time, freq, significance95, [-99, 1], colors="k", linewidths=1, extent=extent
    )
    bx.set_title("b) {} Wavelet Power Spectrum (Morlet)".format(label))

    #logarithmic and reversed axes
    bx.set_yscale("log")
    bx.yaxis.set_inverted(True)

    bx.set_ylabel("frequency [Hz]")
    bx.set_xlabel("time")

    # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
    # noise spectra. Note that period scale is logarithmic.
    cx = plt.axes([0.8, 0.37, 0.2, 0.28], sharey=bx)
    #significane
    cx.plot(glbl_signif, freq, "k--", label = "95sig wavelet")
    cx.plot(variance * fft_theor, freq, "--", color="grey", label = "95sig fft")
    #global wavelet spectrum and fourier spectrum
    cx.plot(
        variance * fft_power, fft_freqs, color="grey", linewidth=1.0, label = "fft")
    cx.plot(variance * glbl_power, freq, "k", linewidth=1.5, label = "wavelet")
    #McNider frequency
    if f_mcnider is not None:
        cx.axhline(f_mcnider, color = "k", 
                   linestyle = "-.", label = "McNider")

    cx.set_title("c) Global Wavelet Spectrum")
    cx.set_xlabel(r"Power [({})^2]".format(units))

    #logarithmic x axis
    cx.set_xscale("log")

    #kolmogorov slope
    y_starts = [1e-7, 1e-6, 1e-4, 1e-3, 0.001, 0.01, 0.1]
    kolmogorov_slope(slope = -5/3, ax = cx, swapped_axes=True, 
                     y_starts=y_starts)

    #legend
    cx.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.4), 
              ncol = 2, fontsize = 7)
    #xaxis limit
    cx.set_xlim(1e-6, 5e2)

    plt.setp(cx.get_yticklabels(), visible=False)

    if output:
        plt.savefig(output, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()