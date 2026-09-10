This dataset contains the data of the two Lidar systems Halo Streamline XR (https://metek.de/de/product/streamline-xr/) and Metek WindRanger200 (https://metek.de/de/product/wind-ranger-100-200/), which were deployed on Hintereisferner from 06.08.- 01.09.2025, during the third HinterEisFerner EXperiment (HEFEXIII). 

Information about the measurement program of the two devices:

StreamlineXR: 

continuous iteration of 

RHI scan along valley axis (add angle)
RHI scan cross valley (add angles)
VAD scan

The whole cycle took ~5 minutes, meaning that for every scan type the temporal resolution of the raw data is 5 minutes. The measurements were conducted with an initial gate length of 18 m, on the 17.08. the gate length was adjusted to 36 m.



WindRanger200: 

continuous VAD measurements with 13 levels between 7 and 200 m above the surface



overview of data quality control measures

 pre-filtering for Signal-to-noise ratio (SNR, must be > -24) and Consensus Percentage (CNS, must be > 60%)
 physical threshold filtering with the following limitations (Observation is only kept, if none of these thresholds is exceeded): 
u = v = 25 m/s
w = 10 m/s
total hor. wind = 25 m/s  for obs <= 200 m above ground level
total hor. wind = 30 m/s  for obs >   200 m above ground level (only applies to StreamlineXR as WindRanger did not look beyond 200 m AGL

 iterative despiking algorithm based on Aubinet et al. (2012): Remove observation x if its abolute value is outside (n_start + 0.3 * i)  std(x), where n_start is a selectable start value, i is the index of iteration and std(x) is the standard deviation of x. This was done in time with computing std for a window of 1h, centered around every observation and in space with computing std over the 5 levels centered around every observation. n_star was chosen to be 3 for the time and 2 for the space domain.



description of the single datafiles:

StreamlineXR:

Streamline_VAD18(36)m_raw.nc: 5 min VAD scans of the StreamlineXR with 18m gate length (until 17.08) and 36 m (from 17.08), respectively, only pre-processed wth point 1 from the quality control list
Streamline18(36)_cleaned.nc: 5 min fully quality controlled VAD scans of the StreamlineXR
Streamline36_interp.nc: fully quality controlled VAD scans of StreamlineXR, where the data of the first half of the campaing (when gate length was 18 m) were averaged to 36 m and merged with the 36 m gate length data of the second half of the campaign

Windranger200:

Windranger.nc: Raw data of WindRanger200, temporal resolution of 16 seconds
Windranger_cleaned.nc: Fully quality controlled data of WindRanger200, teporal resolution of 16 seconds

combined:

lidar_36m_10min.nc: fully quality controlled and merged data of WindRanger200 and StreamlineXR. For the lower levels (< 2880 m above mean sea level) the data of the Windranger was used (=first 11 levels of Windranger), for the levels above the data of the StreamlineXR, where the first half of campaign was averaged to 36 m gate length. 

All files with an extension of the type "_30min" contain data, which was averaged to the respective time window size, originating from the cleaned (and interpolated) files.