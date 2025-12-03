#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg') # Prevent qt/wayland errors

# Import libraries
import numpy as np
import numpy.ma as ma
from scipy.signal import argrelextrema
from scipy.stats import genextreme as gev
from netCDF4 import Dataset 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import xarray as xr
import pandas as pd
import os
import sys
import glob
import fnmatch
import datetime
import cftime
import nc_time_axis
import xlrd
import cmocean as ocm
import xesmf as xe
import cartopy.crs as ccrs
import cmaps

# Initialize CF-compliant time decoder
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

# Load freshwater budget components
directory = './data/'
files = [
    'cesmi6gat31rblc230iveq-noeq500tg7701_atm_PRECT_ncea.nc',
    'cesmi6gat31rblc230iveq-noeq500tg7701_atm_QFLX_ncea.nc',
    'cesmi6gat31rblc230iveq-noeq500tg7701_ocn_ROFF_F_ncea.nc'
]

datasets = [xr.open_dataset(os.path.join(directory, file), decode_times=time_coder, decode_timedelta=True) for file in files]

# Create regridder
lon = datasets[0].PRECT.lon
lat = datasets[0].PRECT.lat
grid_in = {'lon': datasets[2].ROFF_F.TLONG, 'lat': datasets[2].ROFF_F.TLAT}
grid_out = {'lon': lon, 'lat': lat}
regridder = xe.Regridder(grid_in, grid_out, 'nearest_s2d', reuse_weights=False)

# Calculate annual means
PRECT_timeseries = datasets[0].PRECT.resample(time='YE').mean()
QFLX_timeseries = datasets[1].QFLX.resample(time='YE').mean()
ROFF_timeseries = regridder(datasets[2].ROFF_F).resample(time='YE').mean()

# Unit conversions
QFLX_timeseries = QFLX_timeseries * 86400
ROFF_timeseries = ROFF_timeseries * 86400
PRECT_timeseries = PRECT_timeseries * 86400 * 1000

# P-E+R calculation
P_E_R_timeseries = PRECT_timeseries - QFLX_timeseries + ROFF_timeseries

# Helper functions
def load_atm_average_variable(directory, variable_name):
    file_path = os.path.join(directory, f"cesmi6gat31rblc230iveq-noeq500tg7701_atm_{variable_name}_ncea.nc")
    dataset = xr.open_dataset(file_path, decode_times=time_coder, decode_timedelta=True)
    variable_timeseries = dataset[variable_name].resample(time='YE').mean()
    return variable_timeseries

def load_ocn_average_variable(directory, variable_name):
    file_path = os.path.join(directory, f"cesmi6gat31rblc230iveq-noeq500tg7701_ocn_{variable_name}_ncea.nc")
    dataset = xr.open_dataset(file_path, decode_times=time_coder, decode_timedelta=True)
    variable_timeseries = regridder(dataset[variable_name]).resample(time='YE').mean()
    return variable_timeseries

def load_ocn_SSS_average(directory, variable_name):
    file_path = os.path.join(directory, f"cesmi6gat31rblc230iveq-noeq500tg7701_ocn_SSS_ncea.nc")
    dataset = xr.open_dataset(file_path, decode_times=time_coder, decode_timedelta=True)
    variable_timeseries = regridder(dataset['SALT']).resample(time='YE').mean()
    return variable_timeseries

def load_ocn_MOC_average(directory, variable_name):
    file_path = os.path.join(directory, f"cesmi6gat31rblc230iveq-noeq500tg7701_ocn_MOC_ncea.nc")
    dataset = xr.open_dataset(file_path, decode_times=time_coder, decode_timedelta=True)
    variable_timeseries = dataset['MOC'].resample(time='YE').mean()
    return variable_timeseries

# Load variables
base_directory = './data/'
tsanom  = load_atm_average_variable(base_directory, 'TS')
peranom = P_E_R_timeseries
pslanom  = load_atm_average_variable(base_directory, 'PSL')
sssanom  = load_ocn_SSS_average(base_directory, 'SALT')
hmxl_enmean  = load_ocn_average_variable(base_directory, 'HMXL')
iceanom  = load_atm_average_variable(base_directory, 'ICEFRAC')
tauyanom = load_atm_average_variable(base_directory, 'TAUY')
bsfanom  = load_ocn_average_variable(base_directory, 'BSF')
tauxanom = load_atm_average_variable(base_directory, 'TAUX')
mocanom  = load_ocn_MOC_average(base_directory, 'MOC').sum(dim='moc_comp').isel(transport_reg=1)

# Averaging
plt_years = [[7701,7705], [7706,7715], [7716,7720]]
def average_over_years(variable, years):
    return variable.sel(time=variable.time.dt.year.isin(range(years[0], years[1]))).mean('time')

ts_ens = []
psl_ens = []
sss_ens = []
ice_ens = []
bsf_ens = []
taux_ens = []
tauy_ens = []
hmxl_ens = []
per_ens = []
moc_ens = []

for years in plt_years:
    ts_ens.append(average_over_years(tsanom, years))
    psl_ens.append(average_over_years(pslanom, years))
    sss_ens.append(average_over_years(sssanom, years))
    ice_ens.append(average_over_years(iceanom, years))
    bsf_ens.append(average_over_years(bsfanom, years))
    taux_ens.append(average_over_years(tauxanom, years))
    tauy_ens.append(average_over_years(tauyanom, years))
    hmxl_ens.append(average_over_years(hmxl_enmean, years))
    per_ens.append(average_over_years(peranom, years))
    moc_ens.append(average_over_years(mocanom, years))

# Plotting
moc_z = mocanom.moc_z
moc_lat = mocanom.lat_aux_grid
nrow      = 3
ncol      = 5
lon_cnt   = 180
alpha     = 0.05
iw        = 1  # quiver subsampling (every 1st grid point)
# contour alpha
calpha    = 0.9

# Define contour levels for each variable
bd_ts     = np.linspace(-4, 4, 17)        # Surface air temp (°C)
bd_psl    = np.linspace(-2.5, 2.5, 11)    # Sea level pressure (KPa)
bd_sss    = np.linspace(-0.5, 0.5, 11)    # Sea surface salinity (g/kg)
bd_hmxl   = np.linspace(-100, 100, 11)    # Mixed layer depth (m)
bd_per    = np.linspace(-1, 1, 11)        # P-E+R (mm/day)
bd_ice    = np.linspace(-20, 20, 11)      # Sea ice fraction (%)
bd_bsf    = np.linspace(-5, 5, 11)        # Barotropic streamfunction (Sv)
bd_moc    = np.linspace(-2, 2, 11)        # AMOC (Sv)
data_proj = ccrs.PlateCarree(central_longitude=180)
fig       = plt.figure(figsize=(16, 12))
gs        = gridspec.GridSpec(nrow, ncol, figure=fig)
cmap      = cmaps.BlueWhiteOrangeRed
extent    = [100, lon[-1]-lon_cnt, 0, 85]  # Map extent: 100°E-180°W, 0-85°N (Northern Hemisphere focus)
tscmap    = cmaps.BlueWhiteOrangeRed      # Diverging colormap for temperature
ssscmap   = ocm.cm.curl                    # Oceanographic colormap for salinity
percmap   = ocm.cm.balance_r               # Balanced colormap for freshwater
bsfcmap   = ocm.cm.delta                   # Colormap for circulation

# label pos
lx = 0.025
ly = 0.915
firstlabel=True

for p in range(nrow):
    iyr, fyr = plt_years[p][0], plt_years[p][1]
    #print(p)
    ## 1. (TS + PSL)
    ax1 = fig.add_subplot(gs[p, 0], projection=data_proj)
    ts_enmean = ts_ens[p]
    psl_enmean = psl_ens[p]/100.
    CTS = ax1.contourf(lon-lon_cnt, lat, ts_enmean,  bd_ts, transform=data_proj, extend='both', cmap=tscmap)
    CPSL_pos = ax1.contour(lon-lon_cnt, lat, psl_enmean.where(psl_enmean > 0), bd_psl[bd_psl > 0], transform=data_proj, colors='red', linewidths=1.0, alpha=calpha)
    CPSL_neg = ax1.contour(lon-lon_cnt, lat, psl_enmean.where(psl_enmean < 0), bd_psl[bd_psl < 0], transform=data_proj, colors='magenta', linewidths=1.0, alpha=calpha)
    ax1.coastlines(linewidth=0.3, color='black', alpha=0.9)
    ax1.set_extent(extent, crs=data_proj)
    gl = ax1.gridlines(draw_labels=True, linewidth=0)
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.top_labels = False
    gl.right_labels = False
    ax1.set_title(f'Years: {iyr}-{fyr}', fontsize=10)        
    ## 2. (SSS + HMXL)
    ax2 = fig.add_subplot(gs[p, 1], projection=data_proj)
    sss_enmean = sss_ens[p]
    hmxl_enmean = hmxl_ens[p]/100.
    CSSS = ax2.contourf(lon-lon_cnt, lat, sss_enmean,  bd_sss, transform=data_proj, extend='both', cmap=ssscmap)
    CHMXL_pos = ax2.contour(lon-lon_cnt, lat, hmxl_enmean.where(hmxl_enmean > 0), bd_hmxl[bd_hmxl > 0], transform=data_proj, colors='yellow', linewidths=1.0, alpha=calpha)
    CHMXL_neg = ax2.contour(lon-lon_cnt, lat, hmxl_enmean.where(hmxl_enmean < 0), bd_hmxl[bd_hmxl < 0], transform=data_proj, colors='cyan', linewidths=1.0, alpha=calpha)
    ax2.coastlines(linewidth=0.3, color='black', alpha=0.9)
    ax2.set_extent(extent, crs=data_proj)
    gl = ax2.gridlines(draw_labels=True, linewidth=0)
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.top_labels = False
    gl.right_labels = False
    ax2.set_title(f'Years: {iyr}-{fyr}', fontsize=10)        
    ## 3. (PER + ICEFRAC)
    ax3 = fig.add_subplot(gs[p, 2], projection=data_proj)
    per_enmean = per_ens[p]
    ice_con = ice_ens[p]*100.0
    CPER = ax3.contourf(lon-lon_cnt, lat, per_enmean,  bd_per, transform=data_proj, extend='both', cmap=percmap)
    CICE_pos = ax3.contour(lon-lon_cnt, lat, ice_con.where(ice_con > 0), bd_ice[bd_ice > 0], transform=data_proj, colors='dimgray', linewidths=1.0, alpha=calpha)
    CICE_neg = ax3.contour(lon-lon_cnt, lat, ice_con.where(ice_con < 0), bd_ice[bd_ice < 0], transform=data_proj, colors='gold', linewidths=1.0, alpha=calpha)
    ax3.coastlines(linewidth=0.3, color='black', alpha=0.9)
    ax3.set_extent(extent, crs=data_proj)
    gl = ax3.gridlines(draw_labels=True, linewidth=0)
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.top_labels = False
    gl.right_labels = False
    ax3.set_title(f'Years: {iyr}-{fyr}', fontsize=10)        
    ## 4. (BSF + TAU)
    ax4 = fig.add_subplot(gs[p, 3], projection=data_proj)
    bsf_enmean = bsf_ens[p]
    taux     = taux_ens[p]
    tauy     = tauy_ens[p]

    CBSF = ax4.contourf(lon-lon_cnt, lat, bsf_enmean,  bd_bsf, transform=data_proj, extend='both', cmap=bsfcmap)
    CQ = ax4.quiver  (lon[::iw]-lon_cnt, lat[::iw], -taux[::iw,::iw], -tauy[::iw,::iw], \
                        transform=data_proj, color='k', scale=0.5, width=0.003)
    ax4.coastlines(linewidth=0.3, color='black', alpha=0.9)
    ax4.set_extent(extent, crs=data_proj)
    gl = ax4.gridlines(draw_labels=True, linewidth=0)
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.top_labels = False
    gl.right_labels = False
    ax4.set_title(f'Years: {iyr}-{fyr}', fontsize=10)        
    ## 5. (MOC)
    ax5 = fig.add_subplot(gs[p, 4])
    moc_enmean = moc_ens[p]
    C5 = ax5.contourf(moc_z/100, moc_lat, moc_enmean.T, bd_moc, extend='both', cmap=cmap)
    ax5.set_title(f'Years: {iyr}-{fyr}', fontsize=10)
    ax5.set_ylim(extent[2], extent[3])
    ax5.set_yticks(np.arange(20, 90, 20))
    ax5.set_yticklabels(['20°N','40°N','60°N','80°N'], fontsize=8)
    ax5.set_xlim(0, 5000)
    ax5.set_xticks([0, 2000, 4000])
    ax5.set_xticklabels(['0', '2000m','4000m'], fontsize=8)
    ax5.grid(linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    if firstlabel:
        # Add text with bounding boxes
        ax1.text(lx, ly, 'A', transform=ax1.transAxes, size=14, weight='bold',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='Square'))
        ax2.text(lx, ly, 'B', transform=ax2.transAxes, size=14, weight='bold',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='Square'))
        ax3.text(lx, ly, 'C', transform=ax3.transAxes, size=14, weight='bold',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='Square'))
        ax4.text(lx, ly, 'D', transform=ax4.transAxes, size=14, weight='bold',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='Square'))
        ax5.text(lx, ly, 'E', transform=ax5.transAxes, size=14, weight='bold',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='Square'))
        firstlabel=False


plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.2)

titles = ['Surface Air Temperature\nMean Sea Level Pressure', 'Sea Surface Salinity\nMixed Layer Depth',
           'Precip - Evap + Runoff\nSea Ice Fraction', 'Barotropic Stream Function\nSurface Wind Stress', 'AMOC']
for col, title in enumerate(titles):
    fig.text(0.17 + col * 0.165, 0.93, title, ha='center', va='center', fontsize=12, fontweight='bold')
bar1y = 0.14
bar2y = 0.088
cbar1 = fig.colorbar(CTS, cax=fig.add_axes([0.105, bar1y, 0.125, 0.01]), orientation='horizontal') # [left, bottom, width, height]
cbar1a = fig.colorbar(CPSL_neg, cax=fig.add_axes([0.10, bar2y, 0.0625, 0.01]), orientation='horizontal') # [left, bottom, width, height]
cbar1b = fig.colorbar(CPSL_pos, cax=fig.add_axes([0.175, bar2y, 0.0625, 0.01]), orientation='horizontal') # [left, bottom, width, height]

cbar2 = fig.colorbar(CSSS, cax=fig.add_axes([0.271, bar1y, 0.125, 0.01]), orientation='horizontal') # [left, bottom, width, height]
cbar2a = fig.colorbar(CHMXL_neg, cax=fig.add_axes([0.265, bar2y, 0.0625, 0.01]), orientation='horizontal') # [left, bottom, width, height]
cbar2b = fig.colorbar(CHMXL_pos, cax=fig.add_axes([0.34, bar2y, 0.0625, 0.01]), orientation='horizontal') # [left, bottom, width, height]

cbar3 = fig.colorbar(CPER, cax=fig.add_axes([0.44, bar1y, 0.125, 0.01]), orientation='horizontal') # [left, bottom, width, height]
cbar3a = fig.colorbar(CICE_neg, cax=fig.add_axes([0.435, bar2y, 0.0625, 0.01]), orientation='horizontal') # [left, bottom, width, height]
cbar3b = fig.colorbar(CICE_pos, cax=fig.add_axes([0.51, bar2y, 0.0625, 0.01]), orientation='horizontal') # [left, bottom, width, height]

cbar4 = fig.colorbar(CBSF, cax=fig.add_axes([0.605, bar1y, 0.125, 0.01]), orientation='horizontal') # [left, bottom, width, height]
ax4.quiverkey(CQ, 0.65, bar2y, 0.05, '0.05 N/m²', labelpos='E', coordinates='figure', fontproperties={'size': 10})

cbar5 = fig.colorbar(C5, cax=fig.add_axes([0.77, bar1y, 0.125, 0.01]), orientation='horizontal') # [left, bottom, width, height]
cfbars = [cbar1, cbar2, cbar3, cbar4, cbar5]
cclbars = [cbar1a, cbar2a, cbar3a]
ccrbars = [cbar1b, cbar2b, cbar3b]
cfunits = ['[°C]', '[g kg$^{-1}$]', '[mm day$^{-1}$]', '[Sv]', '[Sv]']
ccunits = ['[KPa]', '[m]', '[%]']
for x in range(ncol):
    cfbars[x].ax.tick_params(labelsize=10)
    cfbars[x].ax.set_title(cfunits[x], fontsize=10)
    #plt.setp(cfbars[x].ax.get_xticklabels(), rotation=45, ha='right')
for x in range(3):
    cclbars[x].ax.tick_params(labelsize=10)
    cclbars[x].ax.set_title(ccunits[x], fontsize=10, position=(1.1, 1.1))
    plt.setp(cclbars[x].ax.get_xticklabels(), rotation=45, ha='right')
    ccrbars[x].ax.tick_params(labelsize=10)
    #ccrbars[x].ax.set_title(ccunits[x], fontsize=10)

    plt.setp(ccrbars[x].ax.get_xticklabels(), rotation=45,  position=(0.1, 0.02))

plt.savefig('SA_Figure6_recreated.png')
plt.savefig('SA_Figure6_recreated.pdf')

