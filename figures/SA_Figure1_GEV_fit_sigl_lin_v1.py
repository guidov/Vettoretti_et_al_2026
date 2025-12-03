#!/usr/bin/env python
# coding: utf-8

# # Volcanic Forcing Statistical Analysis - Figure 1

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import genextreme as gev

# Helpers
def setaxesfsize(axl, fontsize, xlabel, ylabel, labsize):
    ticklabelbot = axl.get_xticklabels()
    ticklabelleft = axl.get_yticklabels()
    for labelx in ticklabelbot:
        labelx.set_fontsize(fontsize)
    for labely in ticklabelleft:
        labely.set_fontsize(fontsize)
    
    axl.set_xlabel(xlabel, size=labsize)
    axl.set_ylabel(ylabel, size=labsize)

def estimate_return_level(quantile, loc, scale, shape):
    level = loc + scale / shape * (1 - (-np.log(quantile)) ** (shape))
    return level

# Load Data
# NGRIP
ngrip_t_pedro_2018 = pd.read_csv("./data/Pedro_QSR2018_fig1a.txt", sep='\t', names=["Year", "Temperature"])

# Sigl 2022
sigl2022 = xr.open_dataset("./data/HolVol_volcanic_stratospheric_sulfur_injection_v1.0.nc", engine='netcdf4')
sigl_hol_volc_data = sigl2022 # Alias used in original code

# Lin MIS3
lin_volc_data = pd.ExcelFile("./data/Lin_MIS3_volcanic_supp.xls")
lin_load_data = lin_volc_data.parse('Table S5', header=1)

gicc05_age = lin_load_data[lin_load_data.columns[1]].copy()
volc_load = lin_load_data[lin_load_data.columns[26]].copy()
site_loc = lin_load_data[lin_load_data.columns[31]].copy()

# Fix 2 southern hemisphere eruptions
site_loc[21] = "SH"
site_loc[62] = "SH"

# Prepare Lin Data (MIS3)
mis3age = gicc05_age[20:]
mis3_lin_volc_load = volc_load[20:]

# Prepare Sigl Data (Holocene)
mass = sigl2022.vssi * 3

# Prepare Time Series for Sigl (Plot A)
sigl2022_years = sigl2022.year.values
contmass = (sigl2022.vssi * 3).values
sigl_lats = sigl_hol_volc_data.lat

# Time vector
timevar = np.linspace(-9496+9496, 1892+9496, 11389)

# Create new arrays filled with np.nan
new_contmass = np.full_like(timevar, np.nan)
new_latitude = np.full_like(timevar, np.nan)

# Find the indices in timevar that correspond to sigl2022_years
indices = np.searchsorted(timevar, sigl2022_years + 9496)

# Fill in the values
new_contmass[indices] = contmass
new_latitude[indices] = sigl_lats

# --- Analysis for Plot C (Raw Data GEV Fits) ---

# Sigl
sigl_shape, sigl_loc, sigl_scale = gev.fit(mass)
# Lin
lin_shape, lin_loc, lin_scale = gev.fit(mis3_lin_volc_load)


# --- Analysis for Plot D (Return Levels from Rebinned Data) ---

# 1. Rebin Lin Data
df_lin = pd.DataFrame({'Time': mis3age, 'Load': mis3_lin_volc_load})
num_bins_lin = 23
df_lin['Bin'] = pd.cut(df_lin['Time'], bins=num_bins_lin)
lin_rebinned_df = df_lin.groupby('Bin', observed=False).agg(MaxLoad=('Load', 'max'))
lin_new_df = lin_rebinned_df[2:] # logic from In[15]

lin_ml_shape, lin_ml_loc, lin_ml_scale = gev.fit(lin_new_df["MaxLoad"].values)

# 2. Rebin Sigl Data
df_sigl = pd.DataFrame({'Time': timevar, 'Load': new_contmass})
num_bins_sigl = 242
df_sigl['Bin'] = pd.cut(df_sigl['Time'], bins=num_bins_sigl)
sigl_rebinned_df = df_sigl.groupby('Bin', observed=False).agg(MaxLoad=('Load', 'max'))
sigl_new_rebinned_df = sigl_rebinned_df[1:] # logic from In[66]

sigl_ml_shape, sigl_ml_loc, sigl_ml_scale = gev.fit(sigl_new_rebinned_df["MaxLoad"].values)

# 3. Calculate Return Levels
sigl_periods = np.linspace(1.1, 130, 1000)
lin_periods = np.linspace(1.1, 13, 100)

sigl_quantiles = 1 - 1 / sigl_periods
lin_quantiles = 1 - 1 / lin_periods

sigl_tscale = 47.06 # years in each maximum bin (from In[85])
lin_tscale = 1520.8 # years in each maximum bin (from In[85])

sigl_levels = estimate_return_level(sigl_quantiles, sigl_ml_loc, sigl_ml_scale, sigl_ml_shape)
lin_levels = estimate_return_level(lin_quantiles, lin_ml_loc, lin_ml_scale, lin_ml_shape)


# --- Final Plotting ---

# Set up the figure and gridspec
fig = plt.figure(figsize=(16, 13))  # Adjusted size to accommodate all plots
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])


# Common settings
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)

###################################################################################################
# First plot (wide) - Subplot A
ax1 = fig.add_subplot(gs[0, :])
t_shift = 2000 - 1892
t_start = -11389 + 1892
eleg, nleg, sleg = True, True, True
for e in range(len(timevar)):
    if new_latitude[e] > 20.0:
        if nleg:
            ax1.plot([t_start + timevar[e], t_start + timevar[e]], [0, new_contmass[e]], color='b', label="Northern Hemisphere Extratropics")
            nleg=False
        else:
            ax1.plot([t_start + timevar[e], t_start + timevar[e]], [0, new_contmass[e]], color='b')
    elif new_latitude[e] >= -20 and new_latitude[e] <= 20:
        if eleg:
            eleg=False
            ax1.plot([t_start + timevar[e], t_start + timevar[e]], [0, new_contmass[e]], color='k', label="Equatorial")
        else:
            ax1.plot([t_start + timevar[e], t_start + timevar[e]], [0, new_contmass[e]], color='k')
    elif new_latitude[e] < -20.:
        if sleg:
            sleg=False
            ax1.plot([t_start + timevar[e], t_start + timevar[e]], [0, new_contmass[e]], color='r', label="Southern Hemisphere Extratropics")
        else:
            ax1.plot([t_start + timevar[e], t_start + timevar[e]], [0, new_contmass[e]], color='r')

ax1.legend(fontsize=14)
ax1.set_title("Holocene Volcanic Eruption Data", loc='left', size=18)
setaxesfsize(ax1,14,r'Time ($Year~CE$)',r'Stratospheric Load ($Tg$)',16)
ax1.set_ylim(0,700)

########################################################################################################
# Second plot (wide) - Subplot B
ax2 = fig.add_subplot(gs[1, :])
eleg, nleg, sleg = True, True, True
for e in range(len(gicc05_age)):
    if site_loc[e] == "NHHL":
        if nleg:
            ax2.plot([-gicc05_age[e],-gicc05_age[e]],[0,volc_load[e]],color='b', label="NH Extratropics")
            nleg=False
        else:
            ax2.plot([-gicc05_age[e],-gicc05_age[e]],[0,volc_load[e]],color='b')
    elif site_loc[e] == "LL or SH":
        if eleg:
            ax2.plot([-gicc05_age[e],-gicc05_age[e]],[0,volc_load[e]],color='k', label="Equatorial")
            eleg=False
        else:
            ax2.plot([-gicc05_age[e],-gicc05_age[e]],[0,volc_load[e]],color='k')
    elif site_loc[e] == "SH":
        if sleg:
            ax2.plot([-gicc05_age[e],-gicc05_age[e]],[0,volc_load[e]],color='r', label="SH Extratropics")
            sleg=False
        else:
            ax2.plot([-gicc05_age[e],-gicc05_age[e]],[0,volc_load[e]],color='r')

ax2.set_ylim(0,700)
ax2.set_xlim(-60000,-24000)
ax2.legend(fontsize=14, ncol=3)
ax2.set_title("Marine Isotope Stage 3 (MIS3) Volcanic Eruption Data and North Greenland Ice Core Project (NGRIP) Temperature", loc='left', size=18)
setaxesfsize(ax2,14,r'Time ($Year~BP$)',r'Stratospheric Load ($Tg$)', 16)

# add in NGRIP temperature data
ax2r = ax2.twinx()
ax2r.plot(-1.0*ngrip_t_pedro_2018["Year"],ngrip_t_pedro_2018["Temperature"], color='green', linewidth=1.0, label='NGRIP Temperature')
setaxesfsize(ax2r,14,'',r'Temperature (deg C)', 16)
ax2r.set_ylim(-60,-25)
ax2r.legend(fontsize=14, ncol=1, loc=(0.802,0.67))

## In this subplot we will add horizontal lines at 230, 116, 60 Tg SO4 to show the simulated levels
simlw = 0.5
ax2.axhline(y=230, color='#222222', linestyle='--', label='230 Tg', linewidth=simlw*3)
ax2.axhline(y=116, color='#444444', linestyle='--', label='115 Tg', linewidth=simlw*1.5)
ax2.axhline(y=58, color='#666666', linestyle='--', label='58 Tg', linewidth=simlw)
ax2.legend(fontsize=14, ncol=3)

###################################################################################################
# Third plot (square) - Subplot C
ax3 = fig.add_subplot(gs[2, 0])
hist_sigl, bins_sigl = np.histogram(mass, bins=31, range=(0, 600), density=True)
ax3.bar(bins_sigl[:-1], hist_sigl, width=10, align='edge', alpha=0.3, color='red', label="Holocene data")
hist_lin, bins_lin = np.histogram(mis3_lin_volc_load, bins=31, range=(0, 600), density=True)
ax3.bar(bins_lin[:-1], hist_lin, width=10, align='edge', alpha=0.3, color='blue', label="MIS3 data")
ax3.set_ylabel("Normalized Occurrences")
ax3.set_xlabel(r"Stratospheric Load ($Tg$)", size=16)
setaxesfsize(ax3, 14, r"Stratospheric Load ($Tg$)", "Normalized Occurrences", 16)
ax3.set_ylim(0, 0.01)
ax3.legend(fontsize=14, ncol=2)

# Calculate GEV fit lines for plot C (Raw data fits)
l_sigl = sigl_loc + sigl_scale / sigl_shape
sigl_xx = np.linspace(l_sigl+0.00001, l_sigl+0.00001+600, num=71)
sigl_yy = gev.pdf(sigl_xx, sigl_shape, sigl_loc, sigl_scale)
ax3.plot(sigl_xx, sigl_yy, 'red', linewidth = 1.5, label = "Sigl GEV fit")

l_lin = lin_loc + lin_scale / lin_shape
lin_xx = np.linspace(l_lin+0.00001, l_lin+0.00001+600, num=71)
lin_yy = gev.pdf(lin_xx, lin_shape, lin_loc, lin_scale)
ax3.plot(lin_xx, lin_yy, 'blue', linewidth = 1.5, label = "Lin GEV fit")

ax3.set_title("Generalized Extreme Value Distributions", loc='left', size=18)

# Add text box with parameters
sigl_params = sigl_shape, sigl_loc, sigl_scale
lin_params = lin_shape, lin_loc, lin_scale
bbox = dict(boxstyle="round", color="white", linewidth=2)
## scipy uses 1/c instead of -1/xi , so we have to multiply by -1
ax3.text(0.5, 0.61, r'xi=%.2f' '\n' r'mu=%.2f' '\n' r'sigma=%.2f' % (-1.0*sigl_params[0], sigl_params[1], sigl_params[2]),
         transform=ax3.transAxes, bbox=bbox, color='red', size=14)
ax3.text(0.82, 0.61, r'xi=%.2f' '\n' r'mu=%.2f' '\n' r'sigma=%.2f' % (-1.0*lin_params[0], lin_params[1], lin_params[2]),
         transform=ax3.transAxes, bbox=bbox, color='blue', size=14)
###############################################################################################################################

# Fourth plot (square) - Subplot D
ax4 = fig.add_subplot(gs[2, 1])

ax4.plot(sigl_periods*sigl_tscale, sigl_levels, "-", color='red', label='Holocene Data')
ax4.plot(lin_periods*lin_tscale, lin_levels, "-", color='blue', label='MIS3 Data')
ax4.set_xlabel("Return Period (years)")
ax4.set_ylabel("Return Level (Tg)")
ax4.set_ylim(0,500)
ax4.set_xlim(0,10000)
ax4.grid(True)
setaxesfsize(ax4,14,r"Return Period ($Years$)",r"Return Level ($Tg$)",16)
ax4.set_title("Holocene and MIS3 Return Times", loc='left', size=18)
ax4.legend(fontsize=14, ncol=2)

# Adjust the layout
plt.tight_layout()

## Add a horizontal line going from 230, 116, 60 Tg SO4 on the y-axis to the y value on the red curve 
## Add a vertical line going from 0 on the x-axis to y value on the red curve at 230, 116, 60 Tg SO4 on the y-axis
## I need it to stop when the lines intercept the red curve

# Find the x-values where the horizontal lines intersect the red curve
x_230 = np.interp(230, sigl_levels, sigl_periods*sigl_tscale)
x_116 = np.interp(116, sigl_levels, sigl_periods*sigl_tscale)
x_58 = np.interp(58, sigl_levels, sigl_periods*sigl_tscale)

# Horizontal lines
ax4.plot([0, x_230], [230, 230], color='#222222', linestyle='--', linewidth=simlw*3)
ax4.plot([0, x_116], [116, 116], color='#444444', linestyle='--', linewidth=simlw*1.5)
ax4.plot([0, x_58], [58, 58], color='#666666', linestyle='--', linewidth=simlw)

# Vertical lines
ax4.plot([x_230, x_230], [0, 230], color='#222222', linestyle='--', linewidth=simlw*3)
ax4.plot([x_116, x_116], [0, 116], color='#444444', linestyle='--', linewidth=simlw*1.5)
ax4.plot([x_58, x_58], [0, 58], color='#666666', linestyle='--', linewidth=simlw)

# Add intersection points
ax4.plot(x_230, 230, 'ro')
ax4.plot(x_116, 116, 'ro')
ax4.plot(x_58, 58, 'ro')

# add a,b,c,d
lfsize = 24
ax1.text(-0.05, 1.05, 'A',
         transform=ax1.transAxes, size=lfsize, weight='bold')
ax2.text(-0.05, 1.05, 'B',
         transform=ax2.transAxes, size=lfsize, weight='bold')
ax3.text(-0.13, 1.05, 'C',
         transform=ax3.transAxes, size=lfsize, weight='bold')
ax4.text(-0.13, 1.05, 'D',
         transform=ax4.transAxes, size=lfsize, weight='bold')

# Save the plot
icepaperdir = "./"
saveplot = True

if saveplot:
    plt.savefig(icepaperdir+"Figure1.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
    plt.savefig(icepaperdir+"Figure1.pdf", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)

plt.show()
