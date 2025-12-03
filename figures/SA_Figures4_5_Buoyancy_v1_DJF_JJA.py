#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg') # Prevent Qt warning and use non-interactive backend

import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import gridspec
from xclim import ensembles
import gsw
import xesmf as xe
from dask.distributed import Client
import warnings

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    # Initialize Dask Client
    client = Client()
    print(f"Dask dashboard link: {client.dashboard_link}")

    # Load Data
    print("Loading data...")
    SSS_volc = ensembles.create_ensemble(glob.glob('./data/cesmi6gat31rblc230iveq500tg7701e0??_ocn_SSS.nc'), use_cftime=True).squeeze()
    SSS_novolc = ensembles.create_ensemble(glob.glob('./data/cesmi6gat31rblc230ivnoeq500tg7701e0??_ocn_SSS.nc'), use_cftime=True).squeeze()
    SST_volc = ensembles.create_ensemble(glob.glob('./data/cesmi6gat31rblc230iveq500tg7701e0??_ocn_SST.nc'), use_cftime=True).squeeze()
    SST_novolc = ensembles.create_ensemble(glob.glob('./data/cesmi6gat31rblc230ivnoeq500tg7701e0??_ocn_SST.nc'), use_cftime=True).squeeze()
    SHF_volc = ensembles.create_ensemble(glob.glob('./data/cesmi6gat31rblc230iveq500tg7701e0??_ocn_SHF.nc'), use_cftime=True).squeeze()
    SHF_novolc = ensembles.create_ensemble(glob.glob('./data/cesmi6gat31rblc230ivnoeq500tg7701e0??_ocn_SHF.nc'), use_cftime=True).squeeze()
    SFWF_volc = ensembles.create_ensemble(glob.glob('./data/cesmi6gat31rblc230iveq500tg7701e0??_ocn_SFWF.nc'), use_cftime=True).squeeze()
    SFWF_novolc = ensembles.create_ensemble(glob.glob('./data/cesmi6gat31rblc230ivnoeq500tg7701e0??_ocn_SFWF.nc'), use_cftime=True).squeeze()

    # Calculate ensemble means
    print("Calculating ensemble means...")
    SSS_volc_ensmean = SSS_volc.mean(dim='realization').compute()
    SST_volc_ensmean = SST_volc.mean(dim='realization').compute()
    SHF_volc_ensmean = SHF_volc.mean(dim='realization').compute()
    SFWF_volc_ensmean = SFWF_volc.mean(dim='realization').compute()

    SSS_novolc_ensmean = SSS_novolc.mean(dim='realization').compute()
    SST_novolc_ensmean = SST_novolc.mean(dim='realization').compute()
    SHF_novolc_ensmean = SHF_novolc.mean(dim='realization').compute()
    SFWF_novolc_ensmean = SFWF_novolc.mean(dim='realization').compute()

    # Reference Coordinate & Constants
    lon, lat = np.linspace(-180, 180, 361), np.linspace(-90, 90, 181)
    constfile = xr.open_dataset('./data/cesmi6gat31rblc200_ANN_210912_998911_pop_decclimots.nc')

    grav = constfile["grav"]
    cp_sw = constfile["cp_sw"]
    rho_sw = constfile["rho_sw"]
    
    def Regridder(var, lon_in, lat_in, lon_out, lat_out, varreg):
        grid_in  = {'lon': lon_in,  'lat': lat_in}
        grid_out = {'lon': lon_out, 'lat': lat_out}
        regridder = xe.Regridder(grid_in, grid_out, 'bilinear', periodic=True)
        var_reg = regridder(var)
        if varreg:
            var_reg = var_reg.where(var_reg!=0.)
        return var_reg

    print("Regridding...")
    # Regrid volcanic ensemble means
    SSS_volc_ensmean_rg = Regridder(SSS_volc_ensmean, SSS_volc_ensmean.TLONG, SSS_volc_ensmean.TLAT, lon, lat, True)
    SST_volc_ensmean_rg = Regridder(SST_volc_ensmean, SST_volc_ensmean.TLONG, SST_volc_ensmean.TLAT, lon, lat, True)
    SHF_volc_ensmean_rg = Regridder(SHF_volc_ensmean, SHF_volc_ensmean.TLONG, SHF_volc_ensmean.TLAT, lon, lat, True)
    SFWF_volc_ensmean_rg = Regridder(SFWF_volc_ensmean, SFWF_volc_ensmean.TLONG, SFWF_volc_ensmean.TLAT, lon, lat, True)

    # Regrid non-volcanic ensemble means 
    SSS_novolc_ensmean_rg = Regridder(SSS_novolc_ensmean, SSS_novolc_ensmean.TLONG, SSS_novolc_ensmean.TLAT, lon, lat, True)
    SST_novolc_ensmean_rg = Regridder(SST_novolc_ensmean, SST_novolc_ensmean.TLONG, SST_novolc_ensmean.TLAT, lon, lat, True)
    SHF_novolc_ensmean_rg = Regridder(SHF_novolc_ensmean, SHF_novolc_ensmean.TLONG, SHF_novolc_ensmean.TLAT, lon, lat, True)
    SFWF_novolc_ensmean_rg = Regridder(SFWF_novolc_ensmean, SFWF_novolc_ensmean.TLONG, SFWF_novolc_ensmean.TLAT, lon, lat, True)

    # Define seasons
    winter_months = [12, 1, 2]  # DJF
    summer_months = [6, 7, 8]   # JJA

    print("Calculating seasonal means...")
    # Volcanic Seasonal Means
    SSS_volc_winter_ts = SSS_volc_ensmean_rg.sel(time=SSS_volc_ensmean_rg.time.dt.month.isin(winter_months)).groupby('time.year').mean()
    SST_volc_winter_ts = SST_volc_ensmean_rg.sel(time=SST_volc_ensmean_rg.time.dt.month.isin(winter_months)).groupby('time.year').mean()
    SHF_volc_winter_ts = SHF_volc_ensmean_rg.sel(time=SHF_volc_ensmean_rg.time.dt.month.isin(winter_months)).groupby('time.year').mean()
    SFWF_volc_winter_ts = SFWF_volc_ensmean_rg.sel(time=SFWF_volc_ensmean_rg.time.dt.month.isin(winter_months)).groupby('time.year').mean()

    SSS_volc_summer_ts = SSS_volc_ensmean_rg.sel(time=SSS_volc_ensmean_rg.time.dt.month.isin(summer_months)).groupby('time.year').mean()
    SST_volc_summer_ts = SST_volc_ensmean_rg.sel(time=SST_volc_ensmean_rg.time.dt.month.isin(summer_months)).groupby('time.year').mean()
    SHF_volc_summer_ts = SHF_volc_ensmean_rg.sel(time=SHF_volc_ensmean_rg.time.dt.month.isin(summer_months)).groupby('time.year').mean()
    SFWF_volc_summer_ts = SFWF_volc_ensmean_rg.sel(time=SFWF_volc_ensmean_rg.time.dt.month.isin(summer_months)).groupby('time.year').mean()

    # Non-Volcanic Seasonal Means
    SSS_novolc_winter_ts = SSS_novolc_ensmean_rg.sel(time=SSS_novolc_ensmean_rg.time.dt.month.isin(winter_months)).groupby('time.year').mean()
    SST_novolc_winter_ts = SST_novolc_ensmean_rg.sel(time=SST_novolc_ensmean_rg.time.dt.month.isin(winter_months)).groupby('time.year').mean()
    SHF_novolc_winter_ts = SHF_novolc_ensmean_rg.sel(time=SHF_novolc_ensmean_rg.time.dt.month.isin(winter_months)).groupby('time.year').mean()
    SFWF_novolc_winter_ts = SFWF_novolc_ensmean_rg.sel(time=SFWF_novolc_ensmean_rg.time.dt.month.isin(winter_months)).groupby('time.year').mean()

    SSS_novolc_summer_ts = SSS_novolc_ensmean_rg.sel(time=SSS_novolc_ensmean_rg.time.dt.month.isin(summer_months)).groupby('time.year').mean()
    SST_novolc_summer_ts = SST_novolc_ensmean_rg.sel(time=SST_novolc_ensmean_rg.time.dt.month.isin(summer_months)).groupby('time.year').mean()
    SHF_novolc_summer_ts = SHF_novolc_ensmean_rg.sel(time=SHF_novolc_ensmean_rg.time.dt.month.isin(summer_months)).groupby('time.year').mean()
    SFWF_novolc_summer_ts = SFWF_novolc_ensmean_rg.sel(time=SFWF_novolc_ensmean_rg.time.dt.month.isin(summer_months)).groupby('time.year').mean()

    # Period Definitions
    plt_years = [[7701,7705], [7706,7715], [7716,7720]]

    def split_periods_mean(data, periods):
        return [data.sel(year=slice(p[0], p[1])).mean('year') for p in periods]

    def combine_period_datasets(*var_periods):
        period_arrays = [xr.concat(var_period, dim='period') for var_period in var_periods]
        return xr.merge(period_arrays)

    print("Processing periods...")
    # Volcanic Winter
    volc_winter_periods = combine_period_datasets(
        split_periods_mean(SSS_volc_winter_ts, plt_years),
        split_periods_mean(SST_volc_winter_ts, plt_years),
        split_periods_mean(SHF_volc_winter_ts, plt_years),
        split_periods_mean(SFWF_volc_winter_ts, plt_years)
    )
    
    # Volcanic Summer
    volc_summer_periods = combine_period_datasets(
        split_periods_mean(SSS_volc_summer_ts, plt_years),
        split_periods_mean(SST_volc_summer_ts, plt_years),
        split_periods_mean(SHF_volc_summer_ts, plt_years),
        split_periods_mean(SFWF_volc_summer_ts, plt_years)
    )

    # Non-Volcanic Winter
    novolc_winter_periods = combine_period_datasets(
        split_periods_mean(SSS_novolc_winter_ts, plt_years),
        split_periods_mean(SST_novolc_winter_ts, plt_years),
        split_periods_mean(SHF_novolc_winter_ts, plt_years),
        split_periods_mean(SFWF_novolc_winter_ts, plt_years)
    )

    # Non-Volcanic Summer
    novolc_summer_periods = combine_period_datasets(
        split_periods_mean(SSS_novolc_summer_ts, plt_years),
        split_periods_mean(SST_novolc_summer_ts, plt_years),
        split_periods_mean(SHF_novolc_summer_ts, plt_years),
        split_periods_mean(SFWF_novolc_summer_ts, plt_years)
    )

    # Buoyancy Flux Calculation
    def buoyancy_flux(grav, rho_sw, cp_sw, shf, sfwf, sss, sst, p=0.0):
        alpha = gsw.alpha(sss, sst, p) # alpha > 0
        beta = gsw.beta(sss, sst, p) #beta > 0
        
        hal_const = grav.data * 1.0e-5 * beta
        haline = hal_const * sfwf * sss
        
        therm_const = grav.data * 1.0e-1 * (alpha / (rho_sw.data * cp_sw.data))
        thermal = therm_const * shf

        B0 = (haline + thermal) 

        return B0 * 1.0e6, haline * 1.0e6, thermal * 1.0e6, alpha, beta

    def calculate_buoyancy_fluxes(data):
        # Calculate buoyancy fluxes for period 1
        b_flux_1, haline_1, thermal_1, alpha_1, beta_1 = buoyancy_flux(
            grav, rho_sw, cp_sw,
            data['SHF'].isel(period=0), data['SFWF'].isel(period=0),
            data['SSS'].isel(period=0), data['SST'].isel(period=0)
        )

        # Calculate buoyancy fluxes for period 2
        b_flux_2, haline_2, thermal_2, alpha_2, beta_2 = buoyancy_flux(
            grav, rho_sw, cp_sw,
            data['SHF'].isel(period=1), data['SFWF'].isel(period=1),
            data['SSS'].isel(period=1), data['SST'].isel(period=1)
        )

        # Calculate buoyancy fluxes for period 3  
        b_flux_3, haline_3, thermal_3, alpha_3, beta_3 = buoyancy_flux(
            grav, rho_sw, cp_sw,
            data['SHF'].isel(period=2), data['SFWF'].isel(period=2),
            data['SSS'].isel(period=2), data['SST'].isel(period=2)
        )

        # Store results in dictionaries for easy access
        buoyancy_fluxes = {
            'period1': {'total': b_flux_1, 'haline': haline_1, 'thermal': thermal_1, 'alpha': alpha_1, 'beta': beta_1},
            'period2': {'total': b_flux_2, 'haline': haline_2, 'thermal': thermal_2, 'alpha': alpha_2, 'beta': beta_2},
            'period3': {'total': b_flux_3, 'haline': haline_3, 'thermal': thermal_3, 'alpha': alpha_3, 'beta': beta_3}
        }
        return buoyancy_fluxes

    print("Calculating buoyancy fluxes...")
    volc_winter_periods_bflux = calculate_buoyancy_fluxes(volc_winter_periods)
    novolc_winter_periods_bflux = calculate_buoyancy_fluxes(novolc_winter_periods)
    volc_summer_periods_bflux = calculate_buoyancy_fluxes(volc_summer_periods)
    novolc_summer_periods_bflux = calculate_buoyancy_fluxes(novolc_summer_periods)

    seasons_data = {
        'DJF': {'volc': volc_winter_periods_bflux, 'novolc': novolc_winter_periods_bflux},
        'JJA': {'volc': volc_summer_periods_bflux, 'novolc': novolc_summer_periods_bflux}
    }

    def plot_buoyancy(seasons_data, season_name, filename, anomaly=False):
        print(f"Generating plot for {season_name}...")
        fig = plt.figure(figsize=(10, 7))
        gs = gridspec.GridSpec(3, 5, width_ratios=[1, 1, 1, 0.25, 0.1], height_ratios=[1, 1, 1])

        titles = ['Total Buoyancy Flux', 'Haline Component', 'Thermal Component']
        periods = ['period1', 'period2', 'period3']
        years = ['7701-7705', '7706-7715', '7716-7720']
        components = ['total', 'haline', 'thermal']
        labels = ['A', 'B', 'C']

        proj = ccrs.Orthographic(central_longitude=-40, central_latitude=45)

        season_data = seasons_data[season_name]
            
        for row, period in enumerate(periods):
            for col, component in enumerate(components):
                ax = plt.subplot(gs[row, col], projection=proj)
                
                if anomaly:
                    data = (season_data['volc'][period][component] - 
                           season_data['novolc'][period][component])
                else:
                    data = season_data['volc'][period][component]
                    
                im = ax.contourf(data.lon, data.lat, data,
                                levels=np.linspace(-0.01, 0.01, 21) if anomaly else np.linspace(-0.1, 0.1, 21),
                                transform=ccrs.PlateCarree(),
                                extend='both',
                                cmap='RdBu_r')
                
                ax.coastlines()
                ax.add_feature(cfeature.LAND, color='lightgray')
                ax.set_extent([-90, 0, 20, 70], ccrs.PlateCarree())
                ax.gridlines()
                
                if row == 0:
                    ax.set_title(f'{season_name} {titles[col]}')
                    ax.text(0.02, 0.98, labels[col], transform=ax.transAxes,
                           fontsize=18, fontweight='bold', va='top',
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5))
                if col == 0:
                    ax.text(-0.1, 0.5, f'Year {years[row]}',
                           transform=ax.transAxes, rotation=90,
                           verticalalignment='center', size=12)

        # Single colorbar on right side that spans all plots
        cbar_ax = plt.subplot(gs[:, -1])
        label = f'{"[Volcanism - No Volcanism]\nBuoyancy Flux Anomalies " if anomaly else "Volcanism Experiment Buoyancy Flux"} (mm²/s³)'
        cbar = plt.colorbar(im, cax=cbar_ax, label=label, orientation='vertical')
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label, size=14)

        plt.subplots_adjust(wspace=-0.1, hspace=0.2)
        
        print(f"Saving {filename}...")
        plt.savefig(filename + ".png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
        plt.savefig(filename + ".pdf", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
        plt.close(fig)

    # Figure 4 (DJF)
    plot_buoyancy(seasons_data, 'DJF', 'Figure4_SA', anomaly=True)

    # Figure 5 (JJA)
    plot_buoyancy(seasons_data, 'JJA', 'Figure5_SA', anomaly=True)
    
    print("Done.")
