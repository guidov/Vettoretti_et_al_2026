#!/usr/bin/env python
# coding: utf-8

# # Periodic Volcanic Forcing Experiments - Figure 2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import cftime
import nc_time_axis

mpl.rcParams['figure.figsize'] = (16.0, 10.0)

def setaxesfsize(axl,fontsize):
    ticklabelbot = axl.get_xticklabels()
    ticklabelleft = axl.get_yticklabels()
    for labelx in ticklabelbot:
        labelx.set_fontsize(fontsize)
    for labely in ticklabelleft:
        labely.set_fontsize(fontsize)

def get_amoc_max(mocarray):
    latmin=20.
    latmax=50.
    zmin=50000.
    zmax=200000.
    region = 1
    moc_max_em = (mocarray
               .sel(transport_reg=region, moc_comp=0, lat_aux_grid=slice(latmin, latmax), moc_z=slice(zmin, zmax))
               .load())
    moc_max_ei = (mocarray
               .sel(transport_reg=region, moc_comp=1, lat_aux_grid=slice(latmin, latmax), moc_z=slice(zmin, zmax))
               .load())
    moc_max_sm = (mocarray
               .sel(transport_reg=region, moc_comp=2, lat_aux_grid=slice(latmin, latmax), moc_z=slice(zmin, zmax))
               .load())
    moc_max = (moc_max_em +  moc_max_ei +  moc_max_sm).max(dim=('lat_aux_grid', 'moc_z'))
 
    return moc_max

# Create the CFDatetimeCoder instance
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

run_230i = "./data/cesmi6gat31rblc230i_ANN_210912_998911_pop_decclimots.nc"
xrdata_230i = xr.open_dataset(run_230i, engine='netcdf4', chunks={'time': 20}, decode_times=time_coder, decode_timedelta=True)

run_1000p = "./data/cesmi6gat31rblc230iv1000p_ANN_260912_998911_pop_decclimots.nc"
xrdata_1000p = xr.open_dataset(run_1000p, engine='netcdf4', chunks={'time': 20}, decode_times=time_coder, decode_timedelta=True)

run_1000ph = "./data/cesmi6gat31rblc230iv1000ph_ANN_260912_998911_pop_decclimots.nc"
xrdata_1000ph = xr.open_dataset(run_1000ph, engine='netcdf4', chunks={'time': 20}, decode_times=time_coder, decode_timedelta=True)

run_1000pq = "./data/cesmi6gat31rblc230iv1000pq_ANN_260912_998911_pop_decclimots.nc"
xrdata_1000pq = xr.open_dataset(run_1000pq, engine='netcdf4', chunks={'time': 20}, decode_times=time_coder, decode_timedelta=True)

# control
amoc_max_230i = get_amoc_max(xrdata_230i.MOC)
# volc experiments
amoc_max_1000p = get_amoc_max(xrdata_1000p.MOC)
amoc_max_1000ph = get_amoc_max(xrdata_1000ph.MOC)
amoc_max_1000pq = get_amoc_max(xrdata_1000pq.MOC)

fig = plt.figure(figsize=(16,9))

lw=1.5
lalp=0.7

ax0=plt.subplot(311)
ax1=plt.subplot(312)
ax2=plt.subplot(313)
ax = [ax0, ax1, ax2]

cntcol = 'blue'
cntalp = 0.8
# volcanic run colors
vcols = ["red", "green", "purple"]

for i in range(3):
    amoc_max_230i.plot(ax=ax[i], color=cntcol, alpha=cntalp,linewidth=lw, label="Control: No Volcanism (CO$_2$ = 230 ppm)")

amoc_max_1000p.plot(ax=ax[0], color=vcols[0], alpha=lalp,linewidth=lw, label="Volcanism: 1000 Year Periodic EQ 230 Tg (CO$_2$ = 230 ppm)")
amoc_max_1000ph.plot(ax=ax[1], color=vcols[1], alpha=lalp,linewidth=lw, label="Volcanism: 1000 Year Periodic EQ 115 Tg (CO$_2$ = 230 ppm)")
amoc_max_1000pq.plot(ax=ax[2], color=vcols[2], alpha=lalp,linewidth=lw, label="Volcanism: 1000 Year Periodic EQ 58 Tg (CO$_2$ = 230 ppm)")

for i in range(3):
    ax[i].set_ylabel("AMOC Max (Sv)",size=16)
    ax[i].set_xlabel("",size=16)
    ax[i].set_title("", size=16, loc='center')

ax[2].set_ylabel("AMOC Max (Sv)",size=16)
ax[2].set_xlabel("Model Time (Years)",size=16)
ax[2].set_title("", size=16, loc='center')

# do background volcanic lines 
vgrid0 = [ cftime.num2date(yr*365, 'days since 0000-01-01 00:00:00', calendar='noleap') for yr in range(2701,10000,1000) ]

for j in range(3):
    ax[j].set_ylim(0,30)
    setaxesfsize(ax[j],14)
    
    # make thin black line at start
    [ ax[j].vlines(vgrid0[i],0,30, color=vcols[j], alpha=0.7, linewidths=1.5, zorder=1) for i in range(0,8) ]

    ax[j].legend(loc=4,fontsize=12,ncol=2)

# add a,b,c,d
lx = 0.006
ly = 0.88
ax[0].text(lx, ly, 'A',
         transform=ax[0].transAxes, size=18, weight='bold')
ax[1].text(lx, ly, 'B',
         transform=ax[1].transAxes, size=18, weight='bold')
ax[2].text(lx, ly, 'C',
         transform=ax[2].transAxes, size=18, weight='bold')

ax[0].grid()
ax[1].grid()
ax[2].grid()

plt.savefig("Figure2_volc_230_periodic_EQ.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=True)
plt.savefig("Figure2_volc_230_periodic_EQ.pdf", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=True)