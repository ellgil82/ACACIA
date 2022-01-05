import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
import imageio

filepath = '/storage/silver/acacia/ke923690/acacia/new_runs/'

# Load files
#filename = input("Enter name of diagnostics file\n")
#variable = input("Enter variable to plot\n")
#timestep = input("Which timesteps do you need? (0-indexed)\n")

def plot_ice_mmr(sim):
    fig, ax = plt.subplots(figsize=(5, 6))
    for t in np.linspace(0, prop_dict[sim]['ice_mmr'].shape[0]-1, 9).astype(int):
    #for t in [0, 4, 8, 13, 16, -1]:
       color_list = ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal', 'brown', 'darkgrey', 'indigo', ]
       ax.plot(prop_dict[sim]['ice_mmr'][t, 55:90], model_hts[55:90]/1000, label = str(int(time_srs[t]/60)) + ' mins')#, color = color_list[t])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlim(1e-8, 1e-4)
    ax.set_ylim(model_hts[55]/1000, model_hts[90]/1000)
    ax.yaxis.set_ticks([7, 8, 9, 10,])
    ax.set_xscale('log')
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('Ice MMR (kg/kg)')
    ax.legend(loc=4)
    ax.tick_params(which='both', axis='both', direction='in')
    plt.subplots_adjust(left = 0.2)
    plt.savefig(filepath + '../figures/ice_mmr_test_' + sim + '.png')

def plot_vapour_mmr(sim):
    fig, ax = plt.subplots(figsize=(5, 6))
    for t in np.linspace(0, prop_dict[sim]['vapour_mmr_mean'].shape[0]-1, 9).astype(int):
    #for t in [0, 4, 8, 13, 16, -1]:
       color_list = ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal', 'brown', 'darkgrey', 'indigo', ]
       ax.plot(prop_dict[sim]['vapour_mmr_mean'][t, 55:90], model_hts[55:90]/1000, label = str(int(time_srs[t]/60)) + ' mins')#, color = color_list[t])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlim(1e-5, 1e-3)
    ax.set_ylim(model_hts[55]/1000, model_hts[90]/1000)
    ax.yaxis.set_ticks([7, 8, 9, 10,])
    ax.set_xscale('log')
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('Vapour MMR (kg/kg)')
    ax.legend(loc='best')
    ax.tick_params(which='both', axis='both', direction='in')
    plt.subplots_adjust(left = 0.2)
    plt.savefig(filepath + '../figures/vapour_mmr_' + sim + '.png')

def plot_tendencies(sim):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax = ax.flatten()
    #for i, t, l in zip([0,1,2,3,4,5],[0,44, 88, 138, 177, -1 ], ['60 mins', '120 mins', '180 mins', '210 mins', '240 mins', '270 mins']):
    for i, t, l in zip([0,1,2,3,4,5], np.linspace(0, met_dict[sim]['rhi'].shape[0]-1, 9).astype(int), ['1 mins', '60 mins', '120 mins', '180 mins', '210 mins', '240 mins', ]):
    #for i, t, l in zip([0, 1, 2, 3, 4, 5], [0, 8, 17, 26, 32, 38, ], ['1 mins', '60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
        # ax[i].plot(MPS.data[t, 60:110]*60,z[60:110], label='sed')
        # ax[i].plot(MPM.data[t, 60:110] * 60, z[60:110], label='other\nmphys')
        ax[i].plot(tend_dict[sim]['mphys'][t, 55:90], z[55:90]/1000, label='mphys')
        ax[i].plot(tend_dict[sim]['diff'][t, 55:90], z[55:90]/1000, label='diff')
        ax[i].plot(tend_dict[sim]['tvda'][t, 55:90], z[55:90]/1000, label='adv')
        ax[i].plot(tend_dict[sim]['tot'][t, 55:90], z[55:90]/1000, label='tot')
        ax[i].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax[i].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax[i].set_xlim(-1e-8, 1e-8)
        ax[i].set_ylim(model_hts[55]/1000, model_hts[90]/1000)
        ax[i].yaxis.set_ticks([7, 8, 9, 10,])
        ax[i].set_title(str(int(time_srs[t]/60)) + ' mins')
        # ax[i].yaxis.set_ticks([7, 8, 9, 10])
        # ax[i].set_xscale('log')
        ax[i].set_ylabel('Altitude (km)', rotation=90)
        ax[i].set_xlabel('Tendency (kg/kg/s)')
        ax[i].legend(loc=4)
        ax[i].tick_params(which='both', axis='both', direction='in')
    ax[0].set_xlim(-5e-8, 5e-8)
    plt.subplots_adjust(left=0.2, top=0.95, hspace=0.25, wspace=0.25)
    plt.savefig(filepath + '../figures/tendencies_'+ sim + '.png')

def plot_mphys_tendencies(sim):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax = ax.flatten()
    for i, t, l in zip([0,1,2,3,4,5], np.linspace(0, tend_dict[sim]['mphys'].shape[0]-1, 9).astype(int), ['1 mins', '60 mins', '120 mins', '180 mins', '210 mins', '240 mins', ]):
    #for i, t, l in zip([0, 1, 2, 3, 4, 5], [0, 8, 17, 26, 32, 38, 43, ], ['1 mins', '60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
        # for i, t, l in zip([0, 1, 2, 3, 4, 5], [0, 44, 88, 138, 177, -1], ['1 mins', '60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
        ax[i].plot(MPS[t, 55:90], z[55:90]/1000, label='sed')
        ax[i].plot(MPM[t, 55:90], z[55:90]/1000, label='other\nmphys')
        ax[i].plot(tend_dict[sim]['mphys'][t, 55:90], z[55:90]/1000, label='mphys_tot', color='k')
        ax[i].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax[i].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax[i].set_xlim(-1.e-8, 1e-8)
        ax[i].set_title(str(int(time_srs[t]/60)) + ' mins')
        ax[i].yaxis.set_ticks([7, 8, 9, 10])
        ax[i].set_ylim(z[55]/1000, z[90]/1000)
        # ax[i].set_xscale('log')
        ax[i].set_ylabel('Altitude (km)', rotation=90)
        ax[i].set_xlabel('Tendency (kg/kg/s)')
        ax[i].legend(loc=4)
        ax[i].tick_params(which='both', axis='both', direction='in')
    ax[0].set_xlim(-5e-8, 5e-8)
    plt.subplots_adjust(left=0.2, top=0.95, hspace=0.25, wspace=0.25)
    plt.savefig(filepath + '../figures/mphys_tendencies_' + sim + '.png')

def plot_processes(sim):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax = ax.flatten()
    color_list = ['red',  'lightblue', 'purple', 'orange','darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal',
                  'lime', 'darkgrey', 'indigo', 'darkred', 'cyan', 'bisque', 'lightsteelblue','brown', 'turquoise',
                  'forestgreen',  'coral', 'maroon']
    lgd_labs = []
    lgd_lns = []
    #for i, t in zip([0,1,2,3,4,5], [0, 4, 8, 13, 16, -1]):
    for i, t in zip([0,1,2,3,4,5], np.linspace(0, proc_dict[sim]['ice_nucl'].shape[0]-1, 9).astype(int)):
    #for i, t, l in zip([0, 1, 2, 3, 4, 5], [0, 8, 17, 26, 32, 38 ], ['1 mins', '60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
        # for i, t, l in zip([0, 1, 2, 3, 4, 5], [0, 44, 88, 138, 177, -1], ['1 mins', '60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
        for c, j in enumerate(proc_dict[sim].keys()):
            if proc_dict[sim][j][t].max() > 1e-15:
                if j not in lgd_labs:
                    lgd_labs.append(j)
                    lgd_lns.append(color_list[c])
                ax[i].plot(proc_dict[sim][j][t, 55:90], z[55:90]/ 1000, label=j, color=color_list[c])
                ax[i].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
                ax[i].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
                ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                ax[i].set_xlim(-5.e-5, 5.e-5)
                ax[i].yaxis.set_ticks([7, 8, 9, 10])
                ax[i].set_ylim(z[55] / 1000, z[90] / 1000)
                ax[i].set_title(str(int(time_srs[t]/60)) + ' mins')
                ax[i].set_ylabel('Altitude (km)', rotation=90)
                ax[i].set_xlabel('Process rate (kg/kg/s)')
                ax[i].tick_params(which='both', axis='both', direction='in')
    ax[0].set_xlim(-2.e-4, 2.e-4)
   # ax[1].set_xlim(-5e-6, 5e-6)
    lns = []
    labs = []
    for l, s in zip(lgd_lns, lgd_labs):
        lns.append(Line2D([0], [0], color=l, linewidth=2.5))
        labs.append(s)
        lgd = ax[-1].legend(lns, labs, bbox_to_anchor=(1.05, 1.), loc=2)
    plt.subplots_adjust(left=0.1, right=0.7)
    plt.savefig(filepath + '../figures/process_rates_' + sim + '.png')

def plot_rh(sim):
    fig, ax = plt.subplots(figsize=(5, 6))
    #for t, l in zip([44, 88, 138, 177, -1 ], ['60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
    for t in np.linspace(0, met_dict[sim]['rhi'].shape[0]-1, 9).astype(int):
    #for t in [0, 4, 8, 13, 16, -1]:
        color_list = ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal',  'darkgrey', 'indigo', 'brown',]
        ax.plot(met_dict[sim]['rhi'][t, 55:90], model_hts[55:90]/1000, label = str(int(time_srs[t]/60)) + ' mins')#, color = color_list[t])
        #ax.plot(met_dict[sim]['rhi'][t, 70:100].data*100, model_hts[70:100]/1000, label = l)
    #ax.set_xlim(0, 120)
    ax.set_ylim(model_hts[55]/1000, model_hts[90]/1000)
    ax.yaxis.set_ticks([7, 8, 9, 10])
    ax.set_ylim(model_hts[55]/1000, model_hts[90]/1000)
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('RHi (%)')
    ax.legend(loc=4)
    ax.tick_params(which='both', axis='both', direction='in')
    plt.subplots_adjust(left = 0.2)
    plt.savefig(filepath + '../figures/rhi_' + sim + '.png')

def plot_Tabs(sim):
    fig, ax = plt.subplots(figsize=(5, 6))
    #for t, l in zip([44, 88, 138, 177, -1 ], ['60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
    for t, l in zip(np.linspace(0, met_dict[sim]['TdegC'].shape[0]-1, 9).astype(int), ['30 mins', '60 mins', '120 mins', '180 mins', '210 mins', '240 mins', ]):
    #for t, l in zip([0, 8, 17, 26, 32, 38 ], ['1 mins', '60 mins', '120 mins', '180 mins', '210 mins', '240 mins', ]):
       color_list = ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal', 'brown', 'darkgrey', 'indigo', ]
       ax.plot(met_dict[sim]['TdegC'][t, 55:90], model_hts[55:90]/1000, label = l)#, color = color_list[t])
    ax.yaxis.set_ticks([7, 8, 9, 10])
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('Temp ($^{\circ}$C)')
    ax.legend(loc=4)
    ax.tick_params(which='both', axis='both', direction='in')
    plt.subplots_adjust(left = 0.2)
    plt.savefig(filepath + '../figures/TdegC_' + sim + '.png')

def plot_Nice(sim):
    fig, ax = plt.subplots(figsize=(5, 6))
    for t in np.linspace(0, prop_dict[sim]['ice_nc_mean'].shape[0]-1, 9).astype(int):
    #for t in [0, 4, 8, 13, 16, -1]:
       color_list = ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal', 'brown', 'darkgrey', 'indigo', ]
       ax.plot(prop_dict[sim]['ice_nc_mean'][t, 55:90], model_hts[55:90]/1000, label = str(int(time_srs[t]/60)) + ' mins')#, color = color_list[t])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlim(1e-4,1e8)
    ax.set_ylim(model_hts[55]/1000, model_hts[90]/1000)
    ax.yaxis.set_ticks([7, 8, 9, 10,])
    ax.set_xscale('log')
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('N$_{ice}$ (cm$^{-3}$)')
    ax.legend(loc='best')
    ax.tick_params(which='both', axis='both', direction='in')
    plt.subplots_adjust(left = 0.2)
    plt.savefig(filepath + '../figures/Nice_' + sim + '.png')

prop_dict = {}
proc_dict = {}
met_dict = {}
tend_dict = {}

for filename, sim in zip(['*17_CTRL*', '*22a*','*22b*', '*22c*',],['CTRL', 'slow_uv', 'slow_uv_ICNCx2_8-9','slow_uv_ICNCx2_9-10',]):#, '*18a*', '*18b*', '*14p*8-9*','*14o*9-10*', ], [ 'CTRL', 'ICNCx2_8-9_2000', 'ICNCx2_9-10_2000', 'ICEx2_8-9_2000', 'ICEx2_9-10_2000',]):
    ## Sources of water
    proc_dict[sim] = {}
    for name, v in zip(['cond', 'ice_nucl', 'homg_fr_cl',  'ice_dep', 'snow_dep',  'ice_sed', 'snow_sed', 'graupel_sed', 'ice_subm', 'snow_subm',
                        'graupel_subm', 'ice_melt', 'snow_melt', 'graupel_melt',  'rn_autoconv', 'rn_accr',  'snow_accr_ice', 'sn_accr_rn', 'gr_accr_cl',  'gr_accr_sn', 'ice_acc_w', 'ice_to_snow'],['pcond_mean', 'pinuc_mean', 'phomc_mean', 'pidep_mean', 'psdep_mean', 'psedi_mean', 'pseds_mean', 'psedg_mean',
                        'pisub_mean', 'pssub_mean', 'pgsub_mean','pimlt_mean', 'psmlt_mean', 'pgmlt_mean',  'praut_mean',  'pracw_mean', 'psaci_mean', 'psacr_mean',
                        'pgacw_mean', 'pgacs_mean', 'piacw_mean', 'psaut_mean']):
        #cubes = iris.load(filepath + filename, v)
        #proc_dict[sim][name] = np.concatenate((cubes[0].data, cubes[1].data), axis=0)
        f1 = iris.load_cube(filepath + filename + '?d_3600.nc', v)
        f2 = iris.load_cube(filepath + filename + '?d_7200.nc', v)
        f3 = iris.load_cube(filepath + filename + '?d_10800.nc', v)
        proc_dict[sim][name] = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
    for name in proc_dict[sim].keys():
        print(name)
        print(proc_dict[sim][name].data.max())
    prop_dict[sim] = {}
    for name, v in zip(['ice_mmr', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'ice_nc_mean', 'snow_nc_mean', 'graupel_nc_mean'], ['ice_mmr_mean',  'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'ice_nc_mean', 'snow_nc_mean', 'graupel_nc_mean']):
        f1 = iris.load_cube(filepath + filename + '?d_3600.nc', v)
        f2 = iris.load_cube(filepath + filename + '?d_7200.nc', v)
        f3 = iris.load_cube(filepath + filename + '?d_10800.nc', v)
        prop_dict[sim][name] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
    try:
        time_srs = np.concatenate((f1.coord('time_series_100_100').points, f2.coord('time_series_100_100').points,f3.coord('time_series_100_100').points), axis = 0)
    except iris.exceptions.CoordinateNotFoundError:
        time_srs = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points, f3.coord('time_series_10_30').points), axis=0)
    ## Relevant met
    met_dict[sim] = {}
    for name, v in zip([ 'rh', 'rhi', 'temp', 'w'], ['rh_mean', 'rhi_mean', 'temperature', 'w']):
        # cubes = iris.load(filepath + filename, v)
        # met_dict[sim][name] = np.concatenate((cubes[0].data, cubes[1].data), axis=0)
        f1 = iris.load_cube(filepath + filename + '?d_3600.nc', v)
        f2 = iris.load_cube(filepath + filename + '?d_7200.nc', v)
        f3 = iris.load_cube(filepath + filename + '?d_10800.nc', v)
        met_dict[sim][name] = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
        if name == 'temp':
            met_dict[sim]['TdegC'] = np.mean(met_dict[sim]['temp'].data - 273.15, axis=(1, 2))
        if name == 'rhi':
            met_dict[sim]['rhi'] = met_dict[sim]['rhi'] * 1000 
    #th_init = np.genfromtxt(filepath + 'Yang_forcing/theta.csv', delimiter=',', usecols=1, skip_header=1)
    #th_init = iris.load_cube(filepath + filename, 'thinit')
    #met_dict[sim]['theta'] = th_init + met_dict[sim]['theta']
    # Load model heights
    model_hts = f1.coord('z').points + 50 # add 50 to offset centring in middle of level
    z = model_hts
    # Load tendencies
    tend_dict[sim] = {}
    for name, v in zip(['mphys', 'diff', 'tot', 'tvda'], ['dqi_mphys_mean', 'tend_qi_diff_mean', 'tend_qi_total_mean', 'tend_qi_tvda_mean',]):
        # 'tend_qi_tvdaadvection_3d_local', 'tend_qi_thadvection_3d_local', 'tend_qi_pwadvection_3d_local', 'tend_qi_diffusion_3d_local', 'tend_qi_total_3d_local'
        #cubes = iris.load(filepath + filename, v)
        #proc_dict[sim][name] = np.concatenate((cubes[0].data, cubes[1].data), axis=0)
        f1 = iris.load_cube(filepath + filename + '?d_3600.nc', v)
        f2 = iris.load_cube(filepath + filename + '?d_7200.nc', v)
        f3 = iris.load_cube(filepath + filename + '?d_10800.nc', v)
        tend_dict[sim][name] = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
    MPS = proc_dict[sim]['ice_sed'] + proc_dict[sim]['snow_sed'] + proc_dict[sim]['graupel_sed']
    MPM = tend_dict[sim]['mphys'] - MPS
    #dqi_dt = dqi_tvda_mean + MPS + MPM + dqi_diff_mean
    # this should be equal to dqi_tot_mean
    plot_ice_mmr(sim)
    plot_vapour_mmr(sim)
    plot_Nice(sim)
    plot_tendencies(sim)
    plot_rh(sim)
    plot_Tabs(sim)
    plot_processes(sim)
    plot_mphys_tendencies(sim)

plt.show()

def plot_first_ts():
    filename = 'Yang_test4b_q=1.3_Cooper_*d_60.nc'
    sim = 'ts_1s'
    proc_dict[sim] = {}
    for name, v in zip(
            ['cond', 'ice_nucl', 'homg_fr_cl',  'ice_dep', 'snow_dep', 'ice_sed', 'snow_sed', 'graupel_sed', 'ice_subm',
             'snow_subm',
             'graupel_subm', 'ice_melt', 'snow_melt', 'graupel_melt', 'rn_autoconv', 'rn_accr', 'snow_accr_ice',
             'sn_accr_rn', 'gr_accr_cl', 'gr_accr_sn', 'ice_acc_w', 'ice_to_snow'],
            ['pcond_mean', 'pinuc_mean', 'phomc_mean',  'pidep_mean', 'psdep_mean', 'psedi_mean', 'pseds_mean',
             'psedg_mean',
             'pisub_mean', 'pssub_mean', 'pgsub_mean', 'pimlt_mean', 'psmlt_mean', 'pgmlt_mean', 'praut_mean',
             'pracw_mean', 'psaci_mean', 'psacr_mean',
             'pgacw_mean', 'pgacs_mean', 'piacw_mean', 'psaut_mean']):
        # cubes = iris.load(filepath + filename, v)
        # proc_dict[sim][name] = np.concatenate((cubes[0].data, cubes[1].data), axis=0)
        proc_dict[sim][name] = iris.load_cube(filepath + filename, v)
    prop_dict[sim] = {}
    for name, v in zip(['ice_mmr', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean'],
                       ['ice_mmr_mean', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean']):
        prop_dict[sim][name] = iris.load_cube(filepath + filename, v)
    ## Relevant met
    met_dict[sim] = {}
    for name, v in zip(['theta', 'w', 'rh', 'temp'], ['theta_mean', 'w_wind_mean', 'rh_mean', 'temperature']):
        # cubes = iris.load(filepath + filename, v)
        # met_dict[sim][name] = np.concatenate((cubes[0].data, cubes[1].data), axis=0)
        met_dict[sim][name] = iris.load_cube(filepath + filename, v)
        if name == 'temp':
            met_dict[sim]['TdegC'] = np.mean(met_dict[sim]['temp']-273.15, axis = (1,2))
    model_hts = prop_dict[sim]['ice_mmr'].coord('zn').points + 50  # add 50 to offset centring in middle of level
    z = model_hts
    # Load tendencies
    tend_dict[sim] = {}
    for name, v in zip(['mphys', 'diff', 'tot', 'tvda', 'tvda_3d', 'th_adv_3d', 'pw_adv_3d',
                        'diff_3d', 'tot_3d'],
                       ['dqi_mphys_mean', 'tend_qi_diff_mean', 'tend_qi_total_mean', 'tend_qi_tvda_mean', ]):
        tend_dict[sim][name] = iris.load_cube(filepath + filename, v)
    MPS = proc_dict[sim]['ice_sed'] + proc_dict[sim]['snow_sed'] + proc_dict[sim]['graupel_sed']
    MPM = tend_dict[sim]['mphys'] - MPS
    plot_ice_mmr()
    plot_vapour_mmr()
    plot_processes()
    plot_tendencies()
    plot_mphys_tendencies()
    plot_rh()
    plt.show()

for t in np.arange(0, 197, 6):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax2 = ax.twiny()
    color_list = ['green', 'lime', 'coral', 'maroon']
    for c, i in enumerate(proc_dict[sim].keys()):
        if proc_dict[sim][i][t].data.max() > 0.:
            ax.plot(proc_dict[sim][i][t].data, model_hts/1000, label = i, color = color_list[c])
        ax2.plot(met_dict[sim]['w'][t].data, model_hts/1000, label = 'w', color = 'darkgrey', linestyle = '--')
    ax2.set_xlabel('w wind (m/s)')
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlim(-1.5e-8, 1.5e-8)
    ax2.set_xlim(-3.5e-5, 3.5e-5)
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('Process rate (kg/kg/s)')
    lgd = ax.legend(bbox_to_anchor=(1.05, 1.), loc=2)
    #ax2.legend() sc
    ax.text(0., 1.1, s = 'timestep = '+ str(t), fontweight ='bold', transform = ax.transAxes)
    plt.subplots_adjust(left = 0.1, right = 0.7)
    plt.savefig(filepath + '../figures/process_rates_test4b_timestep_' + str(t) + '.png')

#plt.show()
plt.close()

for t in range(27):
    fig, ax = plt.subplots()
    c = ax.pcolormesh(w_mn[t], vmin=-0.05, vmax = 0.05)
    plt.colorbar(c)
    plt.savefig(filepath + 'w_mean_' + str(t) + '.png' )


filenames = []
# create file name and append it to a list
for i in range(27):
    fn = filepath + f'w_mean_{i}.png'
    filenames.append(fn)

with imageio.get_writer(filepath + 'w_mean_CTRL.gif', mode='I') as writer:
    for f in filenames:
        image = imageio.imread(f)
        writer.append_data(image)


filenames = []
# create file name and append it to a list
for i in np.arange(0, 197, 6):
    fn = filepath + '../figures/' + f'process_rates_test4b_timestep_{i}.png'
    filenames.append(fn)

with imageio.get_writer(filepath + '../figures/Animated_process_rates_test4b.gif', mode='I') as writer:
    for f in filenames:
        image = imageio.imread(f)
        writer.append_data(image)

# Print values
for i in proc_dict[sim].keys():
    print(i)
    print(proc_dict[sim][i][1:].data.max())

# Calculate total water budget of cloud
ins = proc_dict[sim]['cond'] + proc_dict[sim]['ice_nucl'] + proc_dict[sim]['homg_freezing'] + proc_dict[sim]['ice_dep']
outs = proc_dict[sim]['ice_melt'] + proc_dict[sim]['graupel_melt'] + proc_dict[sim]['snow_melt'] + proc_dict[sim]['ice_subm'] + proc_dict[sim]['snow_subm'] + proc_dict[sim]['graupel_subm']
# Sink from individual layers - do we consider the 'cloud system', and if so, how do we define this?
movt = proc_dict[sim]['ice_sed'] + proc_dict[sim]['snow_sed'] + proc_dict[sim]['graupel_sed']


## Questions
# 1. is the budget for the entire cloud?
# 2. is it for specific layers?

ts_5min = prop_dict[sim]['ice_mmr'].coord('time_series_300_300').points


# Define cloud layers
ts = 0
# find indices of array where CF == 1
cl_base = iris.load_cube(filepath + filename, 'clbas')
cl_top = iris.load_cube(filepath + filename, 'cltop')

water_budget = np.sum(ins[ts, cl_base:cl_top].data) - np.sum(outs[ts, cl_base:cl_top].data) - np.sum(movt[ts, cl_base].data)

water_budget = np.zeros(proc_dict['CTRL']['ice_nucl'].shape)
for k in ['ice_nucl', 'homg_fr_cl', 'ice_dep', 'snow_dep', 'ice_sed', 'snow_sed', 'graupel_sed', 'ice_subm', 'snow_subm', 'graupel_subm', 'snow_accr_ice', 'gr_accr_sn', 'ice_acc_w', 'ice_to_snow']:
    water_budget = water_budget + proc_dict['CTRL'][k]

plt.plot(water_budget[:, 60:95].mean())


water_budget17 = np.zeros(proc_dict['17CTRL']['ice_nucl'].shape)
for k in proc_dict['17CTRL'].keys():
    water_budget17 = water_budget17 + proc_dict['17CTRL'][k]


proc_dict = {}
tend_dict = {}
#for filename, sim in zip(['*17_CTRL*', '*18a*', '*18b*', '*14p*8-9*','*14o*9-10*', ], [ 'CTRL', 'ICNCx2_8-9_2000', 'ICNCx2_9-10_2000', 'ICEx2_8-9_2000', 'ICEx2_9-10_2000',]):
for filename, sim in zip([ '*22a*','*22b*', '*22c*','*22d*', '*22e*','*22f*', '*22g*','*22h*', '*22i*','*22j*',],['slow_uv', 'ICNCx2_8-9','ICNCx2_9-10','no_qF','ICNCx1.5_8-9','ICNCx1.5_9-10','ICNCx1.25_8-9','ICNCx1.25_9-10','ICNCx1.1_8-9','ICNCx1.1_9-10',]):#, '*18a*', '*18b*', '*14p*8-9*','*14o*9-10*', ], [ 'CTRL', 'ICNCx2_8-9_2000', 'ICNCx2_9-10_2000', 'ICEx2_8-9_2000', 'ICEx2_9-10_2000',]):
    ## Sources of water
    proc_dict[sim] = {}
    tend_dict[sim] = {}
    for name, v in zip([ 'clbas', 'cltop', 'ice_nucl', 'homg_fr_cl',  'ice_dep', 'snow_dep',  'ice_sed', 'snow_sed', 'graupel_sed', 'ice_subm', 'snow_subm',
                        'graupel_subm',   'snow_accr_ice',   'gr_accr_sn', 'ice_acc_w', 'ice_to_snow',],['clbas', 'cltop', 'pinuc_mean', 'phomc_mean', 'pidep_mean', 'psdep_mean', 'psedi_mean', 'pseds_mean', 'psedg_mean',
                        'pisub_mean', 'pssub_mean', 'pgsub_mean', 'psaci_mean', 'pgacs_mean', 'piacw_mean', 'psaut_mean',  ]):
        #cubes = iris.load(filepath + filename, v)
        #proc_dict[sim][name] = np.concatenate((cubes[0].data, cubes[1].data), axis=0)
        f1 = iris.load_cube(filepath + filename + '?d_3600.nc', v)
        f2 = iris.load_cube(filepath + filename + '?d_7200.nc', v)
        f3 = iris.load_cube(filepath + filename + '?d_10800.nc', v)
        try:
            f4 = iris.load_cube(filepath + filename + '?d_14400.nc', v)
            proc_dict[sim][name] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
            if name == 'ice_nucl':
                proc_dict[sim]['time_srs'] = np.concatenate(
                    (f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points,
                     f3.coord('time_series_10_30').points, f4.coord('time_series_10_30').points), axis=0)
            elif name == 'clbas':
                proc_dict[sim]['time_srs_lowres'] = np.concatenate((f1.coord('time_series_300_300').points,
                                                                    f2.coord('time_series_300_300').points,
                                                                    f3.coord('time_series_300_300').points,
                                                                    f4.coord('time_series_300_300').points), axis=0)
        except:
            print(sim + ' is only 10800 s long')
            proc_dict[sim][name] = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
            if name == 'ice_nucl':
                proc_dict[sim]['time_srs'] = np.concatenate((f1.coord('time_series_10_30').points,
                                                            f2.coord('time_series_10_30').points,
                                                            f3.coord('time_series_10_30').points), axis=0)
            elif name == 'clbas':
                proc_dict[sim]['time_srs_lowres'] = np.concatenate((f1.coord('time_series_300_300').points,
                                                            f2.coord('time_series_300_300').points,
                                                            f3.coord('time_series_300_300').points), axis=0)
    proc_dict[sim]['alt'] = f1.coord('zn').points + 50
    proc_dict[sim]['sed'] = proc_dict[sim]['graupel_sed'] + proc_dict[sim]['ice_sed'] + proc_dict[sim]['snow_sed']
    proc_dict[sim]['subm'] = proc_dict[sim]['graupel_subm'] + proc_dict[sim]['ice_subm'] + proc_dict[sim]['snow_subm']
    proc_dict[sim]['dep'] = proc_dict[sim]['ice_dep'] + proc_dict[sim]['snow_dep']
    proc_dict[sim]['gr'] = proc_dict[sim]['ice_to_snow'] + proc_dict[sim]['snow_accr_ice']
    #for v in ['sed', 'subm', 'dep', 'gr']:
            #proc_dict[sim][v][proc_dict[sim][v] == 0.] = np.nan
    for name, v in zip(['dqi_mphys','dqi_diff', 'dqi_tot', 'dqi_tvda', 'dqi_for','dqs_mphys','dqs_diff', 'dqs_tot', 'dqs_tvda', 'dqs_for',
                        'dqg_mphys','dqg_diff', 'dqg_tot', 'dqg_tvda', 'dqg_for',],
                       [ 'dqi_mphys_mean', 'tend_qi_diff_mean', 'tend_qi_total_mean','tend_qi_tvda_mean', 'tend_qi_for_mean',
                         'dqs_mphys_mean', 'tend_qs_diff_mean', 'tend_qs_total_mean', 'tend_qs_tvda_mean', 'tend_qs_for_mean',
                         'dqg_mphys_mean', 'tend_qg_diff_mean', 'tend_qg_total_mean', 'tend_qg_tvda_mean', 'tend_qg_for_mean']):
        #cubes = iris.load(filepath + filename, v)
        #proc_dict[sim][name] = np.concatenate((cubes[0].data, cubes[1].data), axis=0)
        f1 = iris.load_cube(filepath + filename + '?d_3600.nc', v)
        f2 = iris.load_cube(filepath + filename + '?d_7200.nc', v)
        f3 = iris.load_cube(filepath + filename + '?d_10800.nc', v)
        try:
            f4 = iris.load_cube(filepath + filename + '?d_14400.nc', v)
            tend_dict[sim][name] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            tend_dict[sim][name] = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
        #tend_dict[sim][name][tend_dict[sim][name] == 0.] = np.nan
    tend_dict[sim]['mphys'] = tend_dict[sim]['dqi_mphys'] + tend_dict[sim]['dqs_mphys' ]+ tend_dict[sim]['dqg_mphys']
    tend_dict[sim]['diff'] = tend_dict[sim]['dqi_diff'] + tend_dict[sim]['dqs_diff' ]+ tend_dict[sim]['dqg_diff']
    tend_dict[sim]['tvda'] = tend_dict[sim]['dqi_tvda'] + tend_dict[sim]['dqs_tvda' ]+ tend_dict[sim]['dqg_tvda']
    tend_dict[sim]['for'] = tend_dict[sim]['dqi_for'] + tend_dict[sim]['dqs_for' ]+ tend_dict[sim]['dqg_for']
    tend_dict[sim]['tot'] = tend_dict[sim]['dqi_tot'] + tend_dict[sim]['dqs_tot' ]+ tend_dict[sim]['dqg_tot']

fig, ax = plt.subplots(figsize = (6,4))#(8,5))
ax.plot(time_srs,np.nanmean(tend_dict['CTRL']['dqi_mphys'],axis= 1), label = 'dqi_mphys', color = '#2FAFE7')
ax.plot(time_srs,np.nanmean(tend_dict['CTRL']['dqs_mphys'],axis= 1), label = 'dqs_mphys', color =  '#13C772')
ax.plot(time_srs,np.nanmean(tend_dict['CTRL']['dqi_tot'],axis= 1), label = 'dqi_tot', color ='#DF2FE7')
ax.plot(time_srs,np.nanmean(tend_dict['CTRL']['dqs_tot'],axis= 1), label = 'dqs_tot', color = '#E77D2F')
ax.plot(time_srs,np.nanmean(tend_dict['CTRL']['mphys'],axis= 1), label = 'mphys', lw=2, color = '#0954F3')
ax.plot(time_srs,np.nanmean(tend_dict['CTRL']['tot'],axis= 1), label = 'tot', color = '#F92E01', lw=2)
ax.axhline(y=0, color = '#222222', ls = 'dashed')
#plt.legend(loc =2)
ax.set_ylabel('Tendency (kg/kg/s)', fontsize = 20)#,  color = '#222222',)
ax.set_xlabel('Time (s)', fontsize = 20)#,  color = '#222222',)
ax.set_xlim(8010, 14400)#(30, 600) #
#ax.set_ylim(-1e-9, 1e-8)#
ax.set_ylim(-1.5e-10, 1.5e-10)
ax.tick_params(labelsize = 20, length = 5)
plt.subplots_adjust(left = 0.2, right = 0.95)#, bottom = 0.15)
plt.savefig(filepath + '../figures/Spin-up_figure_tendencies_inset2.png')
plt.show()


import scipy.interpolate as intp

a=np.copy(proc_dict['CTRL']['sed'])
x=np.array(range(a.shape[0]))
xnew=np.linspace(x.min(), x.max(), 342)
f = intp.interp1d(x,a,axis=0)
print(f(xnew).shape)
upsampled_ctrl = f(xnew)

def mega_plot():
    fig, ax = plt.subplots(4,4, figsize = (18, 15), sharex=True, sharey=True)
    ax = ax.flatten()
    CbAx = fig.add_axes([0.4, 0.08, 0.3, 0.02])
    n = 0
    ## set levels for contours
    pos = np.geomspace(1e-12, 1e-2, 20)
    neg = pos * -1
    vmin = -1e-2
    vmax = 1e-2
    linthresh = 1e-10
    linscale = 5e-10
    levels = np.concatenate((np.flip(neg), np.zeros(1, ), pos))
    for axs in ax:
        #axs.set_facecolor('#BEDBE4')
        axs.set_xlim(0,14400)
        axs.set_ylim(proc_dict['slow_uv']['alt'][60], proc_dict['slow_uv']['alt'][90] )
        axs.yaxis.set_ticks([7000, 8000, 9000, 10000 ])
        plt.setp(axs.spines.values(), linewidth=1, color='dimgrey')
        axs.tick_params(which='both', axis='both', width = 1, labelcolor = 'dimgrey',  color = 'dimgrey', direction='in')
    for  k in  [ 'ICNCx2_9-10','ICNCx1.5_9-10','ICNCx1.25_9-10','ICNCx1.1_9-10']:
        #fig.text( s = sim + ' - CTRL')
        n = n-1
        for v in ['dep', 'gr', 'sed', 'subm']:
            ## upsample ctrl for subtracting higher-res variables
            #a = np.copy(proc_dict['CTRL'][v])
            #x = np.array(range(a.shape[0]))
            #xnew = np.linspace(x.min(), x.max(), 342)
            #f = intp.interp1d(x, a, axis=0)
            #upsampled_ctrl = f(xnew)
            if v == 'dep':
                n = n+1
            else:
                n = n
            if k == 'CTRL':
                c = ax[n].contourf(proc_dict[k]['time_srs'][:342], proc_dict[k]['alt'][60:90], (proc_dict[k][v][:281, 60:90].transpose()),  cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels)
            else:
                c = ax[n].contourf(proc_dict[k]['time_srs'][:459], proc_dict[k]['alt'][60:90],
                                   #((proc_dict[k][v][:, 60:90] - upsampled_ctrl[:, 60:90]).transpose()),
                                   ((proc_dict[k][v][:459, 60:90] - proc_dict['slow_uv'][v][:459, 60:90]).transpose()),
                                   norm=matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh,
                                                                     linscale=linscale), cmap='bwr', vmin=vmin,vmax=vmax, levels=levels)
            ax[n].plot(proc_dict[k]['time_srs_lowres'], np.ma.masked_less_equal(np.nanmean(proc_dict[k]['clbas'], axis=(1, 2)),8000), color='darkgrey')
            ax[n].plot(proc_dict[k]['time_srs_lowres'], np.ma.masked_less_equal(np.nanmean(proc_dict[k]['cltop'], axis=(1, 2)),8000), color='darkgrey')
            n=n+1
    cb = plt.colorbar(c, cax = CbAx,orientation = 'horizontal', ticks = [-1e-3, -1e-6, -1e-9, 0, 1e-9, 1e-6, 1e-3])
    cb.ax.set_xlabel('Process rate (kg kg$^{-1}$ s$^{-1}$)', color='dimgrey', fontsize=18)
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=16, labelcolor='dimgrey', pad=10, size=3, color = 'dimgrey', direction = 'in')
    cb.update_ticks()
    cb.outline.set_linewidth(2)
    for f in [0,4,8,12]:
        ax[f].set_ylabel('Altitude (m)', color = 'dimgrey', fontsize = 16,labelpad=50, rotation = 0)
    for f, l in zip([0,1,2,3], ['vapour\ndeposition', 'growth (aggregation\n/accretion)', 'sedimentation', 'sublimation']):
        ax[f].set_title(l, fontsize = 18, color = 'dimgrey', pad = 20, fontweight='bold')
    for f, l in zip([3,7,11,-1], [ 'ICNCx2_9-10','ICNCx1.5_9-10','ICNCx1.25_9-10','ICNCx1.1_9-10']):
        ax[f].text(1.05, 0.5,l + '\n - CTRL',  transform = ax[f].transAxes, fontsize = 18, fontweight='bold', color = 'dimgrey')
    plt.subplots_adjust(left = 0.12, right = 0.83, bottom=0.15)
    plt.savefig(filepath + '../figures/Process_rates_uv_runs_9-10.png')
    plt.show()

mega_plot()

def plot_proc_contour(var):
    fig, ax = plt.subplots(2, 2, figsize=(6, 7), sharex=True, sharey=True)
    if var == 'dep':
        pos = np.geomspace(1e-9, 1e-4, 40)
        neg = pos * -1
        vmin = -1e-4
        vmax = 1e-4
        linthresh = 1e-7
        linscale = 1e-3
    elif var == 'sed':
        pos = np.geomspace(1e-9, 1e-3, 40)
        neg = pos * -1
        vmin = -1e-3
        vmax = 1e-3
        linthresh = 1e-7
        linscale = 1e-5
    elif var == 'gr':
        pos = np.geomspace(1e-9, 1e-3, 40)
        neg = pos * -1
        vmin = -1e-3
        vmax = 1e-3
        linthresh = 1e-7
        linscale = 1e-5
    elif var == 'subm':
        pos = np.geomspace(1e-9, 1e-3, 40)
        neg = pos * -1
        vmin = -1e-3
        vmax = 1e-3
        linthresh = 1e-7
        linscale = 1e-5
    levels = np.concatenate((np.flip(neg), np.zeros(1, ), pos))
    ax = ax.flatten()
    cbax = fig.add_axes([0.3, 0.18, 0.5, 0.02])
    for k, axs in zip(['ICNCx2_8-9','ICNCx1.5_8-9','ICNCx1.25_8-9','ICNCx1.1_8-9' ], ax):
        c = axs.contourf(proc_dict[k]['time_srs'][:459], proc_dict[k]['alt'][60:90], (proc_dict[k][var][:459, 60:90] - proc_dict['slow_uv'][var][:459, 60:90]).transpose(),
                         norm=matplotlib.colors.SymLogNorm(vmin = vmin, vmax = vmax,linthresh=linthresh, linscale=linscale), cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels)
        axs.set_title(k + ' - CTRL')
        #axs.set_facecolor('#BEDBE4')
        axs.plot(proc_dict[k]['time_srs_lowres'],np.nanmean(proc_dict[k]['clbas'], axis=(1, 2)), color='darkgrey')
        axs.plot(proc_dict[k]['time_srs_lowres'],np.nanmean(proc_dict[k]['cltop'], axis=(1, 2)), color='darkgrey')
        axs.set_xlim(0,14000)
        axs.set_ylim(proc_dict['slow_uv']['alt'][60], proc_dict['slow_uv']['alt'][90] )
    cb = plt.colorbar(c, cax=cbax, orientation='horizontal', extend='both', ticks = [-1e-5, 0, 1e-5])
    plt.subplots_adjust(top=0.95, bottom = 0.25)
    plt.savefig(filepath + '../figures/Hovmoller_' + var + '_uv_runs_8-9.png')
    plt.show()

plot_proc_contour('subm')
plot_proc_contour('sed')
plot_proc_contour('dep')
plot_proc_contour('gr')

def tendencies_plot():
    fig, ax = plt.subplots(4,4, figsize = (18, 15), sharex=True, sharey=True)
    ax = ax.flatten()
    CbAx = fig.add_axes([0.4, 0.08, 0.3, 0.02])
    n = 0
    ## set levels for contours
    pos = np.geomspace(1e-14, 1e-5, 20)
    neg = pos * -1
    vmin = -1e-5
    vmax = 1e-5
    linthresh = 1e-14
    linscale = 1e-14
    levels = np.concatenate((np.flip(neg), np.zeros(1, ), pos))
    for axs in ax:
        #axs.set_facecolor('#BEDBE4')
        axs.set_xlim(0,14400)
        axs.set_ylim(proc_dict['slow_uv']['alt'][60], proc_dict['slow_uv']['alt'][90] )
        axs.yaxis.set_ticks([7000, 8000, 9000, 10000 ])
        plt.setp(axs.spines.values(), linewidth=1, color='dimgrey')
        axs.tick_params(which='both', axis='both', width = 1, labelcolor = 'dimgrey',  color = 'dimgrey', direction='in')
    for  k in  ['ICNCx2_9-10','ICNCx1.5_9-10','ICNCx1.25_9-10','ICNCx1.1_9-10' ]:
        #fig.text( s = sim + ' - CTRL')
        n = n-1
        for v in ['dqi_mphys', 'dqi_diff', 'dqi_tvda', 'dqi_for']:
            if v == 'dqi_mphys':
                n = n+1
            else:
                n = n
            c = ax[n].contourf(proc_dict[k]['time_srs'][:459], proc_dict[k]['alt'][60:90],
                               ((tend_dict[k][v][:459, 60:90] - tend_dict['slow_uv'][v][:459, 60:90]).transpose()) ,
                               norm=matplotlib.colors.SymLogNorm(vmin = vmin, vmax = vmax, linthresh=linthresh, linscale=linscale),
                               cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels)
            ax[n].plot(proc_dict[k]['time_srs_lowres'], np.nanmean(proc_dict[k]['clbas'], axis=(1, 2)), color='darkgrey')
            ax[n].plot(proc_dict[k]['time_srs_lowres'], np.nanmean(proc_dict[k]['cltop'], axis=(1, 2)), color='darkgrey')
            n=n+1
    cb = plt.colorbar(c, cax = CbAx,orientation = 'horizontal', ticks = [-1e-3, -1e-6, -1e-9, 0, 1e-9, 1e-6, 1e-3])
    cb.ax.set_xlabel('Ice tendency (kg kg$^{-1}$ s$^{-1}$)', color='dimgrey', fontsize=18)
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=16, labelcolor='dimgrey', pad=10, size=3, color = 'dimgrey', direction = 'in')
    cb.update_ticks()
    cb.outline.set_linewidth(2)
    for f in [0,4,8,12]:
        ax[f].set_ylabel('Altitude (m)', color = 'dimgrey', fontsize = 16,labelpad=50, rotation = 0)
    for f, l in zip([0,1,2,3], ['microphysics', 'diffusion', 'advection', 'forcing']):
        ax[f].set_title(l, fontsize = 18, color = 'dimgrey', pad = 20, fontweight='bold')
    for f, l in zip([3,7,11,-1], ['ICNCx2_9-10','slow_uv_ICNCx1.5_9-10','slow_uv_ICNCx1.25_9-10','slow_uv_ICNCx1.1_9-10' ]):
        ax[f].text(1.05, 0.5,l + '\n - CTRL', transform = ax[f].transAxes, fontsize = 18, fontweight='bold', color = 'dimgrey')
    plt.subplots_adjust(left = 0.12, right = 0.83, bottom=0.15)
    plt.savefig(filepath + '../figures/Tendencies_dqi_minus_CTRL_uv_runs_9-10.png')
    plt.show()

tendencies_plot()

for k in proc_dict[sim].keys():
    print(k)
    t1 = np.ma.masked_where((prop_dict[sim]['ice_mmr'][51:101] < 10e-15), proc_dict[sim][k][51:101])
    print(t1.mean())
    t2 = np.ma.masked_where((prop_dict[sim]['ice_mmr'][200:252] < 10e-15), proc_dict[sim][k][200:252])
    print(t2.mean())

fig, ax = plt.subplots(2,2)
ax = ax.flatten()
for sim, axs in zip(met_dict.keys(), ax):
    c = axs.contourf((met_dict[sim]['rhi'][:, 60:95].transpose() * 1000), vmin=0, vmax = 160)

plt.colorbar(c)
plt.show()
