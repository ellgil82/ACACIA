import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
import imageio

filepath = '/storage/silver/acacia/ke923690/acacia/'

# Load files
#filename = input("Enter name of diagnostics file\n")
#variable = input("Enter variable to plot\n")
#timestep = input("Which timesteps do you need? (0-indexed)\n")

def plot_ice_mmr(prop_dict, model_hts, fig_str):
    fig, ax = plt.subplots(figsize=(5, 6))
    #for t, l in zip([44, 88, 138, 177, -1 ], ['60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
    for t, l in zip([0, 3, 6, 14, 29,  -1 ], ['1 s', '5 s', '10 s', '20 s',  '40 s',  '60 s']):
       color_list = ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal', 'brown', 'darkgrey', 'indigo', ]
       ax.plot(prop_dict['ice_mmr'][t, 70:100].data, model_hts[70:100]/1000, label = l)#, color = color_list[t])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlim(1e-8, 1e-4)
    ax.yaxis.set_ticks([7, 8, 9, 10])
    ax.set_xscale('log')
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('Ice MMR (kg/kg)')
    ax.legend(loc=4)
    ax.tick_params(which='both', axis='both', direction='in')
    plt.subplots_adjust(left = 0.2)
    plt.savefig(filepath + 'figures/ice_mmr_test_' + fig_str + '.png')

def plot_vapour_mmr(prop_dict, model_hts, fig_str):
    fig, ax = plt.subplots(figsize=(5, 6))
    #for t, l in zip([44, 88, 138, 177, -1 ], ['60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
    for t, l in zip([0, 3, 6, 14, 29,  -1 ], ['1 s', '5 s', '10 s', '20 s',  '40 s',  '60 s']):
       color_list = ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal', 'brown', 'darkgrey', 'indigo', ]
       ax.plot(prop_dict['ice_mmr'][t, 70:100].data, model_hts[70:100]/1000, label = l)#, color = color_list[t])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlim(1e-12, 1e-4)
    ax.yaxis.set_ticks([7, 8, 9, 10])
    ax.set_xscale('log')
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('Vapour MMR (kg/kg)')
    ax.legend(loc=4)
    ax.tick_params(which='both', axis='both', direction='in')
    plt.subplots_adjust(left = 0.2)
    plt.savefig(filepath + 'figures/vapour_mmr_' + fig_str + '.png')

def plot_tendencies(tend_dict, z, fig_str):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax = ax.flatten()
    #for i, t, l in zip([0,1,2,3,4,5],[0,44, 88, 138, 177, -1 ], ['60 mins', '120 mins', '180 mins', '210 mins', '240 mins', '270 mins']):
    for i, t, l in zip([0, 1, 2, 3, 4, 5], [0, 3, 6, 14, 29,  -1 ], ['1 s', '5 s', '10 s', '20 s',  '40 s',  '60 s']):
        # ax[i].plot(MPS.data[t, 60:110]*60,z[60:110], label='sed')
        # ax[i].plot(MPM.data[t, 60:110] * 60, z[60:110], label='other\nmphys')
        ax[i].plot(tend_dict['mphys'].data[t, 80:100] * 60, z[80:100], label='mphys')
        ax[i].plot(tend_dict['diff'].data[t, 80:100] * 60, z[80:100], label='diff')
        ax[i].plot(tend_dict['tvda'].data[t, 80:100] * 60, z[80:100], label='adv')
        ax[i].plot(tend_dict['tot'].data[t, 80:100] * 60, z[80:100], label='tot')
        ax[i].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax[i].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax[i].set_xlim(-2.5e-4, 2.5e-4)
        ax[i].set_title(l)
        # ax[i].yaxis.set_ticks([7, 8, 9, 10])
        # ax[i].set_xscale('log')
        ax[i].set_ylabel('Altitude (km)', rotation=90)
        ax[i].set_xlabel('Tendency (kg/k/min)')
        ax[i].legend(loc=4)
        ax[i].tick_params(which='both', axis='both', direction='in')
    #ax[0].set_xlim(-5e-6, 5e-6)
    plt.subplots_adjust(left=0.2, top=0.95, hspace=0.25, wspace=0.25)
    plt.savefig(filepath + 'figures/tendencies_'+ fig_str + '.png')

def plot_mphys_tendencies(tend_dict, z, fig_str):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax = ax.flatten()
    for i, t, l in zip([0, 1, 2, 3, 4, 5], [0, 3, 6, 14, 29,  -1 ], ['1 s', '5 s', '10 s', '20 s',  '40 s',  '60 s']):
        # for i, t, l in zip([0, 1, 2, 3, 4, 5], [0, 44, 88, 138, 177, -1], ['1 mins', '60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
        ax[i].plot(tend_dict['MPS'].data[t, 70:100] * 60, z[70:100], label='sed')
        ax[i].plot(tend_dict['MPM'].data[t, 70:100] * 60, z[70:100], label='other\nmphys')
        ax[i].plot(tend_dict['mphys'].data[t, 70:100] * 60, z[70:100], label='mphys_tot', color='k')
        ax[i].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax[i].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax[i].set_xlim(-2.5e-4, 2.5e-4)
        ax[i].set_title(l)
        # ax[i].yaxis.set_ticks([7, 8, 9, 10])
        # ax[i].set_xscale('log')
        ax[i].set_ylabel('Altitude (km)', rotation=90)
        ax[i].set_xlabel('Tendency (kg/k/min)')
        ax[i].legend(loc=4)
        ax[i].tick_params(which='both', axis='both', direction='in')
    #ax[0].set_xlim(-5e-5, 5e-5)
    plt.subplots_adjust(left=0.2, top=0.95, hspace=0.25, wspace=0.25)
    plt.savefig(filepath + 'figures/mphys_tendencies_' + fig_str + '.png')

def plot_processes(proc_dict, z, fig_str):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax = ax.flatten()
    color_list = ['red',  'lightblue', 'purple', 'orange','darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal',
                  'brown','lime', 'darkgrey', 'indigo', 'darkred', 'cyan', 'bisque', 'lightsteelblue', 'turquoise',
                  'forestgreen',  'coral', 'maroon']
    lgd_labs = []
    lgd_lns = []
    for i, t, l in zip([0, 1, 2, 3, 4, 5], [0, 3, 6, 14, 29,  -1 ], ['1 s', '5 s', '10 s', '20 s',  '40 s',  '60 s']):
        # for i, t, l in zip([0, 1, 2, 3, 4, 5], [0, 44, 88, 138, 177, -1], ['1 mins', '60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
        for c, j in enumerate(proc_dict.keys()):
            if proc_dict[j][t].data.max() > 1e-12:
                if j not in lgd_labs:
                    lgd_labs.append(j)
                    lgd_lns.append(color_list[c])
                ax[i].plot(proc_dict[j][t, 70:100].data, z[70:100]/ 1000, label=j, color=color_list[c])
                ax[i].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
                ax[i].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
                ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                ax[i].set_xlim(-5e-6, 5e-6)
                ax[i].set_title(l)
                ax[i].set_ylabel('Altitude (km)', rotation=90)
                ax[i].set_xlabel('Process rate (kg/kg/s)')
                ax[i].tick_params(which='both', axis='both', direction='in')
    lns = []
    labs = []
    for l, s in zip(lgd_lns, lgd_labs):
        lns.append( Line2D([0], [0], color=l, linewidth=2.5))
        labs.append( s)
        lgd = ax[-1].legend(lns, labs, bbox_to_anchor=(1.05, 1.), loc=2)
    plt.subplots_adjust(left=0.1, right=0.7)
    plt.savefig(filepath + 'figures/process_rates_' + fig_str + '.png')

def plot_rh(met_dict, z, fig_str):
    fig, ax = plt.subplots(figsize=(5, 6))
    #for t, l in zip([44, 88, 138, 177, -1 ], ['60 mins', '120 mins', '180 mins', '210 mins', '240 mins']):
    for t, l in zip([0, 3, 6, 14, 29,  -1 ], ['1 s', '5 s', '10 s', '20 s',  '40 s',  '60 s']):
       color_list = ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal', 'brown', 'darkgrey', 'indigo', ]
       ax.plot(met_dict['rh'][t, 70:100].data*100, z[70:100]/1000, label = l)#, color = color_list[t])
    #ax.set_xlim(0, 120)
    ax.yaxis.set_ticks([7, 8, 9, 10])
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('RH (%)')
    ax.legend(loc=4)
    ax.tick_params(which='both', axis='both', direction='in')
    plt.subplots_adjust(left = 0.2)
    plt.savefig(filepath + 'figures/rh_' + fig_str + '.png')

def plot_first_ts():
    filename = 'Yang_test10_q2*60.nc'
    fig_str = 'ts_1s_q2'
    proc_dict = {}
    for name, v in zip(
            ['cond', 'ice_nucl', 'homg_fr_cl', 'ice_dep', 'snow_dep', 'ice_sed', 'snow_sed', 'graupel_sed', 'ice_subm',
             'snow_subm',
             'graupel_subm', 'ice_melt', 'snow_melt', 'graupel_melt', 'rn_autoconv', 'rn_accr', 'snow_accr_ice',
             'sn_accr_rn', 'gr_accr_cl', 'gr_accr_sn', 'ice_acc_w', 'ice_to_snow'],
            ['pcond_mean', 'pinuc_mean', 'phomc_mean', 'pidep_mean', 'psdep_mean', 'psedi_mean', 'pseds_mean',
             'psedg_mean',
             'pisub_mean', 'pssub_mean', 'pgsub_mean', 'pimlt_mean', 'psmlt_mean', 'pgmlt_mean', 'praut_mean',
             'pracw_mean', 'psaci_mean', 'psacr_mean',
             'pgacw_mean', 'pgacs_mean', 'piacw_mean', 'psaut_mean']):
        proc_dict[name] = iris.load_cube(filepath + filename, v)
    prop_dict = {}
    for name, v in zip(['ice_mmr', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean'],
                       ['ice_mmr_mean', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean']):
        prop_dict[name] = iris.load_cube(filepath + filename, v)
    ## Relevant met
    met_dict = {}
    for name, v in zip(['theta', 'w', 'rh'], ['theta_mean', 'w_wind_mean', 'rh_mean']):
        met_dict[name] = iris.load_cube(filepath + filename, v)
    model_hts = prop_dict['ice_mmr'].coord('zn').points + 50  # add 50 to offset centring in middle of level
    z = model_hts
    # Load tendencies
    tend_dict = {}
    for name, v in zip(['mphys', 'diff', 'tot', 'tvda', 'tvda_3d', 'th_adv_3d', 'pw_adv_3d',
                        'diff_3d', 'tot_3d'],
                       ['dqi_mphys_mean', 'tend_qi_diff_mean', 'tend_qi_total_mean', 'tend_qi_tvda_mean', ]):
        tend_dict[name] = iris.load_cube(filepath + filename, v)
        tend_dict['MPS'] = proc_dict['ice_sed'] + proc_dict['snow_sed'] + proc_dict['graupel_sed']
        tend_dict['MPM'] = tend_dict['mphys'] - tend_dict['MPS']
    plot_ice_mmr(prop_dict,z, fig_str)
    plot_vapour_mmr(prop_dict,z, fig_str)
    plot_processes(proc_dict,z, fig_str)
    plot_tendencies(tend_dict,z, fig_str)
    plot_mphys_tendencies(tend_dict,z, fig_str)
    plot_rh(met_dict,z, fig_str)
    plt.show()

plot_first_ts()

coarse_dust_number = iris.load_cube(filepath+filename, 'q_coarse_dust_number')
coarse_dust_mass = iris.load_cube(filepath+filename, 'q_coarse_dust_mass')
coarse_insol_number = iris.load_cube(filepath+filename, 'q_coarse_insol_number')
coarse_insol_mass = iris.load_cube(filepath+filename, 'q_coarse_insol_mass')
coarse_sol_number = iris.load_cube(filepath+filename, 'q_coarse_sol_number')
coarse_sol_mass = iris.load_cube(filepath+filename, 'q_coarse_sol_mass')
accum_sol_number = iris.load_cube(filepath+filename, 'q_accum_sol_number')
accum_sol_mass = iris.load_cube(filepath+filename, 'q_accum_sol_mass')

plt.plot(accum_sol_number.data.mean(axis=(1,2)).max(axis = 1))



'''ice_mmr = iris.load_cube(filepath+filename, 'ice_mmr_mean')
snow_mmr = iris.load_cube(filepath+filename, 'snow_mmr_mean')
graupel_mmr = iris.load_cube(filepath+filename, 'graupel_mmr_mean')
dqi_mphys_mean = iris.load_cube(filepath+filename, 'dqi_mphys_mean') # is this the total change in ice concentration from mphys? Yes. tendencies.
dqv_mphys_mean = dqi_mphys_mean = iris.load_cube(filepath+filename, 'dqv_mphys_mean')
wv_mmr = iris.load_cube(filepath+filename, 'vapour_mmr_mean')
CF = iris.load_cube(filepath+filename, 'total_cloud_fraction')

# Condensation
proc_dict['cond'] = iris.load_cube(filepath + filename , 'pcond_mean') # source of cloud water and suspended aerosol particles
# heterogeneous ice nucleation rate (flux)
proc_dict['ice_nucl'] = iris.load_cube(filepath + filename , 'pinuc_mean')
# homogeneous ice nucleation rate (flux) <-- including freezing of aqueous aerosol droplets?
proc_dict['homg_freezing'] = iris.load_cube(filepath + filename , 'phomc_mean') # homogeneous freezing of cloud droplets -- assume negligible homog freezing of rain
# advection (?)
# Deposition
proc_dict['ice_dep'] = iris.load_cube(filepath + filename , 'pidep_mean')

## Sinks of water
# sedimentation -- assume negligible flux of cloud and rain particle sedimentation
proc_dict['ice_sed'] = iris.load_cube(filepath + filename , 'psedi_mean')
proc_dict['snow_sed'] = iris.load_cube(filepath + filename , 'pseds_mean')
proc_dict['graupel_sed'] = iris.load_cube(filepath + filename , 'psedg_mean')
# sublimation
proc_dict['ice_subm'] = iris.load_cube(filepath + filename , 'pisub_mean')
proc_dict['snow_subm'] = iris.load_cube(filepath + filename , 'pssub_mean')
proc_dict['graupel_subm'] = iris.load_cube(filepath + filename , 'pgsub_mean')
# Melting
proc_dict['ice_melt'] = iris.load_cube(filepath + filename , 'pimlt_mean')
proc_dict['snow_melt'] = iris.load_cube(filepath + filename , 'psmlt_mean')
proc_dict['graupel_melt'] = iris.load_cube(filepath + filename , 'pgmlt_mean')


# Precipitation

## Other processes
# Aggregation
# Accretion
# Hallet-Mossop (probably not important in expected temperature range)'''