import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
import imageio

filepath = '/storage/silver/acacia/ke923690/acacia/'

file1 = 'Yang_test14_DM15_no_proc_'
fig_str1 = 'fixed_cl_num_ctrl'
proc_dict1 = {}
for name, v in zip(
        ['cond', 'ice_nucl', 'homg_fr_cl', 'ice_dep', 'snow_dep', 'ice_sed', 'snow_sed', 'graupel_sed', 'ice_subm',
         'snow_subm', 'graupel_subm', 'ice_melt', 'snow_melt', 'graupel_melt', 'rn_autoconv', 'rn_accr', 'snow_accr_ice',
         'sn_accr_rn', 'gr_accr_cl', 'gr_accr_sn', 'ice_acc_w', 'ice_to_snow', 'ninuc'],
        ['pcond_mean', 'pinuc_mean', 'phomc_mean', 'pidep_mean', 'psdep_mean', 'psedi_mean', 'pseds_mean',
         'psedg_mean',
         'pisub_mean', 'pssub_mean', 'pgsub_mean', 'pimlt_mean', 'psmlt_mean', 'pgmlt_mean', 'praut_mean',
         'pracw_mean', 'psaci_mean', 'psacr_mean',
         'pgacw_mean', 'pgacs_mean', 'piacw_mean', 'psaut_mean', 'ninuc_mean']):
    f1 = iris.load_cube(filepath + file1 + '?d_3600.nc', v)
    f2 = iris.load_cube(filepath + file1 + '?d_7200.nc', v)
    f3 = iris.load_cube(filepath + file1 + '?d_10800.nc', v)
    proc_dict1[name] = np.concatenate((f1.data, f2.data, f3.data), axis=0)

prop_dict1 = {}
for name, v in zip(['ice_mmr', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP', 'VWP', 'ice_nc_mean', 'snow_nc_mean', 'graupel_nc_mean'],
                   ['ice_mmr_mean', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP_mean', 'VWP_mean','ice_nc_mean', 'snow_nc_mean', 'graupel_nc_mean']):
    f1 = iris.load_cube(filepath + file1 + '?d_3600.nc', v)
    f2 = iris.load_cube(filepath + file1 + '?d_7200.nc', v)
    f3 = iris.load_cube(filepath + file1 + '?d_10800.nc', v)
    prop_dict1[name] = np.concatenate((f1.data, f2.data, f3.data), axis=0)

## Relevant met
met_dict1 = {}
for name, v in zip(['theta', 'w', 'rh', 'temp', 'rhi'], ['theta_mean', 'w_wind_mean', 'rh_mean', 'temperature', 'rhi_mean']):
    f1 = iris.load_cube(filepath + file1 + '?d_3600.nc', v)
    f2 = iris.load_cube(filepath + file1 + '?d_7200.nc', v)
    f3 = iris.load_cube(filepath + file1 + '?d_10800.nc', v)
    met_dict1[name] = np.concatenate((f1.data, f2.data, f3.data), axis=0)

model_hts = f1.coord('zn').points + 50  # add 50 to offset centring in middle of level
z = model_hts
# Load tendencies
tend_dict1 = {}
for name, v in zip(['mphys', 'diff', 'tot', 'tvda', 'tvda_3d', 'th_adv_3d', 'pw_adv_3d',
                    'diff_3d', 'tot_3d'],
                   ['dqi_mphys_mean', 'tend_qi_diff_mean', 'tend_qi_total_mean', 'tend_qi_tvda_mean', ]):
    f1 = iris.load_cube(filepath + file1 + '?d_3600.nc', v)
    f2 = iris.load_cube(filepath + file1 + '?d_7200.nc', v)
    f3 = iris.load_cube(filepath + file1 + '?d_10800.nc', v)
    tend_dict1[name] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
    tend_dict1['MPS'] = proc_dict1['ice_sed'] + proc_dict1['snow_sed'] + proc_dict1['graupel_sed']
    tend_dict1['MPM'] = tend_dict1['mphys'] - tend_dict1['MPS']

time_srs = np.concatenate((f1.coord('time_series_100_100').points, f2.coord('time_series_100_100').points,f3.coord('time_series_100_100').points), axis=0)

file2 ='Yang_test14b_DM15_no_proc_pert_'
fig_str2 = 'fixed_cl_num_pert'
proc_dict2 = {}
for name, v in zip(
        ['cond', 'ice_nucl', 'homg_fr_cl', 'ice_dep', 'snow_dep', 'ice_sed', 'snow_sed', 'graupel_sed', 'ice_subm',
         'snow_subm', 'graupel_subm', 'ice_melt', 'snow_melt', 'graupel_melt', 'rn_autoconv', 'rn_accr', 'snow_accr_ice',
         'sn_accr_rn', 'gr_accr_cl', 'gr_accr_sn', 'ice_acc_w', 'ice_to_snow', 'ninuc'],
        ['pcond_mean', 'pinuc_mean', 'phomc_mean', 'pidep_mean', 'psdep_mean', 'psedi_mean', 'pseds_mean',
         'psedg_mean',
         'pisub_mean', 'pssub_mean', 'pgsub_mean', 'pimlt_mean', 'psmlt_mean', 'pgmlt_mean', 'praut_mean',
         'pracw_mean', 'psaci_mean', 'psacr_mean',
         'pgacw_mean', 'pgacs_mean', 'piacw_mean', 'psaut_mean', 'ninuc_mean']):
    f1 = iris.load_cube(filepath + file2 + '?d_3600.nc', v)
    f2 = iris.load_cube(filepath + file2 + '?d_7200.nc', v)
    f3 = iris.load_cube(filepath + file2 + '?d_10800.nc', v)
    proc_dict2[name] = np.concatenate((f1.data, f2.data, f3.data), axis=0)

prop_dict2 = {}
for name, v in zip(['ice_mmr', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP', 'VWP', 'ice_nc_mean', 'snow_nc_mean', 'graupel_nc_mean'],
                   ['ice_mmr_mean', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP_mean', 'VWP_mean','ice_nc_mean', 'snow_nc_mean', 'graupel_nc_mean']):
    f1 = iris.load_cube(filepath + file2 + '?d_3600.nc', v)
    f2 = iris.load_cube(filepath + file2 + '?d_7200.nc', v)
    f3 = iris.load_cube(filepath + file2 + '?d_10800.nc', v)
    prop_dict2[name] = np.concatenate((f1.data, f2.data, f3.data), axis=0)

## Relevant met
met_dict2 = {}
for name, v in zip(['theta', 'w', 'rh', 'temp', 'rhi'], ['theta_mean', 'w_wind_mean', 'rh_mean', 'temperature', 'rhi_mean']):
    f1 = iris.load_cube(filepath + file2 + '?d_3600.nc', v)
    f2 = iris.load_cube(filepath + file2 + '?d_7200.nc', v)
    f3 = iris.load_cube(filepath + file2 + '?d_10800.nc', v)
    met_dict2[name] = np.concatenate((f1.data, f2.data, f3.data), axis=0)

model_hts = f1.coord('zn').points + 50  # add 50 to offset centring in middle of level
z = model_hts
# Load tendencies
tend_dict2 = {}
for name, v in zip(['mphys', 'diff', 'tot', 'tvda', 'tvda_3d', 'th_adv_3d', 'pw_adv_3d',
                    'diff_3d', 'tot_3d'],
                   ['dqi_mphys_mean', 'tend_qi_diff_mean', 'tend_qi_total_mean', 'tend_qi_tvda_mean', ]):
    f1 = iris.load_cube(filepath + file2 + '?d_3600.nc', v)
    f2 = iris.load_cube(filepath + file2 + '?d_7200.nc', v)
    f3 = iris.load_cube(filepath + file2 + '?d_10800.nc', v)
    tend_dict2[name] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
    tend_dict2['MPS'] = proc_dict2['ice_sed'] + proc_dict2['snow_sed'] + proc_dict2['graupel_sed']
    tend_dict2['MPM'] = tend_dict2['mphys'] - tend_dict2['MPS']

def plot_ice_mmr(dict1, dict2, fig_str1, fig_str2, times):
    fig, ax = plt.subplots(figsize=(5, 6))
    colours = ['orange', 'darkgreen', 'darkred', 'indigo', 'purple', 'darkgrey', 'magenta']
    for t, c in zip(times, colours):
        ax.plot(dict1['ice_mmr'][t, 55:90], model_hts[55:90]/1000, color = c, label = str(int(time_srs[t])) + ' s')
        ax.plot(dict2['ice_mmr'][t, 55:90], model_hts[55:90] / 1000, color = c, linestyle = '--')
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
    plt.savefig(filepath + 'figures/ice_mmr_comparison_' + fig_str1 + '_vs_' + fig_str2 + '.png')

def plot_ICNC(dict1, dict2, fig_str1, fig_str2, times):
    fig, ax = plt.subplots(figsize=(5, 6))
    colours = ['orange', 'darkgreen', 'darkred', 'indigo', 'purple', 'darkgrey', 'magenta']
    for t, c in zip(times, colours):
        ax.plot(dict1['ice_nc_mean'][t, 55:90], model_hts[55:90]/1000, color = c, label = str(int(time_srs[t])) + ' s')
        ax.plot(dict2['ice_nc_mean'][t, 55:90], model_hts[55:90] / 1000, color = c, linestyle = '--')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlim(0, 1e7)
    ax.set_ylim(model_hts[55]/1000, model_hts[90]/1000)
    ax.yaxis.set_ticks([7, 8, 9, 10,])
    #ax.set_xscale('log')
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('ICNC (cm$^{-3}$)')
    ax.legend(loc=4)
    ax.tick_params(which='both', axis='both', direction='in')
    plt.subplots_adjust(left = 0.2)
    plt.savefig(filepath + 'figures/ICNC_comparison_' + fig_str1 + '_vs_' + fig_str2 + '.png')

plot_ice_mmr(dict1=prop_dict1, dict2=prop_dict2, fig_str1='14a_CTRL', fig_str2='14b_pert_r', times = [9, 27, 45, 54,63])
plot_ICNC(dict1=prop_dict1, dict2=prop_dict2, fig_str1='14a_CTRL', fig_str2='14b_pert_r', times = [9, 27, 45, 54, 63])
















f3 = 'Yang_test11c*CP*3600.nc'
fig_str3 = 'subs_thF'
prop_dict3 = {}
for name, v in zip(['ice_mmr', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP', 'VWP'],
                   ['ice_mmr_mean', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP_mean', 'VWP_mean']):
    prop_dict3[name] = iris.load_cube(filepath + f3, v)

## Relevant met
met_dict3 = {}
for name, v in zip(['theta', 'w', 'rh', 'temp', 'rhi'], ['theta_mean', 'w_wind_mean', 'rh_mean', 'temperature', 'rhi_mean']):
    met_dict3[name] = iris.load_cube(filepath + f3, v)

f4 = 'Yang_test11d*CP*3600.nc'
fig_str4 = 'subs_qthF'
prop_dict4 = {}
for name, v in zip(['ice_mmr', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP', 'VWP'],
                   ['ice_mmr_mean', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP_mean', 'VWP_mean']):
    prop_dict4[name] = iris.load_cube(filepath + f4, v)

## Relevant met
met_dict4 = {}
for name, v in zip(['theta', 'w', 'rh', 'temp', 'rhi'], ['theta_mean', 'w_wind_mean', 'rh_mean', 'temperature', 'rhi_mean']):
    met_dict4[name] = iris.load_cube(filepath + f4, v)

f5 = 'Yang_test11e*DM_?d_60.nc'
fig_str5 = 'subs_qF_DM'
prop_dict5 = {}
for name, v in zip(['ice_mmr', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP', 'VWP'],
                   ['ice_mmr_mean', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP_mean', 'VWP_mean']):
    prop_dict5[name] = iris.load_cube(filepath + f5, v)

## Relevant met
met_dict5 = {}
for name, v in zip(['theta', 'w', 'rh', 'temp', 'rhi'], ['theta_mean', 'w_wind_mean', 'rh_mean', 'temperature', 'rhi_mean']):
    met_dict5[name] = iris.load_cube(filepath + f5, v)


f6 = 'Yang_test11e*_upd_aer*60.nc'
fig_str6 = 'subs_qF_DM_upd_aer'
prop_dict6 = {}
for name, v in zip(['ice_mmr', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP', 'VWP'],
                   ['ice_mmr_mean', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'IWP_mean', 'VWP_mean']):
    prop_dict6[name] = iris.load_cube(filepath + f6, v)

## Relevant met
met_dict6 = {}
for name, v in zip(['theta', 'w', 'rh', 'temp', 'rhi'], ['theta_mean', 'w_wind_mean', 'rh_mean', 'temperature', 'rhi_mean']):
    met_dict6[name] = iris.load_cube(filepath + f6, v)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(prop_dict1['IWP'].coord('time_series').points, prop_dict1['IWP'].data, lw=2, label = figstr1)
ax.plot(prop_dict2['IWP'].coord('time_series').points, prop_dict2['IWP'].data, lw=2, label = figstr2)
ax.plot(prop_dict3['IWP'].coord('time_series').points, prop_dict2['IWP'].data, lw=2, label = figstr3)
ax.plot(prop_dict4['IWP'].coord('time_series').points, prop_dict2['IWP'].data, lw=2, label = figstr4)
ax.plot(prop_dict5['IWP'].coord('time_series').points, prop_dict2['IWP'].data, lw=2, label = figstr5)
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.set_xlim(0, 150)
ax.set_xlabel('Time (s)', color='dimgrey', fontsize = 18)
ax.set_ylabel('IWP\n(kg/m$^{2}$)', rotation =0, labelpad=50, color='dimgrey', fontsize = 18)
plt.subplots_adjust(left = 0.25, bottom = 0.15, right=0.97)
plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
ax.tick_params(axis='both', which='both', labelsize=14, labelcolor='dimgrey', pad=10,  color='dimgrey', length=5, width = 2.5)
#plt.savefig(filepath + 'figures/IWP_test4_v_7_Cooper_21600s.png')
plt.show()


ice_dict ={}
for nm, fn in zip(['11a', '11b', '11c', '11d', '11e', '11f -basic', '11f -upd', '11h', '11h-long'],
              ['Yang_test11a*3600.nc','Yang_test11b*CP*3600.nc', 'Yang_test11c*3600.nc',
               'Yang_test11d*3600.nc', 'Yang_test11e*3600.nc', 'Yang_test11f*_qthF*',
               'Yang_test11f*_thF*', 'Yang_test11h*3600.nc', 'Yang_test11h*14400.nc']):
    ice_dict[nm] = iris.load_cube(filepath + fn, 'ice_mmr_mean')

z = ice_dict['11a'].coord('zn').points + 60

fig, ax = plt.subplots(figsize=(5, 6))
#ax.plot(ice_dict['11a'][-1, 59:99].data, z[59:99]/1000, label = '11a')
#ax.plot(ice_dict['11b'][-1, 59:99].data, z[59:99]/1000, label = '11b')
#ax.plot(ice_dict['11c'][-1, 59:99].data, z[59:99]/1000, label = '11c')
#ax.plot(ice_dict['11d'][-1, 59:99].data, z[59:99]/1000, label = '11d')
#ax.plot(ice_dict['11e'][-1, 59:99].data, z[59:99]/1000, label = '11e')
ax.plot(ice_dict['11f -basic'][-1, 59:99].data, z[59:99]/1000, label = '11f-basic')
ax.plot(ice_dict['11f -upd'][-1, 59:99].data, z[59:99]/1000, label = '11f-upd')
#ax.plot(ice_dict['11h'][-1, 59:99].data, z[59:99]/1000, label = '11h')
ax.plot(ice_dict['11h-long'][-1, 59:99].data, z[59:99]/1000, label = '11h-long')
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_xlim(1e-8, 1e-4)
ax.yaxis.set_ticks([7, 8, 9, 10])
ax.set_xscale('log')
ax.set_xlabel('Ice MMR (kg/kg)')
ax.legend(loc=4)
ax.tick_params(which='both', axis='both', direction='in')
plt.subplots_adjust(left = 0.2)
plt.show()

z = ice_dict['11a'].coord('zn').points + 60

c = iris.load_cube(filepath + '*11i*', 'q_ice_number') #phomc
c = c.data.mean(axis=(1,2))

fig,ax = plt.subplots()
for t in range(4):
    ax.plot(c[t, 59:85].data, z[59:85])

fig.legend()
plt.show()

met_dictf = {}
met_dictf['rhi'] = iris.load_cube(filepath + '*test13b*14400.nc', 'rhi_mean')
Nice = iris.load_cube(filepath + '*test13b*14400.nc', 'q_ice_number')
Ns = iris.load_cube(filepath + '*test13b*14400.nc', 'q_snow_number')
met_dictf['Ns'] = Ns.data.mean(axis = (1,2))
met_dictf['ICNC'] = Nice.data.mean(axis = (1,2))
fig_str = '13b'

met_dictg = {}
met_dictg['rhi'] = iris.load_cube(filepath + '*test11k*14400.nc', 'rhi_mean')

z = Ns.coord('zn').points+50

fig, ax = plt.subplots(1,2, figsize = (9,5))
ax = ax.flatten()
for fn in ['*11k*','*13b*', '*13c*']:
    Nice = iris.load_cube(filepath + fn, 'q_ice_number')
    Ns = iris.load_cube(filepath + '*test11k*14400.nc', 'q_snow_number')
    ax[0].plot(Nice.data.mean(axis = (1,2))[0, 55:90].data, z[55:90]/1000, label = fn[1:3])
    ax[1].plot(Ns.data.mean(axis=(1, 2))[0, 55:90].data, z[55:90] / 1000, label=fn[1:3])

for axs in ax:
    axs.set_ylim(z[55]/1000,z[90]/1000)
    axs.yaxis.set_ticks([7, 8, 9, 10])
    axs.set_ylabel('Altitude (km)', rotation =90)
    axs.legend(loc='best')
    axs.tick_params(which='both', axis='both', direction='in')

ax[0].set_xlabel('Ice number (m$^{-3}$)')
ax[1].set_xlabel('Snow number (m$^{-3}$)')
#plt.subplots_adjust(left = 0.2)
#plt.savefig(filepath + 'figures/ICNC_evolution_' + fig_str + '.png')
plt.show()

met_dictg['Ns'] = Ns.data.mean(axis = (1,2))
met_dictg['ICNC'] = Nice.data.mean(axis = (1,2))

plt.plot((met_dictf['ICNC'][0] - met_dictg['ICNC'][0]).data[55:90], z[55:90])

fig, ax = plt.subplots(figsize = (5,6))
for t in [0,2,4,6,8,10,12,14,16,18]:
    ax.plot(met_dictf['ICNC'][t].data[55:90], z[55:90]/1000, label = str(int(Nice.coord('time_series_300_1200').points[t]/60)) + ' mins')

#ax.set_xlim(0, 120)
ax.set_ylim(model_hts[55]/1000, model_hts[90]/1000)
ax.yaxis.set_ticks([7, 8, 9, 10])
ax.set_ylabel('Altitude (km)', rotation =90)
#ax.set_xlabel('RHi (%)')
ax.set_xlabel('Ice number (m$^{-3}$)')
ax.legend(loc='best')
ax.tick_params(which='both', axis='both', direction='in')
plt.subplots_adjust(left = 0.2)
plt.savefig(filepath + 'figures/ICNC_evolution_' + fig_str + '.png')
plt.show()