import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
import imageio

filepath = '/storage/silver/acacia/ke923690/acacia/new_runs/'

proc_dict = {}
tend_dict = {}
for filename, sim in zip(['file_str1', 'file_str2' ], ['CTRL', 'sim1']):
    ## Sources of water
    proc_dict[sim] = {}
    tend_dict[sim] = {}
    for name, v in zip(['ice_nucl', 'homg_fr_cl',  'ice_dep', 'snow_dep',  'ice_sed', 'snow_sed', 'graupel_sed', 'ice_subm', 'snow_subm',
                        'graupel_subm',   'snow_accr_ice',   'gr_accr_sn', 'ice_acc_w', 'ice_to_snow', 'clbas', 'cltop'],[ 'pinuc_mean', 'phomc_mean', 'pidep_mean', 'psdep_mean', 'psedi_mean', 'pseds_mean', 'psedg_mean',
                        'pisub_mean', 'pssub_mean', 'pgsub_mean', 'psaci_mean', 'pgacs_mean', 'piacw_mean', 'psaut_mean',  'clbas', 'cltop']):
        f1 = iris.load_cube(filepath + filename + '?d_3600.nc', v)
        f2 = iris.load_cube(filepath + filename + '?d_7200.nc', v)
        f3 = iris.load_cube(filepath + filename + '?d_10800.nc', v)
        proc_dict[sim][name] = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
        try:
            proc_dict[sim]['time_srs'] = np.concatenate((f1.coord('time_series_10_30').points,
                                                        f2.coord('time_series_10_30').points,
                                                        f3.coord('time_series_10_30').points), axis=0)
            proc_dict[sim]['alt'] = f1.coord('zn').points + 50
        except:
            proc_dict[sim]['time_srs_lowres'] = np.concatenate((f1.coord('time_series_300_300').points,
                                                               f2.coord('time_series_300_300').points,
                                                               f3.coord('time_series_300_300').points), axis=0)
    proc_dict[sim]['sed'] = proc_dict[sim]['graupel_sed'] + proc_dict[sim]['ice_sed'] + proc_dict[sim]['snow_sed']
    proc_dict[sim]['subm'] = proc_dict[sim]['graupel_subm'] + proc_dict[sim]['ice_subm'] + proc_dict[sim]['snow_subm']
    proc_dict[sim]['dep'] = proc_dict[sim]['ice_dep'] + proc_dict[sim]['snow_dep']
    proc_dict[sim]['gr'] = proc_dict[sim]['ice_to_snow'] + proc_dict[sim]['snow_accr_ice']
    for v in ['sed', 'subm', 'dep', 'gr']:
            proc_dict[sim][v][proc_dict[sim][v] == 0.] = np.nan
    for name, v in zip(['dqi_mphys','dqi_diff', 'dqi_tot', 'dqi_tvda', 'dqi_for','dqs_mphys','dqs_diff', 'dqs_tot', 'dqs_tvda', 'dqs_for',
                        'dqg_mphys','dqg_diff', 'dqg_tot', 'dqg_tvda', 'dqg_for',],
                       [ 'dqi_mphys_mean', 'tend_qi_diff_mean', 'tend_qi_total_mean','tend_qi_tvda_mean', 'tend_qi_for_mean',
                         'dqs_mphys_mean', 'tend_qs_diff_mean', 'tend_qs_total_mean', 'tend_qs_tvda_mean', 'tend_qs_for_mean',
                         'dqg_mphys_mean', 'tend_qg_diff_mean', 'tend_qg_total_mean', 'tend_qg_tvda_mean', 'tend_qg_for_mean']):
        f1 = iris.load_cube(filepath + filename + '?d_3600.nc', v)
        f2 = iris.load_cube(filepath + filename + '?d_7200.nc', v)
        f3 = iris.load_cube(filepath + filename + '?d_10800.nc', v)
        tend_dict[sim][name] = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
        tend_dict[sim][name][tend_dict[sim][name] == 0.] = np.nan
    tend_dict[sim]['mphys'] = tend_dict[sim]['dqi_mphys'] + tend_dict[sim]['dqs_mphys' ]+ tend_dict[sim]['dqg_mphys']
    tend_dict[sim]['diff'] = tend_dict[sim]['dqi_diff'] + tend_dict[sim]['dqs_diff' ]+ tend_dict[sim]['dqg_diff']
    tend_dict[sim]['tvda'] = tend_dict[sim]['dqi_tvda'] + tend_dict[sim]['dqs_tvda' ]+ tend_dict[sim]['dqg_tvda']
    tend_dict[sim]['for'] = tend_dict[sim]['dqi_for'] + tend_dict[sim]['dqs_for' ]+ tend_dict[sim]['dqg_for']
    tend_dict[sim]['tot'] = tend_dict[sim]['dqi_tot'] + tend_dict[sim]['dqs_tot' ]+ tend_dict[sim]['dqg_tot']

import scipy.interpolate as intp

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
        axs.set_facecolor('#BEDBE4')
        axs.set_xlim(0,10800)
        axs.set_ylim(proc_dict['CTRL']['alt'][60], proc_dict['CTRL']['alt'][90] )
        axs.yaxis.set_ticks([7000, 8000, 9000, 10000 ])
        plt.setp(axs.spines.values(), linewidth=1, color='dimgrey')
        axs.tick_params(which='both', axis='both', width = 1, labelcolor = 'dimgrey',  color = 'dimgrey', direction='in')
    for  k in  [ 'slow_uv', 'slow_uv_ICNCx2_8-9','slow_uv_ICNCx2_9-10', 'slow_uv_no_qF']:
        #fig.text( s = sim + ' - CTRL')
        n = n-1
        for v in ['dep', 'gr', 'sed', 'subm']:
            # upsample ctrl for subtracting higher-res variables
            a = np.copy(proc_dict['CTRL'][v])
            x = np.array(range(a.shape[0]))
            xnew = np.linspace(x.min(), x.max(), 342)
            f = intp.interp1d(x, a, axis=0)
            upsampled_ctrl = f(xnew)
            if v == 'dep':
                n = n+1
            else:
                n = n
            if k == 'CTRL':
                c = ax[n].contourf(proc_dict[k]['time_srs'], proc_dict[k]['alt'][60:90], (proc_dict[k][v][:281, 60:90].transpose()),  cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels)
            else:
                c = ax[n].contourf(proc_dict[k]['time_srs'], proc_dict[k]['alt'][60:90],
                                   ((proc_dict[k][v][:, 60:90] - upsampled_ctrl[:, 60:90]).transpose()),
                                   norm=matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh,
                                                                     linscale=linscale), cmap='bwr', vmin=vmin,
                                   vmax=vmax, levels=levels)
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
    for f, l in zip([3,7,11,-1], [ 'slow_uv', 'slow_uv_ICNCx2_8-9','slow_uv_ICNCx2_9-10', 'slow_uv_no_qF']):
        ax[f].text(1.05, 0.5,l + '\n - CTRL',  transform = ax[f].transAxes, fontsize = 18, fontweight='bold', color = 'dimgrey')
    plt.subplots_adjust(left = 0.12, right = 0.83, bottom=0.15)
    plt.savefig(filepath + '../figures/Process_rates_uv_runs.png')
    plt.show()

mega_plot()

def plot_proc_contour(var):
    fig, ax = plt.subplots(2, 2, figsize=(6, 8), sharex=True, sharey=True)
    if var == 'dep':
        pos = np.geomspace(1e-9, 1e-5, 40)
        neg = pos * -1
        vmin = -1e-5
        vmax = 1e-5
        linthresh = 1e-7
        linscale = 1e-3
    elif var == 'sed':
        pos = np.geomspace(1e-9, 1e-3, 40)
        neg = pos * -1
        vmin = -1e-3
        vmax = 1e-3
        linthresh = 1e-7
        linscale = 1e-5
    levels = np.concatenate((np.flip(neg), np.zeros(1, ), pos))
    ax = ax.flatten()
    cbax = fig.add_axes([0.3, 0.08, 0.5, 0.02])
    for k, axs in zip(['CTRL', 'slow_uv', 'slow_uv_ICNCx2_8-9','slow_uv_ICNCx2_9-10', ], ax):
        c = axs.contourf(proc_dict[sim][k]['time_srs'][2:281], alt[70:90], (proc_dict[sim][k]['sed'][2:281, 70:90] - proc_dict[sim]['CTRL']['sed'][2:281, 70:90]).transpose(),
                         norm=matplotlib.colors.SymLogNorm(vmin = vmin, vmax = vmax,linthresh=linthresh, linscale=linscale), cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels)
        axs.set_title(k + ' - CTRL')
        axs.set_facecolor('#BEDBE4')
        axs.plot(proc_dict[sim][k]['time_srs_lowres'],np.nanmean(proc_dict[sim][k]['clbas'], axis=(1, 2)), color='darkgrey')
        axs.plot(proc_dict[sim][k]['time_srs_lowres'],np.nanmean(proc_dict[sim][k]['cltop'], axis=(1, 2)), color='darkgrey')
        axs.set_xlim(0,10800)
    cb = plt.colorbar(c, cax=cbax, orientation='horizontal', extend='both', ticks = [-1e-5, 0, 1e-5])
    plt.subplots_adjust(top=0.95, bottom = 0.2)
    plt.savefig(filepath + '../figures/Hovmoller_' + var + '_uv_runs.png')
    plt.show()

plot_proc_contour('sed')

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
        axs.set_facecolor('#BEDBE4')
        axs.set_xlim(0,10800)
        axs.set_ylim(proc_dict[sim]['CTRL']['alt'][60], proc_dict[sim]['CTRL']['alt'][90] )
        axs.yaxis.set_ticks([7000, 8000, 9000, 10000 ])
        plt.setp(axs.spines.values(), linewidth=1, color='dimgrey')
        axs.tick_params(which='both', axis='both', width = 1, labelcolor = 'dimgrey',  color = 'dimgrey', direction='in')
    for  k in  ['CTRL', 'slow_uv', 'slow_uv_ICNCx2_8-9','slow_uv_ICNCx2_9-10',]:
        #fig.text( s = sim + ' - CTRL')
        n = n-1
        for v in ['dqg_mphys', 'dqg_diff', 'dqg_tvda', 'dqg_for']:
            if v == 'dqg_mphys':
                n = n+1
            else:
                n = n
            c = ax[n].contourf(proc_dict[sim][k]['time_srs'][:281], proc_dict[sim][k]['alt'][60:90],
                               ((tend_dict[sim][k][v][:281, 60:90] - tend_dict[sim]['CTRL'][v][:281, 60:90]).transpose()) ,
                               norm=matplotlib.colors.SymLogNorm(vmin = vmin, vmax = vmax, linthresh=linthresh, linscale=linscale),
                               cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels)
            ax[n].plot(proc_dict[sim][k]['time_srs_lowres'], np.nanmean(proc_dict[sim][k]['clbas'], axis=(1, 2)), color='darkgrey')
            ax[n].plot(proc_dict[sim][k]['time_srs_lowres'], np.nanmean(proc_dict[sim][k]['cltop'], axis=(1, 2)), color='darkgrey')
            n=n+1
    cb = plt.colorbar(c, cax = CbAx,orientation = 'horizontal', ticks = [-1e-3, -1e-6, -1e-9, 0, 1e-9, 1e-6, 1e-3])
    cb.ax.set_xlabel('Graupel tendency (kg kg$^{-1}$ s$^{-1}$)', color='dimgrey', fontsize=18)
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=16, labelcolor='dimgrey', pad=10, size=3, color = 'dimgrey', direction = 'in')
    cb.update_ticks()
    cb.outline.set_linewidth(2)
    for f in [0,4,8,12]:
        ax[f].set_ylabel('Altitude (m)', color = 'dimgrey', fontsize = 16,labelpad=50, rotation = 0)
    for f, l in zip([0,1,2,3], ['microphysics', 'diffusion', 'advection', 'forcing']):
        ax[f].set_title(l, fontsize = 18, color = 'dimgrey', pad = 20, fontweight='bold')
    for f, l in zip([3,7,11,-1], ['CTRL', 'slow_uv', 'slow_uv_ICNCx2_8-9','slow_uv_ICNCx2_9-10',]):
        ax[f].text(1.05, 0.5,l + '\n - CTRL', transform = ax[f].transAxes, fontsize = 18, fontweight='bold', color = 'dimgrey')
    plt.subplots_adjust(left = 0.12, right = 0.83, bottom=0.15)
    plt.savefig(filepath + '../figures/Tendencies_dqg_minus_CTRL_uv_runs.png')
    plt.show()

tendencies_plot()