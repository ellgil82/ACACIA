import numpy as np
import iris
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

filepath = '/storage/silver/acacia/ke923690/acacia/new_runs/'

# #'*14b*?d_', '*14c*?d_', '*14e*?d_', '*14f*?d_', '*14h*?d_',  '*14m*?d_', 14n*?d_' '*'SML', 'FXD', 'CDNCx2_SML', 'CDNCx2', 'INPx10',  'ICNCx1.1_8-9', 'ICNCx1.1_9-10'

RH_dict = {}
RHi_dict = {}
wRHi_dict = {}
th_dict = {}
q_dict = {}
ice_dict = {}
T_dict = {}
reske_dict = {}
for fn, nm in zip(['*17_CTRL*', '*22a*', '*22b*', '*22c*', '*22d*', '*22e*', '*22f*', '*22g*', '*22h*'],['CTRL', 'slower_uv', 'slower_uv_ICNCx2_8-9', 'slower_uv_ICNCx2_9-10', 'slower_uv_no_qF', 'slower_uv_ICNCx1.5_8-9', 'slower_uv_ICNCx1.5_9-10','slower_uv_ICNCx1.25_8-9', 'slower_uv_ICNCx1.25_9-10', ]):
    try:
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'rhi_mean')
        #rhi = iris.load_cube(filepath + fn + '3600.nc', 'rhi_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'rhi_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'rhi_mean')
        rhi = (np.concatenate((f1.data, f2.data, f3.data), axis=0) * 100) + 100
        rhi[rhi == 100] = np.nan # supersaturation
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'rh_mean')
        #rhi = iris.load_cube(filepath + fn + '3600.nc', 'rhi_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'rh_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'rh_mean')
        RH_dict[nm] = np.concatenate((f1.data, f2.data, f3.data), axis=0) * 100
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'ice_mmr_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'ice_mmr_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'ice_mmr_mean')
        #ice_mmr = iris.load_cube(filepath + fn + '3600.nc', 'ice_mmr_mean')
        ice_mmr = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
        ice_dict[nm] = ice_mmr
        RHi_dict[nm] = np.ma.average(np.ma.MaskedArray(rhi.data, mask=np.isnan(rhi.data)), axis=1)
        wRHi_dict[nm] = np.ma.average(np.ma.MaskedArray(rhi.data, mask = np.isnan(rhi.data)), weights=ice_mmr, axis=1)
        #f1 = iris.load_cube(filepath + fn + '3600.nc', 'reske')
        #f2 = iris.load_cube(filepath + fn + '7200.nc', 'reske')
        #f3 = iris.load_cube(filepath + fn + '10800.nc', 'reske')
        #reske_dict[nm] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
    except:
        print('Uh oh, sutn ain\'t right.')
#    try:
#        f1 = iris.load_cube(filepath + fn + '3600.nc', 'theta_mean')
#        f2 = iris.load_cube(filepath + fn + '7200.nc', 'theta_mean')
#        f3 = iris.load_cube(filepath + fn + '10800.nc', 'theta_mean')
#        theta = np.concatenate((f1.data, f2.data, f3.data), axis=0)
#        th_dict[nm] = np.average(theta, weights = ice_mmr, axis= 1)
#        f1 = iris.load_cube(filepath + fn + '3600.nc', 'vapour_mmr_mean')
#        f2 = iris.load_cube(filepath + fn + '7200.nc', 'vapour_mmr_mean')
#        f3 = iris.load_cube(filepath + fn + '10800.nc', 'vapour_mmr_mean')
#        q = np.concatenate((f1.data, f2.data, f3.data), axis=0)
#        q_dict[nm] = np.average(q, weights=ice_mmr, axis=1)
#        f1 = iris.load_cube(filepath + fn + '3600.nc', 'temperature')
#        f2 = iris.load_cube(filepath + fn + '7200.nc', 'temperature')
#        f3 = iris.load_cube(filepath + fn + '10800.nc', 'temperature')
#        temp = np.concatenate((f1.data, f2.data, f3.data), axis=0).mean(axis = (1,2))
#        T_dict[nm] = temp #np.average(temp, weights = ice_mmr, axis= 1)
#    except:
#        print('Uh oh, sutn ain\'t right.')
    if nm == 'CTRL':
        time_srs_ctrl = np.concatenate((f1.coord('time_series_10_30').points,f2.coord('time_series_10_30').points, f3.coord('time_series_10_30').points), axis=0)
    else:
        time_srs_hires = np.concatenate((f1.coord('time_series_10_30').points,f2.coord('time_series_10_30').points, f3.coord('time_series_10_30').points), axis=0)


plt.plot(time_srs_ctrl, RHi_dict['CTRL'].mean(axis=1), label = 'CTRL')
plt.plot(time_srs_nF, RHi_dict['noFOR'].mean(axis=1), label = 'noFOR')
plt.plot(time_srs_hires, th_dict['th-FOR'], label = 'th-FOR')
plt.plot(time_srs_q, th_dict['q-th-FOR'], label = 'q-th-FOR')
plt.plot(time_srs_incr, th_dict['q-incr'], label = 'q-incr')
plt.ylabel('Mean ice MMR-weighted RH$_{ice}$ (%)')#
plt.legend()
plt.show()

plt.plot(ice_dict['CTRL17'].mean(axis=1), label = 'CTRL17')
plt.plot(ice_dict['ICNCx2_8-9_2000'].mean(axis=1), label = 'ICNC')
plt.ylabel('Mean ice MMR-weighted RH$_{ice}$ (%)')#
plt.legend()
plt.show()

for k in wRHi_dict:
    try:
        plt.plot(time_srs_hires, wRHi_dict[k], label = k)
    except:
        plt.plot(time_srs_ctrl, wRHi_dict[k], label=k)
    plt.ylabel('Mean ice MMR-weighted RH$_{ice}$ (%)')#

plt.legend()
plt.xlim(0, 10800)
plt.savefig(filepath + '../figures/RHice_weighted.png')
plt.show()

for k in RHi_dict:
    try:
        plt.plot(time_srs_hires, RHi_dict[k], label = k)
    except:
        plt.plot(time_srs_ctrl, RHi_dict[k], label=k)
    plt.ylabel('Mean RH$_{ice}$ (%)')#

plt.legend()
plt.xlim(0, 10800)
plt.savefig(filepath + '../figures/RHice_absolute.png')
plt.show()



tend_dict = {}
for fn, nm in zip(['*15a*', '*15b*', '*15c*', '*15d*'], ['CTRL', 'th-FOR', 'q-th-FOR', 'q-incr']):
    tend_dict[nm] = {}
    try:
        for tend, v in zip(['mphys', 'diff', 'tot', 'tvda'], ['dqi_mphys_mean', 'tend_qi_diff_mean', 'tend_qi_total_mean', 'tend_qi_tvda_mean', ]):
            f1 = iris.load_cube(filepath + fn + '?d_3600.nc', v)
            f2 = iris.load_cube(filepath + fn + '?d_7200.nc', v)
            f3 = iris.load_cube(filepath + fn + '?d_10800.nc', v)
            tend_dict[nm][tend] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        if fn == '*14a*':
            time_srs_lowres = np.concatenate((f1.coord('time_series_100_100').points,f2.coord('time_series_100_100').points,f3.coord('time_series_100_100').points), axis=0)
        elif fn == '*15b*':
            time_srs_hires = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points,f3.coord('time_series_10_30').points), axis=0)
        elif fn == '*15c*':
            time_srs_q = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points,f3.coord('time_series_10_30').points), axis=0)
        elif fn == '*15d*':
            time_srs_incr = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points, f3.coord('time_series_10_30').points), axis=0)
    except:
        print('Uh oh, sutn ain\'t right.')
#,
fig, ax = plt.subplots(2,2, figsize = (7, 5))
for nm, c, t in zip(['CTRL', 'th-FOR', 'q-th-FOR', 'q-incr'], ['k', 'r', 'b', 'g'], [time_srs_hires[1:], time_srs_hires, time_srs_q, time_srs_incr]):
    ax = ax.flatten()
    for axs, tend in zip(ax, ['mphys', 'diff',  'tvda', 'tot',]):
        axs.set_title(tend)
        axs.plot(t[1:], tend_dict[nm][tend][1:].max(axis=1), color=c, label = nm)

axs.legend()
plt.subplots_adjust(wspace = 0.15, hspace = 0.25)
plt.show()

proc_dict = {}
for fn, nm in zip(['*15e*'],['noFOR']):#(['*15a*', '*15b*', '*15c*', '*15d*'], ['CTRL', 'th-FOR', 'q-th-FOR', 'q-incr']):
    proc_dict[nm] = {}
    try:
        for proc, v in zip(
                ['cond', 'ice_nucl', 'homg_fr_cl', 'ice_dep', 'snow_dep', 'ice_sed', 'snow_sed', 'graupel_sed',
                 'ice_subm', 'snow_subm',
                 'graupel_subm', 'ice_melt', 'snow_melt', 'graupel_melt', 'rn_autoconv', 'rn_accr', 'snow_accr_ice',
                 'sn_accr_rn', 'gr_accr_cl', 'gr_accr_sn', 'ice_acc_w', 'ice_to_snow'],
                ['pcond_mean', 'pinuc_mean', 'phomc_mean', 'pidep_mean', 'psdep_mean', 'psedi_mean', 'pseds_mean',
                 'psedg_mean',
                 'pisub_mean', 'pssub_mean', 'pgsub_mean', 'pimlt_mean', 'psmlt_mean', 'pgmlt_mean', 'praut_mean',
                 'pracw_mean', 'psaci_mean', 'psacr_mean',
                 'pgacw_mean', 'pgacs_mean', 'piacw_mean', 'psaut_mean']):
            f1 = iris.load_cube(filepath + fn + '?d_3600.nc', v)
            f2 = iris.load_cube(filepath + fn + '?d_7200.nc', v)
            f3 = iris.load_cube(filepath + fn + '?d_10800.nc', v)
            proc_dict[nm][proc] = np.concatenate((f1.data.max(axis=1), f2.data.max(axis=1), f3.data.max(axis=1)), axis=0)
        if fn == '*15a*':
            time_srs_lowres = np.concatenate((f1.coord('time_series_100_100').points,f2.coord('time_series_100_100').points,f3.coord('time_series_100_100').points), axis=0)
        elif fn == '*15b*':
            time_srs_hires = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points,f3.coord('time_series_10_30').points), axis=0)
        elif fn == '*15c*':
            time_srs_q = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points,f3.coord('time_series_10_30').points), axis=0)
        elif fn == '*15d*':
            time_srs_incr = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points, f3.coord('time_series_10_30').points), axis=0)
        elif fn == '*15e*':
            time_srs_nF = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points, f3.coord('time_series_10_30').points), axis=0)
    except:
        print('Uh oh, sutn ain\'t right.')

fig, ax = plt.subplots(2,2, figsize = (9, 5))
ax = ax.flatten()
for axs, nm, t in zip(ax, ['CTRL', 'th-FOR', 'q-th-FOR', 'q-incr'], [time_srs_hires[1:], time_srs_hires, time_srs_q, time_srs_incr]):
    for proc, c in zip(proc_dict[nm].keys(), ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal',
                  'brown', 'darkgrey', 'indigo', 'darkred', 'cyan', 'bisque', 'lightsteelblue', 'turquoise', 'forestgreen']):
        axs.set_title(nm)
        if proc_dict[nm][proc].max() > 1e-15:
            axs.plot(t[1:], proc_dict[nm][proc][1:], color = c, label = proc)

plt.legend(bbox_to_anchor=(1.65, 1.8))
plt.subplots_adjust(wspace = 0.2, hspace = 0.25, left = 0.1, right = 0.8)
plt.show()


qi_forcing = iris.load_cube(filepath + '*15e*3600.nc', 'tend_qi_forcing_3d_local')
qi_tvd = iris.load_cube(filepath + '*15e*3600.nc', 'tend_qi_tvdadvection_3d_local')
qi_diff = iris.load_cube(filepath + '*15e*3600.nc', 'tend_qi_diffusion_3d_local')
qi_tot = iris.load_cube(filepath + '*15e*3600.nc', 'tend_qi_total_3d_local')

qi_mp = iris.load_cube(filepath + '*15e*3600.nc', 'dqi_mphys_mean')
qi_diff = iris.load_cube(filepath + '*15e*3600.nc', 'tend_qi_diff_mean')
qi_tot = iris.load_cube(filepath + '*15e*3600.nc', 'tend_qi_total_mean')
qi_tvd = iris.load_cube(filepath + '*15e*3600.nc', 'tend_qi_tvda_mean')

fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
axs = axs.flatten()
cbax = fig.add_axes([0.9,0.25, 0.03, 0.5])
for ax, tend, title in zip(axs,[qi_forcing, qi_tvd, qi_diff, qi_mp],['forcing', 'advection', 'diffusion', 'mphys']):
    if title == 'forcing':
        c = ax.contourf(qi_forcing.coord('time_series_300_300').points, qi_mp.coord('zn').points[60:95],
                        tend[:, :,:, 60:95].data.mean(axis=(1,2)).transpose(), cmap='bwr', levels=np.linspace(-1.5e-9, 1.5e-9, 11),
                        extend='both')
    else:
        c=ax.contourf(qi_mp.coord('time_series_10_30').points, qi_mp.coord('zn').points[60:95], tend[:,60:95].data.transpose(), cmap='bwr', levels = np.linspace(-1.5e-9,1.5e-9, 11) , extend='both')
    ax.set_title(title)

for ax in [axs[0], axs[2]]:
    ax.set_ylabel('altitude (m)')

for ax in [axs[2], axs[3]]:
    ax.set_xlabel('time (s)')

plt.colorbar(c, cax = cbax)
plt.subplots_adjust(right = 0.83)
plt.show()