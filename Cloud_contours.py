import numpy as np
import iris
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/users/ke923690/scripts/Tools/')
from divg_temp_colourmap import shiftedColorMap

filepath = '/storage/silver/acacia/ke923690/acacia/new_runs/'

contour_dict = {}
#for fn, nm in zip(['*17_CTRL*', '*22a*','*22b*', '*22c*','*22d*', '*22e*','*22f*', '*22g*','*22h*', '*22i*','*22j*',],['CTRL', 'slow_uv', 'slow_uv_ICNCx2_8-9','slow_uv_ICNCx2_9-10','slow_uv_no_qF','slow_uv_ICNCx1.5_8-9','slow_uv_ICNCx1.5_9-10','slow_uv_ICNCx1.25_8-9','slow_uv_ICNCx1.25_9-10','slow_uv_ICNCx1.1_8-9','slow_uv_ICNCx1.1_9-10',]):#, '*18a*', '*18b*', '*14p*8-9*','*14o*9-10*', ], [ 'CTRL', 'ICNCx2_8-9_2000', 'ICNCx2_9-10_2000', 'ICEx2_8-9_2000', 'ICEx2_9-10_2000',]):
#for fn, nm in zip(['*17_CTRL*', '*18a*', '*18b*',   '*20i*','*20j*', '*20k*','*20l*'], ['CTRL', 'ICNCx2_8-9_2000', 'ICNCx2_9-10_2000',  'ICNCx1.25_8-9_2000', 'ICNCx1.25_9-10_2000', 'ICNCx1.5_8-9_2000', 'ICNCx1.5_9-10_2000']):#, '*20a*', '*20b*', '*20c*', '*20d*', '*20e*', '*20f*', '*20g*', '*20h*',]['95', '90', '85', '50', 'uv_q100', 'uv_q75', 'uv_q125', 'uv_q110']):
for fn,nm in zip(['*22a*','*22k*', '*22l*', '*22m*'],['CTRL','ICNCx2_full', 'ICEx2_full', 'MASSx2_full' ]):
    contour_dict[nm] = {}
    try:
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'clbas')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'clbas')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'clbas')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'clbas')
            contour_dict[nm]['clbas'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            contour_dict[nm]['clbas'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contour_dict[nm]['clbas'][contour_dict[nm]['clbas'] == 0.] = np.nan
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'cltop')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'cltop')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'cltop')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'cltop')
            contour_dict[nm]['cltop'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            contour_dict[nm]['cltop'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contour_dict[nm]['cltop'][contour_dict[nm]['cltop'] == 0.] = np.nan
        contour_dict[nm]['time_srs_lowres'] = np.concatenate((f1.coord('time_series_300_300').points, f2.coord('time_series_300_300').points,
                        f3.coord('time_series_300_300').points, f4.coord('time_series_300_300').points), axis=0)
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'ice_mmr_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'ice_mmr_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'ice_mmr_mean')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'ice_mmr_mean')
            contour_dict[nm]['M'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            contour_dict[nm]['M'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contour_dict[nm]['M'][contour_dict[nm]['M'] == 0.] = np.nan
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'ice_nc_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'ice_nc_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'ice_nc_mean')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'ice_nc_mean')
            contour_dict[nm]['N'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            contour_dict[nm]['N'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contour_dict[nm]['N'][contour_dict[nm]['N'] == 0.] = np.nan
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'temperature')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'temperature')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'temperature')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'temperature')
            contour_dict[nm]['T'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            contour_dict[nm]['T'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contour_dict[nm]['T'] = contour_dict[nm]['T'].mean(axis=(1,2))
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'pidep_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'pidep_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'pidep_mean')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'pidep_mean')
            contour_dict[nm]['dep'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            contour_dict[nm]['dep'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contour_dict[nm]['dep'][contour_dict[nm]['dep'] == 0.] = np.nan
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'psedi_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'psedi_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'psedi_mean')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'psedi_mean')
            contour_dict[nm]['sed'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            contour_dict[nm]['sed'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contour_dict[nm]['sed'][contour_dict[nm]['sed'] == 0.] = np.nan
    except:
        print('Oops - forgot my keys')
    try:
        contour_dict[nm]['time_srs'] = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points, f3.coord('time_series_10_30').points, f4.coord('time_series_10_30').points), axis=0)
    except iris.exceptions.CoordinateNotFoundError:
        contour_dict[nm]['time_srs'] = np.concatenate((f1.coord('time_series_100_100').points, f2.coord('time_series_100_100').points,
                        f3.coord('time_series_100_100').points, f4.coord('time_series_100_100').points), axis=0)

alt = f1.coord('zn').points

def contourplot(var):
    fig, ax = plt.subplots(4,2, figsize = (6,13), sharex=True, sharey=True)
    #fig, ax = plt.subplots(3, 2, figsize=(6, 11), sharex=True, sharey=True)
    ax = ax.flatten()
    cbax = fig.add_axes([0.3,0.08, 0.5, 0.02])
    ## Define colour levels
    if var == 'M':
        pos = np.geomspace(1e-10, 5e-5, 20)
        neg = pos*-1
        vmin = -5e-5
        vmax =  5e-5
        linthresh = 1e-8
        linscale = 5e-8
    elif var == 'N':
        pos = np.geomspace(1, 1e7, 20)
        neg = pos * -1
        vmin = -1e7
        vmax =  1e7
        linthresh = 1e2
        linscale = 0.0001
    levels = np.concatenate((np.flip(neg), np.zeros(1,), pos))
    #bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-5e-6, max_val = 5e-5, name='bwr_zero', var=((contour_dict['CTRL']['M']-contour_dict['CTRL']['M'])), start=0.15,stop=0.85)
    for axs, sim, n in zip(ax, [ 'slow_uv_ICNCx2_8-9', 'slow_uv_ICNCx2_9-10','slow_uv_ICNCx1.5_8-9', 'slow_uv_ICNCx1.5_9-10','slow_uv_ICNCx1.25_8-9', 'slow_uv_ICNCx1.25_9-10','slow_uv_ICNCx1.1_8-9','slow_uv_ICNCx1.1_9-10',], [1,2,3,4,5,6,7,8]):
    #for axs, sim, n in zip(ax, ['ICEx0.5_8-9_2000', 'ICEx0.5_9-10_2000','ICEx1.1_8-9_2000', 'ICEx1.1_9-10_2000', 'ICEx2_9-10_2000', 'ICEx2_8-9_2000','INPx0.5_8-9_2000', 'INPx0.5_9-10_2000'], [1, 2, 3, 4, 5, 6, 7, 8]):
        try:
            c = axs.contourf(contour_dict[sim]['time_srs'], alt[60:95],
                             (contour_dict[sim][var][:, 60:95].transpose() -
                             contour_dict['slow_uv'][var][:, 60:95].transpose()),
                             norm=matplotlib.colors.SymLogNorm(vmin = vmin, vmax = vmax, linthresh=linthresh, linscale=linscale),  cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels) #
            #axs.contour(contour_dict[sim]['time_srs'][:274], alt[60:95], contour_dict['CTRL'][var][:274, 60:95].transpose()>0.,
            #                levels=[0], colors = 'darkgrey')
            axs.set_facecolor('#BEDBE4')
            axs.plot(contour_dict[sim]['time_srs_lowres'],np.nanmean(contour_dict[sim]['clbas'], axis=(1, 2)), color='darkgrey')
            axs.plot(contour_dict[sim]['time_srs_lowres'],np.nanmean(contour_dict[sim]['cltop'], axis=(1, 2)), color='darkgrey')
            axs.set_xlim(0, 10800)
            axs.set_yticks([8000, 9000, 10000, 11000])
            axs.set_title(sim + '\n - CTRL', fontweight='bold', loc='center')
        except ValueError:
            c = axs.contourf(contour_dict[sim]['time_srs'], alt[60:95],
                             (contour_dict[sim][var][:, 60:95].transpose() -
                              contour_dict['CTRL_lowres'][var][:, 60:95].transpose()),
                             norm=matplotlib.colors.SymLogNorm(vmin = vmin, vmax = vmax,linthresh=linthresh, linscale=linscale), cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels)
            axs.set_facecolor('#BEDBE4')
            axs.plot( np.nanmean(contour_dict[sim]['clbas'], axis=(1, 2)), color='darkgrey')
            axs.plot( np.nanmean(contour_dict[sim]['cltop'], axis=(1, 2)),color='darkgrey')
            axs.set_title(sim + '\n - CTRL', fontweight= 'bold',loc='center')
            axs.set_xlim(0, 10800)
            axs.set_yticks([8000, 9000, 10000, 11000])
        if (n % 2) != 0: # odd number
            axs.set_ylabel('altitude (m)')
        if n > 6:
            axs.set_xlabel('time (s)')
        axs.tick_params(direction='in')
        if var == 'M':
            cb = plt.colorbar(c, cax=cbax, orientation='horizontal', extend='both', ticks = [-4.9e-5, -5e-6, -5e-7, -5e-8, -1e-8, 0, 1e-8, 5e-8,  5e-7, 5e-6, 4.9e-5])
            cb.set_ticklabels([r'$-5 \times 10^{-5}$', '', r'$-5 \times 10^{-7}$', '',r'-1 $\times$ 10$^{-8}$', '0', r'$1 \times 10^{-8}$', '', r'$5 \times 10^{-7}$','', r'$5 \times 10^{-5}$'])
            cb.ax.set_xlabel('Ice MMR (g kg$^{-1}$)')
        elif var == 'N':
            cb = plt.colorbar(c, cax=cbax, orientation='horizontal', extend='both',
                              ticks=[-1e7,  -1e5,  -1e3, -1e2, 0, 1e2, 1e3,1e5, 1e7])
            cb.ax.set_xlabel('ICNC (kg$^{-1}$)')
    plt.subplots_adjust(right = 0.95, top = 0.95, hspace = 0.23, wspace= 0.08, bottom = 0.15)
    plt.savefig(filepath + '../figures/Hovmoller_'+ var + '_slow_uv_runs.png')
    plt.show()

contourplot('M')
contourplot('N')


def full_prof_contours(var):
    fig, ax = plt.subplots(2,2, figsize = (6,9), sharex=True, sharey=True)
    #fig, ax = plt.subplots(3, 2, figsize=(6, 11), sharex=True, sharey=True)
    ax = ax.flatten()
    cbax = fig.add_axes([0.3,0.08, 0.5, 0.02])
    ## Define colour levels
    if var == 'M':
        pos = np.geomspace(1e-10, 1e-2, 20)
        neg = pos*-1
        vmin = -1e-2
        vmax = 1e-2
        linthresh = 1e-10
        linscale = 5e-10
    elif var == 'N':
        pos = np.geomspace(1, 1e9, 20)
        neg = pos * -1
        vmin = -1e9
        vmax =  1e9
        linthresh = 1e2
        linscale = 0.0001
    levels = np.concatenate((np.flip(neg), np.zeros(1,), pos))
    c = ax[0].contourf(contour_dict['CTRL']['time_srs'], alt[60:95], contour_dict['CTRL'][var][:, 60:95].transpose(),
                     norm=matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh, linscale=linscale),
                     cmap='bwr', vmin=vmin, vmax=vmax, levels=levels)  #
    # axs.contour(contour_dict['CTRL']['time_srs'][:274], alt[60:95], contour_dict['CTRL'][var][:274, 60:95].transpose()>0.,
    #                levels=[0], colors = 'darkgrey')
    ax[0].set_facecolor('#BEDBE4')
    ax[0].plot(contour_dict['CTRL']['time_srs_lowres'], np.nanmean(contour_dict['CTRL']['clbas'], axis=(1, 2)),
             color='darkgrey')
    ax[0].plot(contour_dict['CTRL']['time_srs_lowres'], np.nanmean(contour_dict['CTRL']['cltop'], axis=(1, 2)),
             color='darkgrey')
    ax[0].set_xlim(0, 14400)
    ax[0].set_yticks([8000, 9000, 10000, 11000])
    ax[0].set_title('CTRL', fontweight='bold', loc='center')
    #bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-5e-6, max_val = 5e-5, name='bwr_zero', var=((contour_dict['CTRL']['M']-contour_dict['CTRL']['M'])), start=0.15,stop=0.85)
    for axs, sim, n in zip(ax[1:], ['ICNCx2_full', 'MASSx2_full', 'ICEx2_full'], [1,2,3,4,5,6,7,8]):
    #for axs, sim, n in zip(ax, ['ICEx0.5_8-9_2000', 'ICEx0.5_9-10_2000','ICEx1.1_8-9_2000', 'ICEx1.1_9-10_2000', 'ICEx2_9-10_2000', 'ICEx2_8-9_2000','INPx0.5_8-9_2000', 'INPx0.5_9-10_2000'], [1, 2, 3, 4, 5, 6, 7, 8]):
        try:
            c = axs.contourf(contour_dict[sim]['time_srs'], alt[60:95],
                             (contour_dict[sim][var][:, 60:95].transpose() -
                             contour_dict['CTRL'][var][:, 60:95].transpose()),
                             norm=matplotlib.colors.SymLogNorm(vmin = vmin, vmax = vmax, linthresh=linthresh, linscale=linscale),  cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels) #
            #axs.contour(contour_dict[sim]['time_srs'][:274], alt[60:95], contour_dict['CTRL'][var][:274, 60:95].transpose()>0.,
            #                levels=[0], colors = 'darkgrey')
            axs.set_facecolor('#BEDBE4')
            axs.plot(contour_dict[sim]['time_srs_lowres'],np.nanmean(contour_dict[sim]['clbas'], axis=(1, 2)), color='darkgrey')
            axs.plot(contour_dict[sim]['time_srs_lowres'],np.nanmean(contour_dict[sim]['cltop'], axis=(1, 2)), color='darkgrey')
            axs.set_xlim(0, 14400)
            axs.set_yticks([8000, 9000, 10000, 11000])
            axs.set_title(sim + '\n - CTRL', fontweight='bold', loc='center')
        except ValueError:
            c = axs.contourf(contour_dict[sim]['time_srs'], alt[60:95],
                             (contour_dict[sim][var][:, 60:95].transpose() -
                              contour_dict['CTRL_lowres'][var][:, 60:95].transpose()),
                             norm=matplotlib.colors.SymLogNorm(vmin = vmin, vmax = vmax,linthresh=linthresh, linscale=linscale), cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels)
            axs.set_facecolor('#BEDBE4')
            axs.plot( np.nanmean(contour_dict[sim]['clbas'], axis=(1, 2)), color='darkgrey')
            axs.plot( np.nanmean(contour_dict[sim]['cltop'], axis=(1, 2)),color='darkgrey')
            axs.set_title(sim + '\n - CTRL', fontweight= 'bold',loc='center')
            axs.set_xlim(0, 10800)
            axs.set_yticks([8000, 9000, 10000, 11000])
        if (n % 2) == 0: # odd number
            axs.set_ylabel('altitude (m)')
        if n > 6:
            axs.set_xlabel('time (s)')
        axs.tick_params(direction='in')
        if var == 'M':
            cb = plt.colorbar(c, cax=cbax, orientation='horizontal', extend='both', ticks = [-4.9e-5, -5e-6, -5e-7, -5e-8, -1e-8, 0, 1e-8, 5e-8,  5e-7, 5e-6, 4.9e-5])
            cb.set_ticklabels([r'$-5 \times 10^{-5}$', '', r'$-5 \times 10^{-7}$', '',r'-1 $\times$ 10$^{-8}$', '0', r'$1 \times 10^{-8}$', '', r'$5 \times 10^{-7}$','', r'$5 \times 10^{-5}$'])
            cb.ax.set_xlabel('Ice MMR (kg kg$^{-1}$)')
        elif var == 'N':
            cb = plt.colorbar(c, cax=cbax, orientation='horizontal', extend='both',
                              ticks=[-1e7,  -1e5,  -1e3, -1e2, 0, 1e2, 1e3,1e5, 1e7])
            cb.ax.set_xlabel('ICNC (kg$^{-1}$)')
    plt.subplots_adjust(right = 0.95, top = 0.95, hspace = 0.23, wspace= 0.08, bottom = 0.15)
    plt.savefig(filepath + '../figures/Hovmoller_'+ var + '_full_profile.png')
    plt.show()

full_prof_contours('M')
full_prof_contours('N')

for sim in contour_dict.keys():
    try:
        print(np.nanmin(contour_dict[sim]['N'][:, 60:95]- contour_dict['CTRL_lowres']['N'][:, 60:95]))
        print(np.nanmax(contour_dict[sim]['N'][:, 60:95] - contour_dict['CTRL_lowres']['N'][:, 60:95]))
    except:
        print(np.nanmin(contour_dict[sim]['N'][:282, 60:95] - contour_dict['CTRL']['N'][:282, 60:95]))
        print(np.nanmax(contour_dict[sim]['N'][:282, 60:95] - contour_dict['CTRL']['N'][:282, 60:95]))

plt.contourf((contour_dict['ICNCx2_9-10_2000']['N'][:281, 60:95]-contour_dict['ICEx2_9-10_2000']['N'][:281, 60:95]).transpose())
plt.colorbar()
plt.show()


contrail_dict = {}
for fn, nm in zip(['*17_CTRL*', '*20e*'], ['CTRL', 'uv_f']):
    contrail_dict[nm] = {}
    try:
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'q_ice_mass')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'q_ice_mass')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'q_ice_mass')
        contrail_dict[nm]['M'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contrail_dict[nm]['M'][contrail_dict[nm]['M'] == 0.] = np.nan
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'q_ice_number')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'q_ice_number')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'q_ice_number')
        contrail_dict[nm]['N'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contrail_dict[nm]['N'][contrail_dict[nm]['N'] == 0.] = np.nan
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'q_coarse_dust_number')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'q_coarse_dust_number')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'q_coarse_dust_number')
        contrail_dict[nm]['INP'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contrail_dict[nm]['INP'][contrail_dict[nm]['INP'] == 0.] = np.nan
        try:
            contrail_dict[nm]['time_srs_lowres'] = np.concatenate(
                (f1.coord('time_series_300_300').points, f2.coord('time_series_300_300').points,
                 f3.coord('time_series_300_300').points), axis=0)
        except:
            print('No low res times')
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'iwp')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'iwp')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'iwp')
        contrail_dict[nm]['iwp'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contrail_dict[nm]['iwp'][contrail_dict[nm]['iwp'] == 0.] = np.nan
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'dqi_mphys_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'dqi_mphys_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'dqi_mphys_mean')
        contrail_dict[nm]['dqi_mp'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        contrail_dict[nm]['dqi_mp'][contrail_dict[nm]['dqi_mp'] == 0.] = np.nan
    except:
        print('Oops - forgot my keys')
    try:
        contrail_dict[nm]['time_srs'] = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points,
                                   f3.coord('time_series_10_30').points), axis=0)
    except iris.exceptions.CoordinateNotFoundError:
        contrail_dict[nm]['time_srs'] = np.concatenate((f1.coord('time_series_100_100').points, f2.coord('time_series_100_100').points,
                        f3.coord('time_series_100_100').points), axis=0)


for k in contrail_dict.keys():
    plt.plot(contrail_dict[k]['q_ice_number'].mean(axis=(1,2,3)), label = k)

plt.legend()
plt.show()

t = 6
fig, ax = plt.subplots(1,3)
ax = ax.flatten()
for k, axs in zip(contrail_dict.keys(), ax):
    c = axs.contourf(np.nanmean((contrail_dict[k]['N'][t]-contrail_dict['CTRL']['N'][t]), axis=2))#.transpose(), vmin = 0, vmax=6e-5)

plt.colorbar(c)
plt.show()

def plot_proc_contour(var):
    fig, ax = plt.subplots(3, 2, figsize=(6, 11), sharex=True, sharey=True) #2,2 (6,8)
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
    for k, axs in zip(['ICNCx1.25_8-9_2000', 'ICNCx1.25_9-10_2000','ICNCx1.5_8-9_2000', 'ICNCx1.5_9-10_2000','ICNCx2_8-9_2000', 'ICNCx2_9-10_2000', ], ax):
        c = axs.contourf(contour_dict[k]['time_srs'][2:281], alt[70:90], (contour_dict[k]['sed'][2:281, 70:90] - contour_dict['CTRL']['sed'][2:281, 70:90]).transpose(),
                         norm=matplotlib.colors.SymLogNorm(vmin = vmin, vmax = vmax,linthresh=linthresh, linscale=linscale), cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels)
        axs.set_title(k + ' - CTRL')
        axs.set_facecolor('#BEDBE4')
        axs.plot(contour_dict[k]['time_srs_lowres'],np.nanmean(contour_dict[k]['clbas'], axis=(1, 2)), color='darkgrey')
        axs.plot(contour_dict[k]['time_srs_lowres'],np.nanmean(contour_dict[k]['cltop'], axis=(1, 2)), color='darkgrey')
        axs.set_xlim(0,10800)
    cb = plt.colorbar(c, cax=cbax, orientation='horizontal', extend='both', ticks = [-1e-5, 0, 1e-5])
    plt.subplots_adjust(top=0.95, bottom = 0.2)
    plt.savefig(filepath + '../figures/Hovmoller_' + var + '_ICNC_runs_2000s.png')
    plt.show()

plot_proc_contour('dep')


fig, ax =plt.subplots(2,2)
ax = ax.flatten()
for axs in ax:
    axs.set_facecolor('#BEDBE4')
    axs.plot(np.nanmean(contour_dict[sim]['clbas'], axis=(1, 2)), color='darkgrey')
    axs.plot(np.nanmean(contour_dict[sim]['cltop'], axis=(1, 2)), color='darkgrey')
    axs.set_xlim(0, 10800)

c = ax[0].contourf(contour_dict['CTRL']['time_srs'][:274], alt[60:95], contour_dict['CTRL']['M'][:274, 60:95].transpose())
ax[0].set_title('M')
ax[0].colorbar(c)
c = ax[1].contourf(contour_dict['CTRL']['time_srs'][:274], alt[60:95], contour_dict['CTRL']['N'][:274, 60:95].transpose())
ax[0].set_title('N')
ax[1].colorbar(c)
c = ax[2].contourf(contour_dict['CTRL']['time_srs'][:274], alt[60:95], contour_dict['CTRL']['T'][:274, 60:95].transpose())
ax[2].set_title('T')
ax[2].colorbar(c)
#ax[3].contourf(contour_dict['CTRL']['time_srs'][:274], alt[60:95], contour_dict['CTRL']['M'][:274, 60:95].transpose())
ax[3].set_title('re')

fig, ax = plt.subplots(1,2)
ax = ax.flatten()
for k,i in zip(contrail_dict.keys(), [0,1]):
    c = ax[i].contourf(contrail_dict[k]['time_srs'], alt[60:95], contrail_dict[k]['dqi_mp'][:,60:95].transpose(), vmin = -1e-7, vmax=1e-7)

plt.colorbar(c)
plt.show()