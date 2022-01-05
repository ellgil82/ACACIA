import numpy as np
import iris
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/users/ke923690/scripts/Tools/')
from divg_temp_colourmap import shiftedColorMap
import imageio

filepath = '/storage/silver/acacia/ke923690/acacia/new_runs/'

map_dict = {}
for fn, nm in zip(['*17_CTRL*', '*20j*', '*20l*'], ['CTRL', 'fast_uv', 'slow_uv']):
    map_dict[nm] = {}
    try:
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'q_ice_mass')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'q_ice_mass')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'q_ice_mass')
        map_dict[nm]['M'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        map_dict[nm]['M'][map_dict[nm]['M'] == 0.] = np.nan
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'q_ice_number')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'q_ice_number')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'q_ice_number')
        map_dict[nm]['N'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        map_dict[nm]['N'][map_dict[nm]['N'] == 0.] = np.nan
        map_dict[nm]['time_srs_lowres'] = np.concatenate((f1.coord('time_series_300_300').points, f2.coord('time_series_300_300').points,
                        f3.coord('time_series_300_300').points), axis=0)
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'temperature')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'temperature')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'temperature')
        map_dict[nm]['T'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        map_dict[nm]['T'] = map_dict[nm]['T'].mean(axis=(1,2))
    except:
        print('Oops - forgot my keys')

alt = f1.coord('zn').points

for t in range(28):
    fig, ax = plt.subplots(1,3, figsize=(10, 4), sharex=True, sharey=True)
    ax = ax.flatten()
    CbAx1 = fig.add_axes([0.13, 0.15, 0.22, 0.02])
    CbAx2 = fig.add_axes([0.5, 0.15, 0.3, 0.02])
    for sim, n in zip(map_dict.keys(), range(3)):
        if sim == 'CTRL':
            c = ax[n].contourf(np.nanmean(map_dict[sim]['M'][t, :, :, 60:95], axis=2), vmin=0, vmax = 1.5e-5, levels = np.linspace(0, 1.5e-5, 21))
            ax[n].set_title(sim)
            plt.colorbar(c, cax=CbAx1, orientation='horizontal', ticks = [0, 5e-6, 1e-5, 1.5e-5])
        else:
            c = ax[n].contourf(np.nanmean(map_dict[sim]['M'][t, :, :, 60:95], axis=2) -
                               np.nanmean(map_dict['CTRL']['M'][t, :, :, 60:95], axis=2), vmin=-3.5e-6, vmax=3.5e-6,
                               levels=np.linspace(-3e-6, 3e-6, 21), cmap= 'bwr')
            ax[n].set_title(sim + ' - CTRL')
            plt.colorbar(c, cax=CbAx2, orientation='horizontal', ticks = [-3e-6, -1.5e-6, 0, 1.5e-6, 3e-6])
        ax[n].axis('off')
    plt.subplots_adjust(bottom = 0.25, top = 0.9)
    plt.savefig(filepath + '../figures/IWC_map_CTRL_v_uv_ts' + str(t) + '.png')
    #plt.show()
    plt.close('all')

filenames = []
# create file name and append it to a list
for i in range(27):
    fn = filepath + f'../figures/IWC_map_CTRL_v_uv_ts{i}.png'
    filenames.append(fn)

with imageio.get_writer(filepath + '../figures/IWC_map_CTRL_v_uv.gif', mode='I') as writer:
    for f in filenames:
        image = imageio.imread(f)
        writer.append_data(image)