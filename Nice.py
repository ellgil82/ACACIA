import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
import imageio

filepath = '/storage/silver/acacia/ke923690/acacia/'

Nice_dict = {}
for fn in ['*14g_LOTS_INPx2*', '*14g_unchanged*', '*14h_INPx2*', '*CTRL*']:
    f1 = iris.load_cube(filepath + fn + '3600.nc', 'ice_nc_mean')
    f2 = iris.load_cube(filepath + fn + '7200.nc', 'ice_nc_mean')
    f3 = iris.load_cube(filepath + fn + '10800.nc', 'ice_nc_mean')
    Nice_dict[fn] = np.concatenate((f1.data, f2.data, f3.data), axis=0)

model_hts = f1.coord('zn').points + 50 # add 50 to offset centring in middle of level
z = model_hts
time_srs = np.concatenate((f1.coord('time_series_100_100').points, f2.coord('time_series_100_100').points,f3.coord('time_series_100_100').points), axis = 0)

def plot_Nice():
    fig, ax = plt.subplots(figsize=(5,6))
    for fn in Nice_dict.keys():
       color_list = ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal', 'brown', 'darkgrey', 'indigo', ]
       ax.plot(Nice_dict[fn][-7, 55:90], model_hts[55:90]/1000, label = fn)#, color = color_list[t])
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
    plt.savefig(filepath + 'figures/Nice_test_14_comparison.png')
    plt.show()

plot_Nice()

for filename, fig_str in zip(['Yang_test14h_INPx2*', 'Yang_test14g_LOTS_INPx2*', 'Yang_test14g_unchanged*', ],
                             ['INPx2', 'LOTS_INPx2', 'LOTS', ]):

prop_dict = {}
for name, v in zip(['ice_mmr', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'ice_nc_mean', 'snow_nc_mean',
                    'graupel_nc_mean'],
                   ['ice_mmr_mean', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean', 'ice_nc_mean',
                    'snow_nc_mean', 'graupel_nc_mean']):
    f1 = iris.load_cube(filepath + filename + '?d_3600.nc', v)
    f2 = iris.load_cube(filepath + filename + '?d_7200.nc', v)
    f3 = iris.load_cube(filepath + filename + '?d_10800.nc', v)
    prop_dict[name] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
try:
    time_srs = prop_dict['ice_mmr'].coord('time_series_300_300').points
except:
    time_srs = np.concatenate((f1.coord('time_series_100_100').points, f2.coord('time_series_100_100').points,
                               f3.coord('time_series_100_100').points), axis=0)


f1 = iris.load_cube(filepath + 'Yang_test14h_INPx2_?d_3600.nc', 'q_ice_number') #1st
f2 = iris.load_cube(filepath + 'Yang_test14h_INPx2_?d_7200.nc',  'q_ice_number')
f3 = iris.load_cube(filepath + 'Yang_test14h_INPx2_?d_10800.nc',  'q_ice_number')
f_14h_inp = np.concatenate((f1.data, f2.data, f3.data), axis=0) #, iwp4.data

f1 = iris.load_cube(filepath + 'Yang_*CTRL*_?d_3600.nc', 'q_ice_number') #1st
f2 = iris.load_cube(filepath + 'Yang_*CTRL*_?d_7200.nc',  'q_ice_number')
f3 = iris.load_cube(filepath + 'Yang_*CTRL*_?d_10800.nc',  'q_ice_number')
f_14b_ctrl = np.concatenate((f1.data, f2.data, f3.data), axis=0) #, iwp4.data
time_srs300 = np.concatenate((f1.coord('time_series_300_300').points, f2.coord('time_series_300_300').points,
                               f3.coord('time_series_300_300').points), axis=0)
time_srs100 = np.concatenate((f1.coord('time_series_100_100').points, f2.coord('time_series_100_100').points,
                               f3.coord('time_series_100_100').points), axis=0)

f1 = iris.load_cube(filepath + 'Yang_test14g_LOTS_INPx2_?d_3600.nc', 'q_ice_number') #1st
f2 = iris.load_cube(filepath + 'Yang_test14g_LOTS_INPx2_?d_7200.nc',  'q_ice_number')
f3 = iris.load_cube(filepath + 'Yang_test14g_LOTS_INPx2_?d_10800.nc',  'q_ice_number')
f_14g_lots_inp = np.concatenate((f1.data, f2.data, f3.data), axis=0) #, iwp4.data

plt.plot(time_srs, f_14b_ctrl.data.mean(axis=(1,2)).max(axis=1), label = 'ctrl')
plt.plot(time_srs, f_14h_inp.mean(axis=(1,2)).max(axis=1), label = 'INPx10')
plt.plot(time_srs[4:], f_14g_lots_inp.mean(axis=(1,2)).max(axis=1), label = 'LOTS_INPx10')
#plt.plot(f_14b_ctrl.data.max(axis=1))
#plt.plot(f_14h_inp.data.max(axis=1))
#plt.plot(f_14g_lots_inp.data.max(axis=1))
plt.legend()
plt.show()