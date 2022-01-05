import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

filepath = '/storage/silver/acacia/ke923690/acacia/new_runs/'

iwp_dict = {}
for fn, nm in zip(['file_str1', 'file_str2'...], ['CTRL', 'sim1'...]):
    f1 = iris.load_cube(filepath + fn + '3600.nc', 'IWP_mean')
    f2 = iris.load_cube(filepath + fn + '7200.nc', 'IWP_mean')
    f3 = iris.load_cube(filepath + fn + '10800.nc', 'IWP_mean')
    try:
        f4 = iris.load_cube(filepath + fn + '14400.nc', 'IWP_mean')
        iwp_dict[nm] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
    except:
        iwp_dict[nm] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
    try:
        time_srs_hires = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points,f3.coord('time_series_10_30').points,f4.coord('time_series_10_30').points), axis = 0) #, iwp4.coord('time_series_100_100').points
    except:
        time_srs_hires = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points,
                                         f3.coord('time_series_10_30').points), axis=0)  # , iwp4.coord('time_series_100_100').points

# Load files

fig, ax = plt.subplots(figsize=(8, 5))
for nm, col in zip(['CTRL', 'sim1' ...], ['w', '#FF6347', '#FF8C00', '#F4A460',  '#DC143C', '#FF69B4', '#FFC0CB', '#006400', '#20B2AA', '#9ACD32',  '#00FF00', '#00008B', '#4169E1', '#1E90FF', '#87CEEB', '#222222']):
    try:
        ax.plot(time_srs_hires, iwp_dict[nm], lw=2, label = nm)
    except:
        ax.plot(np.linspace(0, 10800, 280), iwp_dict[nm][:280], color=col, lw=2, label=nm)

ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.set_xlim(0, 10800)
ax.set_ylim(0, 0.01)
#ax.set_ylim(0,0.005)
ax.set_xlabel('Time (s)', color='dimgrey', fontsize = 18)
ax.set_ylabel('IWP\n(kg/m$^{2}$)', rotation =0, labelpad=50, color='dimgrey', fontsize = 18)
plt.subplots_adjust(left = 0.25, bottom = 0.15, right=0.97)
plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='best',ncol=2)
ax.tick_params(axis='both', which='both', labelsize=14, labelcolor='dimgrey', pad=10,  color='dimgrey', length=5, width = 2.5)
plt.savefig(filepath + '../figures/IWP_comparison_pert_runs.png')
plt.show()