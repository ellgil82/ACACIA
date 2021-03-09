import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import imageio

filepath = '/storage/silver/acacia/ke923690/acacia/'

# Load files
#filename = input("Enter name of diagnostics file\n")
#variable = input("Enter variable to plot\n")
#timestep = input("Which timesteps do you need? (0-indexed)\n")

filename = 'Yang_casim_proc_pert_GCSS_28800.nc'
IWP_n = iris.load_cube(filepath + 'acacia_diagnostic_3d_3600_negative_subs.nc', 'IWP_mean')
IWP_p = iris.load_cube(filepath + 'acacia_diagnostic_0d_positive_subs_3600.nc', 'IWP_mean')

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(IWP_n.data, label = 'subs=negative')#, np.linspace(0,14400, 141))#, color = color_list[t])
ax.plot(IWP_p.data, label = 'subs=positive')#, np.linspace(0,14400, 141))#, color = color_list[t])
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax.set_xlim(0, 248)
# #xt = [0,50,100,150, 200, 250]
#ax.set_xticks(xt)
l = (np.array(xt) * 100).tolist()
ax.set_xticklabels(l)
ax.set_ylim(0, 2.5e-2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('IWP (kg/m$^{2}$)', rotation =0, labelpad=50)
plt.subplots_adjust(left = 0.2)
#plt.savefig(filepath + 'figures/IWP_perturbed_dz100m.png')
plt.show()