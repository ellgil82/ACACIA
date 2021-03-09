import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D

filepath = '/storage/silver/acacia/ke923690/acacia/'
filename = 'acacia_diagnostic_?d_positive_subs_14000.nc'

# Load files
v = iris.load_cube(filepath + filename, variable)


w_mn = iris.load_cube(filepath +filename, 'w_wind_mean')
w = iris.load_cube(filepath +filename, 'w')
CF =iris.load_cube(filepath +filename, 'ice_cloud_fraction')
TKE =iris.load_cube(filepath +filename, 'tkesg_mean')
z = w.coord('z').points

fig, ax = plt.subplots(2)
ax.flatten()
cbax = fig.add_axes([0.9,0.25, 0.03, 0.5])
c = ax[0].contourf(range(60), z, np.mean(w[0].data, axis = 1).transpose(), vmin = -0.2, vmax = 0.2, cmap = 'RdBu')
ax[1].contourf(range(60), z, np.mean(w[0].data, axis =0).transpose(), vmin = -0.2, vmax = 0.2, cmap = 'RdBu')#
plt.colorbar(c, cax = cbax)
plt.subplots_adjust(right = 0.85)
plt.show()

ts = 0
w_min = iris.load_cube(filepath+filename, 'critical_downdraft')
w_max = iris.load_cube(filepath+filename, 'critical_updraft')
w_bins = iris.load_cube(filepath+filename, 'w_histogram_bins')
w_hist = iris.load_cube(filepath+filename, 'w_histogram_profile')
tke =iris.load_cube(filepath+filename, 'tkesg_mean')
rh_mn= iris.load_cube(filepath+filename, 'rh_mean')
rho = iris.load_cube(filepath+filename, 'rho')
nice =  iris.load_cube(filepath+filename, 'q_ice_number')
nice_cm3 = np.mean((nice*rho).data, axis = (0,1,2))/10e6
ice_mmr = iris.load_cube(filepath+filename, 'ice_mmr_mean')
iwp =iris.load_cube(filepath+filename, 'iwp')
theta = iris.load_cube(filepath+filename, 'th')
P = iris.load_cube(filepath+filename, 'p')
T = metpy.calc.temperature_from_potential_temperature(P.data*unit, theta.data * metpy.units.kelvin)

plt.hist(w_hist.data, bins=w_bins)

fig, ax = plt.subplots()
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(0,19000)
ax.fill_betweenx(y=z, x1=w_min[ts].data, x2=np.min(w[ts].data, axis = (0,1)), color='dimgrey')
ax.plot(w_min[ts].data, z,color = 'k', linestyle ='--', lw=2.5)
ax.plot(np.min(w[ts].data, axis = (0,1)),z, color = 'k', lw=2.5)
ax.fill_betweenx(y=z, x1=w_max[ts].data, x2=np.max(w[ts].data, axis = (0,1)), color='dimgrey')
ax.plot(w_max[ts].data, z, color = 'k', linestyle ='--', lw=2.5)
ax.plot(np.max(w[ts].data,axis = (0,1)),  z,color = 'k', lw=2.5)
ax.vlines(x=0, ymin=0, ymax=19000, color = 'k', linestyle=':', lw=1)
ax.set_xlabel('W (m/s)', color='dimgrey', fontsize = 20)
ax.set_ylabel('Altitude (m)', color='dimgrey', fontsize = 20, rotation = 90)
plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
ax.tick_params(axis='both', which='both', labelsize=18, labelcolor='dimgrey', pad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both',  color='dimgrey', length=5, width = 2.5)
plt.subplots_adjust(right=0.98, top = 0.98, bottom=0.19, left = 0.22)
plt.savefig(filepath+'figures/critical_and_max_min_W.png')
plt.show()


fig, ax = plt.subplots(figsize=(8,12))
ax.set_ylim(0,19000)
ax.set_xlim(0, 1e-5)
ax2 = ax.twiny()
ax2.set_xlim(0,1)
ax.plot(ice_mmr[0].data, z,color = 'k',  lw=2.5, label = 'MMR')
ax2.plot(nice_cm3,z, color = 'r', lw=2.5, label='Nice')
ax.set_xlabel('Ice MMR (kg kg$^{-1}$)', labelpad=10, color='dimgrey', fontsize = 20)
ax2.set_xlabel('N$_{ice}$ (cm$^{-3}$)',color='red', labelpad = 20, fontsize = 20)
ax.set_ylabel('Altitude (m)', color='dimgrey', fontsize = 20, rotation = 90)
plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
plt.setp(ax2.spines.values(), linewidth=2, color='dimgrey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_color('red')
ax.tick_params(axis='both', which='both', labelsize=18, labelcolor='dimgrey', pad=10,  color='dimgrey', length=5, width = 2.5)
ax2.tick_params(axis='both', which='both', labelsize=18, labelcolor='r', pad=10, color='r', length=5, width = 2.5)
plt.subplots_adjust(right=0.95, top = 0.85, bottom=0.19, left = 0.22)
plt.savefig(filepath+'figures/Nice_ice_MMR.png')
plt.show()

unit_dict={'u': 'm/s',
           'v': 'm/s',
           'w': 'm/s',
           'q': 'kg/kg',
           'theta': 'K',
           'z': 'km'}

df={}
df['u'] = iris.load_cube(filepath+filename, 'u_wind_mean')
df['v'] = iris.load_cube(filepath+filename, 'v_wind_mean')
df['w'] = iris.load_cube(filepath+filename, 'w_wind_mean')
df['theta'] = iris.load_cube(filepath+filename, 'theta_mean')
df['q'] = iris.load_cube(filepath+filename, 'vapour_mmr_mean')

