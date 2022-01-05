import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D

filepath = '/storage/silver/acacia/ke923690/acacia/'
filename = 'Yang_test14a*d_7200.nc'

# Load files
v = iris.load_cube(filepath + filename, variable)

w_mn = iris.load_cube(filepath +filename, 'w_wind_mean')
w = iris.load_cube(filepath +filename, 'w')
CF =iris.load_cube(filepath +filename, 'ice_cloud_fraction')
IWP = iris.load_cube(filepath + filename, 'tot_iwp')
iwp = iris.load_cube(filepath + filename, 'iwp')
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


fig, ax = plt.subplots(1)
ax.flatten()
cbax = fig.add_axes([0.8,0.2, 0.03, 0.6])
#cbax = fig.add_axes([0.8,0.53, 0.03, 0.35])
#cbax2 = fig.add_axes([0.9,0.25, 0.03, 0.5])
c = ax.pcolormesh(iwp.data.mean(axis=0), vmin = 0, vmax = 0.005)#, cmap = 'RdBu')
#ax[1].contourf(CF.data.mean(axis=0), vmin = 0, vmax = 1, cmap = 'RdBu')#
#plt.colorbar(d, cax = cbax2)
cb = plt.colorbar(c, cax = cbax,  ticks = [0., 0.0025, 0.005])
#cb.ticklabelformat(axis='y', style='sci', scilimits=(0,0))
#cb.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
ax.text( 0.25,1.1, 'IWP (kg m$^{-2}$)', fontname='Helvetica', color='dimgrey', fontsize=24, transform=ax.transAxes)
cb.formatter.set_powerlimits((0,0))
cb.solids.set_edgecolor("face")
cb.outline.set_edgecolor('dimgrey')
cb.ax.tick_params(which='both', axis='both', labelsize=14, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
cb.outline.set_linewidth(2)
plt.subplots_adjust(right = 0.75, top = 0.8)
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
nice_cm3 = (np.mean(nice.data, axis = (1,2))*rho.data)/10e6
ice_mmr = iris.load_cube(filepath+filename, 'ice_mmr_mean')
iwp =iris.load_cube(filepath+filename, 'iwp')
theta = iris.load_cube(filepath+filename, 'theta_mean')
#th_init = iris.load_cube(filepath+filename, 'thinit')
theta = theta+th_init
P = iris.load_cube(filepath+filename, 'p')
T = metpy.calc.temperature_from_potential_temperature(P.data*unit, theta.data * metpy.units.kelvin)

plt.hist(w_hist.data, bins=w_bins)

fig, ax = plt.subplots()
ax.set_xlim(-.4, .4)
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


fig, ax = plt.subplots(figsize=(5,6))
ax.set_xlim(0, 2e-7)
ax2 = ax.twiny()
ax2.set_xlim(0,1.5)
#ax2.set_xticks([0, 0.005, 0.01])
ax.plot(ice_mmr[1, 70:100].data, z[70:100]/1000,color = 'k',  lw=2.5, label = 'MMR')
ax2.plot(nice_cm3[1, 70:100],z[70:100]/1000, color = 'r', lw=2.5, label='Nice')
ax.set_xlabel('Ice MMR (kg kg$^{-1}$)', labelpad=10, color='dimgrey', fontsize = 20)
ax2.set_xlabel('N$_{ice}$ (cm$^{-3}$)',color='red', labelpad = 20, fontsize = 20)
ax.set_ylabel('Altitude (km)', color='dimgrey', fontsize = 20, rotation = 90)
plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
plt.setp(ax2.spines.values(), linewidth=2, color='dimgrey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_color('red')
ax.tick_params(axis='both', which='both', labelsize=18, labelcolor='dimgrey', pad=10,  color='dimgrey', length=5, width = 2.5)
ax2.tick_params(axis='both', which='both', labelsize=18, labelcolor='r', pad=10, color='r', length=5, width = 2.5)
plt.subplots_adjust(right=0.92, top = 0.82, bottom=0.2, left = 0.28)
plt.savefig(filepath+'figures/Nice_ice_MMR_DeMott_ts1.png')
plt.show()


fig, ax = plt.subplots(figsize=(5,6))
ax.set_xlim(0, 1.5e-7)
ax2 = ax.twiny()
ax2.set_xlim(0,5e-10)
#ax2.set_xticks([0, 0.005, 0.01])
ax.plot(proc_dict['cond'][0, 70:100].data, z[70:100]/1000,color = 'k',  lw=2.5, label = 'cond')
ax2.plot(proc_dict['ice_nucl'][0, 70:100].data,z[70:100]/1000, color = 'r', lw=2.5, label='ice nucl')
ax.set_xlabel('Condensation (kg kg$^{-1}$ s$^{-1}$)', labelpad=10, color='dimgrey', fontsize = 20)
ax2.set_xlabel('Ice nucleation (kg kg$^{-1}$ s$^{-1}$)',color='red', labelpad = 20, fontsize = 20)
ax.set_ylabel('Altitude (km)', color='dimgrey', fontsize = 20, rotation = 90)
plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
plt.setp(ax2.spines.values(), linewidth=2, color='dimgrey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_color('red')
ax.tick_params(axis='both', which='both', labelsize=18, labelcolor='dimgrey', pad=10,  color='dimgrey', length=5, width = 2.5)
ax2.tick_params(axis='both', which='both', labelsize=18, labelcolor='r', pad=10, color='r', length=5, width = 2.5)
plt.subplots_adjust(right=0.92, top = 0.82, bottom=0.2, left = 0.28)
plt.savefig(filepath+'figures/cond_v_ice_nucl_1s.png')
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

th_DM = iris.load_cube(filepath + '*test4b_*_DeMott_?d_21600.nc', 'vapour_mmr_mean')
th_C = iris.load_cube(filepath + '*test4b_q=1.3_?d_14400.nc', 'vapour_mmr_mean')
th_C7 = iris.load_cube(filepath + '*test7_q=1.3_?d_21600.nc', 'vapour_mmr_mean')

fig, ax = plt.subplots(figsize=(4,6))
ax.set_ylim(0,19)
#ax.set_xlim(250, 450)
ax.plot(th_DM[0].data, z/1000,color = 'k',  lw=2.5, label = 'DM')
ax.plot(th_C[0].data, z/1000,color = 'b',  lw=2.5, label = 'C4')
ax.plot(th_C7[0].data, z/1000,color = 'g',  lw=2.5, label = 'C7')
#ax.set_xlabel('Theta (K)', labelpad=10, color='dimgrey', fontsize = 20)
ax.set_ylabel('Altitude (km)', color='dimgrey', fontsize = 20, rotation = 90)
plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
plt.setp(ax2.spines.values(), linewidth=2, color='dimgrey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', labelsize=18, labelcolor='dimgrey', pad=10,  color='dimgrey', length=5, width = 2.5)
plt.subplots_adjust(right=0.95, top = 0.85, bottom=0.19, left = 0.22)
#plt.savefig(filepath+'figures/Nice_ice_MMR_DeMott.png')
plt.legend()
plt.show()