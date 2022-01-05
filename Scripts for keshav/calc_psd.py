from math import gamma, pi
import numpy as np
import iris
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/users/ke923690/scripts/Tools/')
from divg_temp_colourmap import shiftedColorMap
import scipy.stats as stats
import imageio
import scipy

filepath = '/storage/silver/acacia/ke923690/acacia/new_runs/'

p1 = 3.0 # number
p2 = 0.0 # mass
mu = 2.5 # shape parameter
c_x = pi*997.0/6.0
rho = iris.load_cube(filepath + '*14a*CTRL*3600.nc', 'rho')
z = rho.coord('z').points

def calc_effective_radius(N,M,p1,p2,mu,c_x):
    ''' Function to calculate re.

    Author: Rachel Hawker

    inputs: N, M: ice number and mass (m-3 and kg m-3)
    *Make sure these are converted from kg-1 in MONC

    p1, p2, mu, c_x: constants in mphys_parameters.F90

    outputs: re, N0, lam

    '''
    m1 = M/c_x
    m2 = N
    j1 = 1.0/(p1-p2)
    lam = ((gamma(1.0+mu+p1)/gamma(1.0+mu+p2))*(m2/m1))**(j1)
    #l2 = 1.0 / (p2 - p1)
    #lam = ((gamma(1.0 + mu + p2) / gamma(1.0 + mu + p1)) * (m1 / m2)) ** l2 # from lookup.F90
    Moment2 = (N/(lam**2))*((gamma(1+mu+2))/gamma(1+mu))
    Moment3 = (N/(lam**3))*((gamma(1+mu+3))/gamma(1+mu))
    effective_radius = Moment3/Moment2
    effective_radius = effective_radius*1e6 #convert to microns
    n0 = lam ** p2 * m2 * ((gamma(1.0+mu)/gamma(1.0+mu+p2)))
    return effective_radius, n0, lam

PSD_dict = {}
for fn, nm in zip(['file_str1', 'file_str2'  ...], ['CTRL', 'sim1' ...]):
    PSD_dict[nm] = {}
    try:
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'q_ice_number')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'q_ice_number')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'q_ice_number')
        PSD_dict[nm]['N'] = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
        PSD_dict[nm]['N'] = PSD_dict[nm]['N'] * np.tile(rho[0].data, (PSD_dict[nm]['N'].shape[0], 1))[:, np.newaxis,
                                                np.newaxis, :]
        f1 = iris.load_cube(filepath + fn + '3600.nc','clbas' )
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'clbas')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'clbas')
        PSD_dict[nm]['clbas'] = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
        f1 = iris.load_cube(filepath + fn + '3600.nc','cltop' )
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'cltop')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'cltop')
        PSD_dict[nm]['cltop'] = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'q_ice_mass')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'q_ice_mass')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'q_ice_mass')
        PSD_dict[nm]['M'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        PSD_dict[nm]['M'] = PSD_dict[nm]['M'] * np.tile(rho[0].data, (PSD_dict[nm]['M'].shape[0], 1))[:, np.newaxis, np.newaxis, :]
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'temperature')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'temperature')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'temperature')
        PSD_dict[nm]['T'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        PSD_dict[nm]['T'] = np.ma.masked_where((PSD_dict[nm]['M']< 1e-10), PSD_dict[nm]['T'])
    except:
        print('Couldn\'t find PSD properties, soz m8')
    try:
        PSD_dict[nm]['re'], PSD_dict[nm]['n0'],PSD_dict[nm]['lam'] = calc_effective_radius(PSD_dict[nm]['N'], PSD_dict[nm]['M'], p1, p2, mu, c_x)
        PSD_dict[nm]['re'][PSD_dict[nm]['re'] > 1000] = 0.
        PSD_dict[nm]['re'][PSD_dict[nm]['re'] < 0] = 0.
    except:
        print('Oops - forgot my keys')

time_srs_3d = np.concatenate((f1.coord('time_series_300_300').points, f2.coord('time_series_300_300').points, f3.coord('time_series_300_300').points), axis=0)
alt = f1.coord('zn').points

import matplotlib.ticker as ticker

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def CTRL_contourplot():
    fig, ax = plt.subplots(2, 2, figsize=(13,8), sharex=True, sharey=True)
    ax = ax.flatten()
    cbax1 = fig.add_axes([0.42, 0.61, 0.015, 0.3])
    cbax2 = fig.add_axes([0.88, 0.61, 0.015, 0.3])
    cbax3 = fig.add_axes([0.42, 0.17, 0.015, 0.3])
    cbax4 = fig.add_axes([0.88, 0.17, 0.015, 0.3])
    for axs, var, n, tit in zip(ax, [ 'M', 'N', 'T', 're', ], [1,2,3,4], ['Ice MMR', 'ICNC', 'Temperature', 'r$_{e}$']):
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.tick_params(axis='both', which='both', length=8, width=2, direction = 'in', color='dimgrey', labelsize=16, tick1On=True,
                       tick2On=True, labelcolor='dimgrey', pad=10)
        if var == 'M':
            vmin, vmax = 1e-10, 1.2e-5
        if var == 'N':
            vmin, vmax = 1e3, 5e6
        if var == 'T':
            vmin, vmax = 220, 240
        if var == 're':
            vmin, vmax = 0, 300
        try:
            c = axs.contourf(time_srs_3d, alt[60:95], np.nanmean(PSD_dict['CTRL'][var][:, :, :, 60:95], axis = (1,2)).transpose(), cmap = 'Greys', vmin=vmin, vmax=vmax, levels = np.linspace(vmin, vmax, 21))
            axs.set_facecolor('#BEDBE4')
            axs.plot(time_srs_3d,np.nanmean(PSD_dict['CTRL']['clbas'], axis=(1, 2)), color='darkgrey')
            axs.plot(time_srs_3d,np.nanmean(PSD_dict['CTRL']['cltop'], axis=(1, 2)), color='darkgrey')
            axs.set_xlim(350, 10800)
            axs.set_title(tit, fontsize = 20, color = 'dimgrey', fontweight = 'bold')
        except ValueError:
            c = axs.contourf(time_srs_3d, alt[60:95], np.nanmean(PSD_dict['CTRL'][var][:, :, :, 60:95], axis = (1,2)).transpose(), cmap = 'Greys', vmin=vmin, vmax=vmax,levels = np.linspace(vmin, vmax, 21))
            axs.set_facecolor('#BEDBE4')
            axs.plot( np.nanmean(PSD_dict['CTRL']['clbas'], axis=(1, 2)), color='darkgrey')
            axs.plot( np.nanmean(PSD_dict['CTRL']['cltop'], axis=(1, 2)),color='darkgrey')
            axs.set_title(tit, fontweight= 'bold')
            axs.set_xlim(350, 10800)
            axs.set_title(tit, fontsize = 20, color = 'dimgrey', fontweight = 'bold')
        if (n % 2) != 0: # odd number
            axs.set_ylabel('altitude (m)', fontsize=16, color='dimgrey')
        if n > 2:
            axs.set_xlabel('time (s)', fontsize=16, color='dimgrey')
        axs.tick_params(direction='in')
        if var == 'M':
            cb = plt.colorbar(c, cax=cbax1, orientation='vertical', extend='both', ticks = np.linspace(vmin, vmax, 5), format=ticker.FuncFormatter(fmt))
            cb.ax.set_title('          (kg m$^{-3}$)', pad = 10, fontsize=16, color='dimgrey')
        elif var == 'N':
            cb = plt.colorbar(c, cax=cbax2, orientation='vertical', extend='both', ticks=np.linspace(vmin, vmax, 5), format=ticker.FuncFormatter(fmt))
            cb.ax.set_title('  (m$^{-3}$)', pad = 10, fontsize=16, color='dimgrey')
        elif var == 'T':
            cb = plt.colorbar(c, cax=cbax3, orientation='vertical', extend='both', ticks=np.linspace(vmin, vmax, 5))
            cb.ax.set_title('(K)', pad = 10, fontsize=16, color='dimgrey')
        elif var == 're':
            cb = plt.colorbar(c, cax=cbax4, orientation='vertical', extend='both', ticks=np.linspace(vmin, vmax, 5))
            cb.ax.set_title(' ($\mu$m)', pad = 10, fontsize=16, color='dimgrey')
        cb.solids.set_edgecolor("face")
        cb.outline.set_edgecolor('dimgrey')
        cb.ax.tick_params(which='both', axis='both', labelsize=18, labelcolor='dimgrey', pad=10, size=8, width =2, color='dimgrey', direction='in', tick1On=True,tick2On=False)
        cb.outline.set_linewidth(2)
    plt.subplots_adjust(left = 0.12, right = 0.85, top = 0.95, hspace = 0.15, wspace= 0.53, bottom = 0.1)
    plt.savefig(filepath + '../figures/Hovmoller_'+ var + '_CTRL.png')
    plt.show()

CTRL_contourplot()

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

import matplotlib.ticker

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

def contourplot(var):
    #fig, ax = plt.subplots(4,2, figsize = (6,13), sharex=True, sharey=True)
    fig, ax = plt.subplots(3, 2, figsize=(6, 10), sharex=True, sharey=True)
    ax = ax.flatten()
    cbax = fig.add_axes([0.3,0.08, 0.5, 0.02])
    ## Define colour levels
    vmin = -100
    vmax = 100
    levels = np.linspace(vmin, vmax, 41)
    for axs, sim, n in zip(ax, [ 'sim1', 'sim2', 'sim3', ... ], [1,2,3,4,5,6,7,8]):
        try:
            c = axs.contourf(time_srs_3d, alt[60:95],
                             (np.nanmean(PSD_dict[sim][var][:274, :, :, 60:95],axis = (1,2)) -
                             np.nanmean(PSD_dict['CTRL'][var][:274, :,:, 60:95],axis = (1,2))).transpose(),
                              cmap = 'bwr', vmin = vmin, vmax =vmax, levels = levels) #
            axs.set_facecolor('#BEDBE4')
            axs.plot(time_srs_3d, np.nanmean(PSD_dict[sim]['clbas'][:274, :,:], axis = (1,2)),color = 'darkgrey')
            axs.plot(time_srs_3d, np.nanmean(PSD_dict[sim]['cltop'][:274, :, :,], axis=(1, 2)), color='darkgrey')
        except ValueError:
            c = axs.contourf(time_srs_3d, alt[60:95],
                             (np.nanmean(PSD_dict[sim][var][:, :,:, 60:95],axis = (1,2)).transpose() -
                              np.nanmean(PSD_dict['CTRL_lowres'][var][:, 60:95],axis = (1,2)).transpose()),
                              cmap = 'bwr')#, vmin = vmin, vmax =vmax, levels = levels)
            axs.contour(time_srs_3d, alt[60:95],
                            np.nanmean(PSD_dict['CTRL_lowres'][var][:282, :,:, 60:95], axis = (1,2)).transpose() > 0.,
                            levels=[0], colors = 'darkgrey')
        axs.set_title(sim + ' - CTRL', fontweight= 'bold') #
        if (n % 2) != 0: # odd number
            axs.set_ylabel('altitude (m)')
        if n > 6:
            axs.set_xlabel('time (s)')
        axs.tick_params(direction='in')
        if var == 'M':
            cb = plt.colorbar(c, cax=cbax, orientation='horizontal', extend='both', ticks = [-4.9e-5, -5e-6, -5e-7, -5e-8, -1e-8, 0, 1e-8, 5e-8,  5e-7, 5e-6, 4.9e-5])
            # Hack
            cb.set_ticklabels([r'$-5 \times 10^{-5}$', '', r'$-5 \times 10^{-7}$', '',r'-1 $\times$ 10$^{-8}$', '0', r'$1 \times 10^{-8}$', '', r'$5 \times 10^{-7}$','', r'$5 \times 10^{-5}$'])
            cb.ax.set_xlabel('Ice MMR (g kg$^{-1}$)')
        elif var == 'N':
            cb = plt.colorbar(c, cax=cbax, orientation='horizontal', extend='both',
                              ticks=[-1e7, -1e6, -1e5, -1e4, -1e3, -1e2, 0, 1e2, 1e3,1e4, 1e5, 1e6, 1e7])
            #cb.set_ticklabels([r'$-5 \times 10^{-5}$', '', r'$-5 \times 10^{-7}$', '', r'-1 $\times$ 10$^{-8}$', '0',
                               #r'$1 \times 10^{-8}$', '', r'$5 \times 10^{-7}$', '', r'$5 \times 10^{-5}$'])
            cb.ax.set_xlabel('ICE (kg$^{-1}$)')
        elif var == 're':
            cb = plt.colorbar(c, cax=cbax, orientation='horizontal')#, extend='both', ticks=[0, 50, 100, 150, 200, 250, 300])
            cb.ax.set_xlabel('r$_e$ ($\mu$m)')
    plt.subplots_adjust(right = 0.95, top = 0.95, hspace = 0.23, wspace= 0.08, bottom = 0.15)
    plt.savefig(filepath + '../figures/Contour_plot_'+ var + '_ICNC_runs_2000s.png')
    plt.show()

contourplot('re')

## Plot animated PSD
for t in range(PSD_dict['CTRL']['re'].shape[0]):
#for t in [0]:
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlim(0, 250)
    ax.set_ylim(100, 1e8)
    #ax.scatter(re_1d, N_1d)
    ax.scatter(PSD_dict['CTRL']['re'][t,:,:, 77:85], PSD_dict['CTRL']['N'][t,:,:, 77:85], color = 'blue', label = 'CTRL', alpha=0.5)
    ax.vlines(np.ma.median(PSD_dict['CTRL']['re'][t,:,:, 77:85]), ymin=100, ymax = 1e8, colors = '#0C2A57', linestyle = '--', lw=2)
    ax.text(x=np.ma.median(PSD_dict['CTRL']['re'][t,:,:, 77:85])+10, y=5e7, s=str(int(np.ma.median(PSD_dict['CTRL']['re'][t,:,:, 77:85]))) + '$\mu$m', color = '#0C2A57', fontsize = 12)
    ax.scatter(PSD_dict['ICNCx2_9-10_2000']['re'][t,:,:, 77:85], PSD_dict['ICNCx2_9-10_2000']['N'][t,:,:, 77:85], color = 'orange', label = 'ICNCx2 9-10 km', alpha=0.5)
    ax.vlines(np.ma.median(PSD_dict['ICNCx2_9-10_2000']['re'][t,:,:, 77:85]), ymin=100, ymax = 1e8, colors = '#71340D', linestyle = '--', lw=2)
    ax.text(x=np.ma.median(PSD_dict['ICNCx2_9-10_2000']['re'][t,:,:, 77:85])+10, y=1e7, s=str(int(np.ma.median(PSD_dict['ICNCx2_9-10_2000']['re'][t,:,:, 77:85]))) + '$\mu$m', color = '#71340D', fontsize = 12)
    ax.set_xlabel('Particle effective radius ($\mu$m)')
    ax.set_ylabel('Particle number density (m$^{-3}$)')
    plt.legend()
    plt.savefig(filepath + '../figures/PSD_9-10_CTRL_ICNCx2_9-10km_ts_' + str(t) + '.png')
    #plt.show()

plt.close('all')

# Create GIF
filenames = []
# create file name and append it to a list
for i in range(27):
    fn = filepath + f'../figures/PSD_8-9_CTRL_ICNCx2_8-9km_ts_{i}.png'
    filenames.append(fn)

for i in range(6):
    filenames.append(filenames[-1]) # add freeze frame at end

with imageio.get_writer(filepath + '../figures/PSD_8-9_CTRL_v_ICNCx2_8-9km_animated.gif', mode='I') as writer:
    for f in filenames:
        image = imageio.imread(f)
        writer.append_data(image)
