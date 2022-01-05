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
#for fn, nm in zip([ '*22a*', '*22b*', '*22c*', '*22d*', '*22e*', '*22f*'], [ 'CTRL', 'ICNCx2_8-9', 'ICNCx2_9-10', 'no_qF', 'ICNCx1.5_8-9', 'ICNCx1.5_9-10']):
#for fn, nm in zip(['*17_CTRL*', '*18a*', '*18b*',  '*22a*', '*22b*', '*22c*', '*20i*', '*20j*','*20k*', '*20l*','*20m*', '*20n*'], [ 'CTRL', 'ICNCx2_8-9_2000', 'ICNCx2_9-10_2000','uv', 'ICNCx2_8-9', 'ICNCx2_9-10', 'ICNCx1.25_8-9', 'ICNCx1.25_9-10', 'ICNCx1.5_8-9', 'ICNCx1.5_9-10', 'ICNCx1.1_8-9', 'ICNCx1.1_9-10']):
        #['*15a*', '*14a*?d_', '*14b*?d_', '*14c*?d_', '*14e*?d_','*14f*?d_', '*14q*?d_', '*14m*?d_', '*14n*?d_', '*14o*?d_', '*14p*?d_', '*16*', '*14r*', '*14s*', '*14j*', '*14k*'  '*17p*9-10*', '*17o*8-9*', '*17s*', '*17r*',],
        #['CTRL', 'CTRL_lowres', 'SML', 'FXD', 'CDNCx2_SML', 'CDNCx2', 'INPx10', 'ICNCx1.1_8-9', 'ICNCx1.1_9-10', 'ICNCx2_8-9_2000', 'ICNCx2_9-10_2000', 'no_inuc', 'ICNCx0.5_8-9_2000', 'ICNCx0.5_9-10_2000', 'INPx0.5_8-9_2000', 'INPx0.5_9-10_2000', 'ICNCx2_9-10_4000', 'ICNCx2_8-9_4000','ICNCx0.5_9-10_4000', 'ICNCx0.5_8-9_4000',]):
for fn, nm in zip(['*22a*'], ['CTRL']):
    PSD_dict[nm] = {}
    try:
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'q_ice_number')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'q_ice_number')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'q_ice_number')
        f4 = iris.load_cube(filepath + fn + '14400.nc', 'q_ice_number')
        PSD_dict[nm]['N'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis = 0)
        PSD_dict[nm]['N'] = PSD_dict[nm]['N'] * np.tile(rho[0].data, (PSD_dict[nm]['N'].shape[0], 1))[:, np.newaxis, np.newaxis, :]
        f1 = iris.load_cube(filepath + fn + '3600.nc','clbas' )
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'clbas')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'clbas')
        f4 = iris.load_cube(filepath + fn + '14400.nc', 'clbas')
        PSD_dict[nm]['clbas'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis = 0)
        f1 = iris.load_cube(filepath + fn + '3600.nc','cltop' )
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'cltop')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'cltop')
        f4 = iris.load_cube(filepath + fn + '14400.nc', 'cltop')
        PSD_dict[nm]['cltop'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis = 0)
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'q_ice_mass')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'q_ice_mass')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'q_ice_mass')
        f4 = iris.load_cube(filepath + fn + '14400.nc', 'q_ice_mass')
        PSD_dict[nm]['M'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        PSD_dict[nm]['M'] = PSD_dict[nm]['M'] * np.tile(rho[0].data, (PSD_dict[nm]['M'].shape[0], 1))[:, np.newaxis, np.newaxis, :]
       #try:
       #    PSD_dict[nm]['time_srs'] = np.concatenate((f1.coord('time_series_10_30').points,
       #                                               f2.coord('time_series_10_30').points,
       #                                               f3.coord('time_series_10_30').points), axis=0)
    #except:
     #       print('Tiiiiiiiime.... to DIEEEEEE')
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'temperature')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'temperature')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'temperature')
        f4 = iris.load_cube(filepath + fn + '14400.nc', 'temperature')
        PSD_dict[nm]['T'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        PSD_dict[nm]['T'] = np.ma.masked_where((PSD_dict[nm]['M']< 1e-10), PSD_dict[nm]['T'])
        time_srs_3d = np.concatenate((f1.coord('time_series_300_300').points, f2.coord('time_series_300_300').points,f3.coord('time_series_300_300').points, f4.coord('time_series_300_300').points), axis=0)
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'rhi_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'rhi_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'rhi_mean')
        f4 = iris.load_cube(filepath + fn + '14400.nc', 'rhi_mean')
        PSD_dict[nm]['RHi'] = (np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0) * 100. ) + 100
        PSD_dict[nm]['RHi'][PSD_dict[nm]['RHi'] == 100] = np.nan # supersaturation
        #PSD_dict[nm]['RHi'] = np.ma.masked_where(np.nanmean(PSD_dict[nm]['M'], axis = (1,2))< 1e-10, PSD_dict[nm]['RHi'])
    except:
        print('Couldn\'t find PSD properties, soz m8')
    try:
        PSD_dict[nm]['re'], PSD_dict[nm]['n0'],PSD_dict[nm]['lam'] = calc_effective_radius(PSD_dict[nm]['N'], PSD_dict[nm]['M'], p1, p2, mu, c_x)
        PSD_dict[nm]['re'][PSD_dict[nm]['re'] > 1000] = 0.
        PSD_dict[nm]['re'][PSD_dict[nm]['re'] < 0] = 0.
    except:
        print('Oops - forgot my keys')

time_srs_1d = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points, f3.coord('time_series_10_30').points), axis=0)
alt = f1.coord('zn').points

def plot_re_srs(sim_list):
    ## Plot time series of re evolution from different simulations
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize = (10,6))
    ax = ax.flatten()
    for axs in ax:
        axs.plot(time_srs, np.average(PSD_dict['CTRL']['re'][:,:,:,60:95], weights = PSD_dict['CTRL']['M'][:,:,:,60:95], axis=(1,2,3)), lw=3, color = 'k', label = 'CTRL')
        #axs.fill_between(x=time_srs, y1=sim_list[0].min(axis=(1,2,3)), y2=sim_list[0].max(axis=(1,2,3)), color='darkgrey', alpha = 0.3)
        plt.setp(axs.spines.values(), linewidth=1, color='dimgrey')
    lns = [Line2D([0], [0], color='k', linewidth=2.5)]
    labs = ['CTRL']
    for axs, re, lab, col, plotlab in zip(ax,sim_list, ['ICNCx2_8-9_2000', 'ICNCx2_9-10_2000', 'ICEx2_8-9_2000', 'ICEx2_9-10_2000'], ['darkred', 'darkblue', 'darkgreen', 'darkorange'],['a', 'b', 'c', 'd']):
        axs.plot(time_srs, np.average(re[:,:,:,60:95], weights = PSD_dict[lab]['M'][:,:,:,60:95], axis=(1,2,3)), lw=3, color = col, label = lab)
        #axs.fill_between(x=time_srs, y1=re[:,:,:,60:95].min(axis=(1,2,3)), y2=re[:,:,:,60:95].max(axis=(1,2,3)), color=col, alpha = 0.3)
        axs.set_xlim(0, 10800)
        axs.set_ylim(0,180)
        axs.text(0.05, 0.85, s=plotlab, color = 'dimgrey', fontsize = 24, transform = axs.transAxes)
        axs.tick_params(which='both', axis='both', direction='in', color = 'dimgrey', labelcolor='dimgrey', labelsize = 16)
        lns.append(Line2D([0],[0], color=col, linewidth = 2.5))
        labs.append(lab)
        axs.set_title(lab, fontsize=16, color = 'dimgrey')
    lgd = ax[3].legend(lns, labs, bbox_to_anchor=(1, 1.4), loc=2, fontsize=14)
    ax[2].set_xlabel('Time (s)', fontsize=16, color = 'dimgrey')
    ax[3].set_xlabel('Time (s)', fontsize=16, color = 'dimgrey')
    ax[0].set_ylabel('r$_e$ ($\mu$m)', color= 'dimgrey', labelpad=35, fontsize=16, rotation=0)
    ax[2].set_ylabel('r$_e$ ($\mu$m)', color= 'dimgrey', labelpad=35, fontsize=16, rotation=0)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(left = 0.15, wspace = 0.05, top=0.95, hspace=0.08, right = 0.8, bottom=0.1)
    plt.savefig(filepath + '../figures/time_srs_re_ICE_ICNCx2.png')
    plt.show()

plot_re_srs([PSD_dict['ICNCx2_8-9_2000']['re'],PSD_dict['ICNCx2_9-10_2000']['re'],PSD_dict['ICEx2_8-9_2000']['re'],PSD_dict['ICEx2_9-10_2000']['re']])

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
    #for axs, var, n, tit in zip(ax, ['M', 'N', 'RHi', 're', ], [1, 2, 3, 4], ['Ice MMR', 'ICNC', 'RH$_{i}$', 'r$_{e}$']):
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
        if var  == 'RHi':
            vmin, vmax = 100, 120
        try:
            c = axs.contourf(time_srs_3d, alt[60:95], np.nanmean(PSD_dict['CTRL'][var][:, :, :, 60:95], axis = (1,2)).transpose(), cmap = 'Greys', vmin=vmin, vmax=vmax, levels = np.linspace(vmin, vmax, 21))
            axs.set_facecolor('#BEDBE4')
            axs.plot(time_srs_3d,np.nanmean(PSD_dict['CTRL']['clbas'], axis=(1, 2)), color='darkgrey')
            axs.plot(time_srs_3d,np.nanmean(PSD_dict['CTRL']['cltop'], axis=(1, 2)), color='darkgrey')
            axs.set_xlim(350, 14400)
            axs.set_title(tit, fontsize = 20, color = 'dimgrey', fontweight = 'bold')
        except ValueError:
            c = axs.contourf(time_srs_3d, alt[60:95], np.nanmean(PSD_dict['CTRL'][var][:, :, :, 60:95], axis = (1,2)).transpose(), cmap = 'Greys', vmin=vmin, vmax=vmax,levels = np.linspace(vmin, vmax, 21))
            axs.set_facecolor('#BEDBE4')
            axs.plot( np.nanmean(PSD_dict['CTRL']['clbas'], axis=(1, 2)), color='darkgrey')
            axs.plot( np.nanmean(PSD_dict['CTRL']['cltop'], axis=(1, 2)),color='darkgrey')
            axs.set_title(tit, fontweight= 'bold')
            axs.set_xlim(350, 14400)
            axs.set_title(tit, fontsize = 20, color = 'dimgrey', fontweight = 'bold')
        except IndexError:
            c = axs.contourf(time_srs_1d, alt[60:95], PSD_dict['CTRL'][var][:, 60:95].transpose(), cmap = 'Greys', vmin=vmin, vmax=vmax,levels = np.linspace(vmin, vmax, 21))
            axs.set_facecolor('#BEDBE4')
            axs.plot( np.nanmean(PSD_dict['CTRL']['clbas'], axis=(1, 2)), color='darkgrey')
            axs.plot( np.nanmean(PSD_dict['CTRL']['cltop'], axis=(1, 2)),color='darkgrey')
            axs.set_title(tit, fontweight= 'bold')
            axs.set_xlim(350, 14400)
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
        elif var == 'RHi':
            cb = plt.colorbar(c, cax=cbax3, orientation='vertical', extend='both', ticks=np.linspace(vmin, vmax, 5))
            cb.ax.set_title(' (%)', pad = 10, fontsize=16, color='dimgrey')
        cb.solids.set_edgecolor("face")
        cb.outline.set_edgecolor('dimgrey')
        cb.ax.tick_params(which='both', axis='both', labelsize=18, labelcolor='dimgrey', pad=10, size=8, width =2, color='dimgrey', direction='in', tick1On=True,tick2On=False)
        cb.outline.set_linewidth(2)
    plt.subplots_adjust(left = 0.12, right = 0.85, top = 0.95, hspace = 0.15, wspace= 0.53, bottom = 0.1)
    plt.savefig(filepath + '../figures/Hovmoller_'+ var + '_CTRL.png')
    plt.show()

CTRL_contourplot()

## n(x) = n0 * x ** mu exp(-lambda x ** gamma) # PSD format
x = np.linspace(0, 200, 21)
n_x = (n0 * x ** mu) * np.exp(-lam * x ** gamma)

def re_map(nm):
    #Plot 2D maps of re
    fig, ax = plt.subplots(2,4, figsize = (12,5))
    CbAx = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    ax = ax.flatten()
    plt.title(nm, fontsize =  18)
    for axs, t in zip(ax, [0, 3, 7, 11, 15, 19, 23, 27]):
        axs.axis('off')
        c = axs.pcolormesh(np.average(PSD_dict[nm]['re'][t], weights=PSD_dict[nm]['M'][t], axis=2), vmin = 15, vmax = 65)
        axs.set_title(str(int(time_srs_3d[t])) + 's')
    plt.subplots_adjust(left = 0.05, right = 0.85)
    cb = plt.colorbar(c, cax= CbAx, orientation='vertical')
    cb.ax.set_xlabel('IWC weighted r$_{e}$ (cm$^{-3}$)')
    plt.savefig(filepath + '../figures/re_map_time_' + nm + '.png')
    plt.show()

for nm in PSD_dict.keys():
    re_map(nm)

a, loc, scale = mu, n0.mean(), lam.mean()
#x = np.linspace(scipy.stats.gamma.ppf(0.001, a), scipy.stats.gamma.ppf(0.999, a), 300)
x = np.linspace(0.01,30, 30)
plt.plot(x, scipy.stats.gamma.pdf(x, a, scale = lam_d), 'r-', lw=5, alpha=0.6, label='gamma pdf')
plt.show()

PSD_dict['CTRL']['N'][PSD_dict['CTRL']['N']==0.] = np.nan
N_prof = np.nanmean(PSD_dict['CTRL']['N'][0], axis = (0,1))
PSD_dict['CTRL']['M'][PSD_dict['CTRL']['M']==0.] = np.nan
M_prof = np.nanmean(PSD_dict['CTRL']['M'][0], axis = (0,1))

re, n0, lam = calc_effective_radius(N_prof, M_prof, p1, p2, mu, c_x)

y = stats.gamma.pdf(x, a=mu, scale=lam_d,  loc=n0)
plt.plot(x, y)
plt.show()

## Heymsfield et al. (2002) method of calculating lambda and n0 as a function of ratios of moments
M1 = np.nanmean(M_prof)
M2 = np.nanmean(N_prof)
lam_d = (M1*(mu+2))/M2
n0 = (M1*(lam_d**(mu +2)))/math.gamma(mu+2)

fig, ax = plt.subplots(4,2, figsize = (6,13), sharex=True, sharey=True)
ax = ax.flatten()
#cbax = fig.add_axes([0.9,0.25, 0.03, 0.5])
cbax = fig.add_axes([0.3,0.08, 0.5, 0.02])
for axs, sim, n in zip(ax, PSD_dict.keys(), [1,2,3,4,5,6,7,8]):
    c = axs.contourf(time_srs_3d, alt[60:95], np.ma.masked_where((PSD_dict[sim]['M'][:,:,:,60:95].mean(axis = (1,2)).transpose()<10e-15),PSD_dict[sim]['M'][:,:,:,60:95].mean(axis = (1,2)).transpose()),
                     levels = np.linspace(0, 1.2e-5, 15),vmin=0, vmax=1.2e-5) # M
                     #levels = np.linspace(0, 7e6, 15),vmin=0, vmax=7e6) # N
    axs.set_title(sim, fontweight= 'bold')
    if (n % 2) != 0: # odd number
        axs.set_ylabel('altitude (m)')
    if n > 6:
        axs.set_xlabel('time (s)')
    axs.tick_params(direction='in')

cb = plt.colorbar(c, cax=cbax, orientation='horizontal', extend='max', )

#cb.ax.set_xlabel('r$_e$ ($\mu$m)')
cb.ax.set_xlabel('Ice MMR (g kg$^{-1}$)')
plt.subplots_adjust(right = 0.95, top = 0.95, hspace = 0.23, wspace= 0.08, bottom = 0.15)
plt.savefig(filepath + '../figures/Hovmoller_M_time.png')
plt.show()

PSD_dict['CTRL']['M'][PSD_dict['CTRL']['M']<1e-9] = np.nan
PSD_dict['ICNCx2_8-9_2000']['M'][PSD_dict['ICNCx2_8-9_2000']['M']<1e-9] = np.nan
M_ctrl = np.nanmean(PSD_dict['CTRL']['M'], axis = 1)
M_pert = np.nanmean(PSD_dict['ICNCx2_8-9_2000']['M'], axis = 1)
PSD_dict['CTRL']['N'][PSD_dict['CTRL']['N']<100] = np.nan
PSD_dict['ICNCx2_8-9_2000']['N'][PSD_dict['ICNCx2_8-9_2000']['N']<100] = np.nan
N_ctrl = np.nanmean(PSD_dict['CTRL']['N'], axis = 1)
N_pert = np.nanmean(PSD_dict['ICNCx2_8-9_2000']['N'], axis = 1)

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

from matplotlib.gridspec import GridSpec

for t in range(M_ctrl.shape[0]):
#for t in [0]:
    fig = plt.figure(constrained_layout =True, figsize = (11,8))
    gs = fig.add_gridspec(nrows = 3,ncols=2)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[:2, 1])
    ax3 = fig.add_subplot(gs[2, :])
    cbax = fig.add_axes([0.8, 0.4, 0.02, 0.4])
    ax2.yaxis.set_visible(False)
   # b = ax1.contourf(range(60), alt[60:100], N_ctrl[t, :, 60:100].transpose(), cmap = 'Greys', vmin=0, vmax = 6e6, levels = np.linspace(0, 6.5e6, 21))
   # c = ax2.contourf(range(60), alt[60:100], N_pert[t, :, 60:100].transpose(), cmap = 'Greys', vmin=0, vmax = 6e6, levels = np.linspace(0, 6.5e6, 21))
    b = ax1.contourf(range(60), alt[60:100], M_ctrl[t, :, 60:100].transpose(), cmap = 'Greys', vmin=0, vmax = 5e-5, levels = np.linspace(0, 5e-5, 21))
    c = ax2.contourf(range(60), alt[60:100], M_pert[t, :, 60:100].transpose(), cmap = 'Greys', vmin=0, vmax = 5e-5, levels = np.linspace(0, 5e-5, 21))
    bas1 = ax1.plot(PSD_dict['CTRL']['clbas'][t].mean(axis=0), color='k', )
    top1 = ax1.plot(PSD_dict['CTRL']['cltop'][t].mean(axis=0), color='k', )
    bas2 = ax2.plot(PSD_dict['ICNCx2_8-9_2000']['clbas'][t].mean(axis=0), color='k', )
    top2 = ax2.plot(PSD_dict['ICNCx2_8-9_2000']['cltop'][t].mean(axis=0), color='k', )
    ax1.set_ylabel('Altitude (m)',  color='dimgrey', fontsize=18)
    ax2.text(0.3, 0.1, s='time = ' + str(int(time_srs_3d[t])) + ' s', transform=ax2.transAxes, fontsize=22,
             fontweight='bold', color='dimgrey')
    for axs in [ax1, ax2]:
        axs.tick_params(which='both', color = 'dimgrey', width = 2, direction = 'in', axis='both', labelsize=18, labelcolor='dimgrey', pad=10, size=5)
        plt.setp(axs.get_xticklabels(), visible=False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey')
        #axs.set_xlabel('Horizontal y distance', color='dimgrey', fontsize=18)
        axs.set_facecolor('#A9E6F8')
    #ax3.set_ylim(0, 2e6)
    ax3.set_ylim(0, 1e-5)
    ax3.set_xlim(0, time_srs_3d.max())
    #ax3.set_title('Mean in-cloud ice mass concentration (kg m$^{-3}$)',  color='dimgrey', fontsize=18)
    ax3.set_title('Mean in-cloud ice crystal number concentration (m$^{-3}$)',  color='dimgrey', pad= 15, fontsize=18)
    ax3.set_xlabel('Time (s)', color='dimgrey', fontsize=18)
    ax3.plot(time_srs_3d[:t], np.nanmean(M_ctrl[:t, :,60:100], axis = (1,2)), label = 'CTRL')
    ax3.plot(time_srs_3d[:t], np.nanmean(M_pert[:t, :,60:100], axis = (1,2)),  label = 'PERT')
    ax3.tick_params(which='both', color='dimgrey', width=2, direction='in', axis='both', labelsize=18, labelcolor='dimgrey',
                    pad=10, size=5)
    ax3.yaxis.major.formatter._useMathText = True
    ax3.yaxis.set_major_formatter(OOMFormatter(-5, "%1.1f"))
    #ax3.yaxis.set_major_formatter(OOMFormatter(6, "%1.1f"))
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,1), useOffset=None, useLocale=None, useMathText=True)
    lgd = ax3.legend(fontsize=18, markerscale=2, loc=1)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.setp(ax3.spines.values(), linewidth=2, color='dimgrey')
    cb=plt.colorbar(c, cax = cbax, ticks = [0, 2.5e-5, 5e-5])#, format = matplotlib.ticker.FuncFormatter(fmt))
    #cb=plt.colorbar(c, cax = cbax, ticks = [0, 2e6, 4e6, 6e6])
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.get_offset_text().set_text(r'$1 \times {10^{-5}}$')
    #cb.ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 1), useOffset=None, useLocale=None, useMathText=True)
    cb.ax.tick_params(which='both', axis='both', labelsize=18, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                      tick2On=False)
    cb.update_ticks()
    #cb.ax.yaxis.set_major_formatter(OOMFormatter(6, "%1.1f"))
    cb.ax.yaxis.set_major_formatter(OOMFormatter(-5, "%1.1f"))
    cb.outline.set_linewidth(2)
    #cb.ax.xaxis.set_ticks_position('top')
    #cb.ax.set_title('Ice number \nconc. (m$^{-3}$)', loc='left', pad=20, color='dimgrey', fontsize=18)
    cb.ax.set_title('Ice mass \nconc. (kg m$^{-3}$)', loc='left', pad=20, color='dimgrey', fontsize=18)
    ax1.set_title('background\nunperturbed cirrus', fontweight='bold', color='dimgrey', fontsize=24)
    ax2.set_title('\"aircraft\"\nperturbed cirrus',  fontweight='bold', color='dimgrey', fontsize=24)
    plt.subplots_adjust(left = 0.15, hspace = 0.4, wspace=0.05, right = 0.78)
    #plt.show()
    plt.savefig(filepath + '../figures/M_ctrl_v_pert_ts' + str(t) + '_y.png')

filenames = []
# create file name and append it to a list
for i in range(26):
    fn = filepath + f'../figures/M_ctrl_v_pert_ts{i}_y.png'
    filenames.append(fn)

import imageio
with imageio.get_writer(filepath + '../figures/M_CTRL_v_PERT_y_8-9_ICNC.gif', mode='I', fps=5) as writer:
    for f in filenames:
        image = imageio.imread(f)
        writer.append_data(image)
        
def contourplot(var):
    #fig, ax = plt.subplots(4,2, figsize = (6,13), sharex=True, sharey=True)
    fig, ax = plt.subplots(3, 2, figsize=(6, 10), sharex=True, sharey=True)
    ax = ax.flatten()
    cbax = fig.add_axes([0.3,0.08, 0.5, 0.02])
    ## Define colour levels
    vmin = -100
    vmax = 100
    levels = np.linspace(vmin, vmax, 41)
    #bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-5e-6, max_val = 5e-5, name='bwr_zero', var=((contour_dict['CTRL']['M']-contour_dict['CTRL']['M'])), start=0.15,stop=0.85)
    for axs, sim, n in zip(ax, [  'ICNCx1.25_8-9_2000', 'ICNCx1.25_9-10_2000', 'ICNCx1.5_8-9_2000', 'ICNCx1.5_9-10_2000', 'ICNCx2_8-9_2000', 'ICNCx2_9-10_2000', ], [1,2,3,4,5,6,7,8]):
    #for axs, sim, n in zip(ax, ['ICEx0.5_8-9_2000', 'ICEx0.5_9-10_2000','ICEx1.1_8-9_2000', 'ICEx1.1_9-10_2000', 'ICEx2_9-10_2000', 'ICEx2_8-9_2000','INPx0.5_8-9_2000', 'INPx0.5_9-10_2000'], [1, 2, 3, 4, 5, 6, 7, 8]):
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
            cb.set_ticklabels([r'$-5 \times 10^{-5}$', '', r'$-5 \times 10^{-7}$', '',r'-1 $\times$ 10$^{-8}$', '0', r'$1 \times 10^{-8}$', '', r'$5 \times 10^{-7}$','', r'$5 \times 10^{-5}$'])
            #cb.ax.set_xlabel('r$_e$ ($\mu$m)')
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
    plt.savefig(filepath + '../figures/Hovmoller_'+ var + '_ICNC_runs_2000s.png')
    plt.show()

contourplot('re')

for sim in PSD_dict.keys():
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlim(0, 400)
    ax.set_ylim(100, 1e8)
    ax.scatter(PSD_dict[sim]['re'][27,:,:, 77:85], PSD_dict[sim]['N'][27,:,:, 77:85], color = 'blue', label = '9-10 km', alpha=0.5)
    ax.scatter(PSD_dict[sim]['re'][27,:,:, 68:76], PSD_dict[sim]['N'][27,:,:, 68:76], color = 'orange', label = '8-9 km', alpha=0.5)
    ax.set_xlabel('Particle size ($\mu$m)')
    ax.vlines(np.ma.mean(PSD_dict[sim]['re'][27, :, :, 77:85]), ymin=100, ymax=1e8, colors='#0C2A57',linestyle='--', lw=2)
    ax.text(x=np.ma.mean(PSD_dict[sim]['re'][27, :, :, 77:85]) + 10, y=5e7,
            s=str(int(np.ma.mean(PSD_dict[sim]['re'][-1, :, :, 77:85]))) + '$\mu$m', color='blue', fontsize=12)
    ax.vlines(np.ma.mean(PSD_dict[sim]['re'][27, :, :, 68:76]), ymin=100, ymax=1e8, colors='#0C2A57', linestyle='--',lw=2)
    ax.text(x=np.ma.mean(PSD_dict[sim]['re'][27, :, :, 68:76]) + 10, y=5e7,
            s=str(int(np.ma.mean(PSD_dict[sim]['re'][27, :, :, 68:76]))) + '$\mu$m', color='orange', fontsize=12)
    ax.set_ylabel('Particle number density (m$^{-3}$)')
    plt.legend()
    plt.savefig(filepath + '../figures/PSD_alt_diff_' + sim + '.png')
    plt.show()

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

## Plot size-differentiated PSD
for t in range(PSD_dict['CTRL']['re'].shape[0]):
#for t in [0]:
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlim(0, 250)
    ax.set_ylim(100, 1e8)
    #ax.scatter(re_1d, N_1d)
    ax.scatter(PSD_dict['CTRL']['re'][t,:,:, 68:76], PSD_dict['CTRL']['N'][t,:,:, 77:85], color = 'blue', label = 'CTRL', alpha=0.5)
    ax.vlines(np.ma.median(PSD_dict['CTRL']['re'][t,:,:, 68:76]), ymin=100, ymax = 1e8, colors = '#0C2A57', linestyle = '--', lw=2)
    ax.text(x=np.ma.median(PSD_dict['CTRL']['re'][t,:,:, 68:76])+10, y=5e7, s=str(int(np.ma.median(PSD_dict['CTRL']['re'][t,:,:, 68:76]))) + '$\mu$m', color = '#0C2A57', fontsize = 12)
    ax.scatter(PSD_dict['CTRL']['re'][t,:,:, 77:85], PSD_dict['CTRL']['N'][t,:,:, 77:85], color = 'orange', label = 'ICNCx2 9-10 km', alpha=0.5)
    ax.vlines(np.ma.median(PSD_dict['CTRL']['re'][t,:,:, 77:85]), ymin=100, ymax = 1e8, colors = '#71340D', linestyle = '--', lw=2)
    ax.text(x=np.ma.median(PSD_dict['CTRL']['re'][t,:,:, 77:85])+10, y=1e7, s=str(int(np.ma.median(PSD_dict['CTRL']['re'][t,:,:, 77:85]))) + '$\mu$m', color = '#71340D', fontsize = 12)
    ax.set_xlabel('Particle effective radius ($\mu$m)')
    ax.set_ylabel('Particle number density (m$^{-3}$)')
    plt.legend()
    plt.savefig(filepath + '../figures/PSD_CTRL_hts_ts_' + str(t) + '.png')
    #plt.show()

# Create GIF
filenames = []
# create file name and append it to a list
for i in range(27):
    fn = filepath + f'../figures/PSD_CTRL_hts_ts_{i}.png'
    filenames.append(fn)

for i in range(6):
    filenames.append(filenames[-1]) # add freeze frame at end

with imageio.get_writer(filepath + '../figures/PSD_CTRL_hts_animated.gif', mode='I') as writer:
    for f in filenames:
        image = imageio.imread(f)
        writer.append_data(image)


## PDF
# Create 1D array of interest
for sim in PSD_dict.keys():
    for t in range(28):
        fig, ax = plt.subplots()
        re_1d_8_9 = PSD_dict[sim]['re'][t,:,:, 68:76].flatten() # 8-9 km
        re_1d_9_10 = PSD_dict[sim]['re'][t, :, :, 77:86].flatten()  # 8-9 km
        D_b= np.ma.masked_less_equal(re_1d_8_9, 0.) # mask values < 0
        D_t=np.ma.masked_less_equal(re_1d_9_10, 0.)
        val_b, bins_b, _ = ax.hist(D_b, 401, density=1, alpha=0.5, label = 'Cloud base', color = 'skyblue')
        mu_b, sigma_b = scipy.stats.norm.fit(D_b)
        best_fit_line_b = scipy.stats.norm.pdf(bins_b, mu_b, sigma_b)
        val_t, bins_t,  g = ax.hist(D_t, 401, density=1, alpha=0.5, label='Cloud top', color = 'crimson')
        mu_t, sigma_t = scipy.stats.norm.fit(D_t)
        best_fit_line_t = scipy.stats.norm.pdf(bins_t, mu_t, sigma_t)
        ax.plot(bins_b, best_fit_line_b, color = 'skyblue')
        ax.plot(bins_t, best_fit_line_t, color = 'crimson')
        ax.vlines(bins_t[np.where(val_t==val_t.max())], ymin = 0, ymax = 0.2, colors='crimson',  linestyle=':', )
        ax.vlines(bins_b[np.where(val_b == val_b.max())], ymin=0, ymax=0.2, colors='skyblue', linestyle='--', )
        ax.text(x=np.ma.median(PSD_dict[sim]['re'][t, :, :, 77:85]) + 2, y=0.18,
                s=str(int(np.ma.median(PSD_dict[sim]['re'][t, :, :, 77:85]))) + '$\mu$m', color='crimson', fontsize=12)
        ax.text(x=np.ma.median(PSD_dict[sim]['re'][t, :, :, 68:76]) + 2, y=0.16,
                s=str(int(np.ma.median(PSD_dict[sim]['re'][t, :, :, 68:76]))) + '$\mu$m', color='skyblue',fontsize=12)
        ax.set_xlim(-1,350)
        ax.legend()
        ax.set_ylim(0, 0.2)
        ax.set_xlabel('r$_{e}$ ($\mu$m)',)
        ax.set_ylabel('PDF')
        plt.savefig(filepath + '../figures/' + sim + '_PDF_re_two_heights_' + str(t) + '.png')
        plt.close('all')
    # Create GIF
    filenames = []
    # create file name and append it to a list
    for i in range(28):
        fn = filepath + '../figures/' + sim + f'_PDF_re_two_heights_{i}.png'
        filenames.append(fn)
    for i in range(6):
        filenames.append(filenames[-1]) # add freeze frame at end
    with imageio.get_writer(filepath + '../figures/PDF_re_' + sim + '_two_heights_animated.gif', mode='I') as writer:
        for f in filenames:
            image = imageio.imread(f)
            writer.append_data(image)


re_1d = np.mean(PSD_dict['CTRL']['re'][:,:,:, 77:86], axis=(1,2)).flatten() # 8-9 km
D= np.ma.masked_less_equal(re_1d, 0.) # mask values < 0
N_1d = np.mean(PSD_dict['CTRL']['N'][:,:,:, 77:86], axis=(1,2)).flatten()
N = np.ma.masked_less_equal(N_1d, 0.)
plt.hist2d(D,N, bins = (401,401))#, density=1, alpha=0.5)
plt.show()


Nvals, Nbins, _ = plt.hist(N, 401, density=1, alpha=0.5)
mu, sigma = scipy.stats.norm.fit(D)
mu2, sigma2 = scipy.stats.norm.fit(N)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
best_fit_line2 = scipy.stats.norm.pdf(Nbins, mu2, sigma2)
plt.plot(bins, best_fit_line)
pdf = scipy.stats.norm.pdf(x = bins, loc=mu, scale=sigma)
pdf2 = scipy.stats.norm.pdf(x = Nbins, loc=mu2, scale=sigma2)
plt.scatter(Nbins[1:-1], Nvals[1:], c = pdf2[1:-1])
plt.colorbar()
#plt.xlim(0,250)
plt.show()


# bin the data

from scipy.stats import gaussian_kde
x_bins = np.linspace(0,1000, 250)
#x_bins = [ 9., 10., 11., 12., 13., 14., 15., 16., 18., 20., 22., 24., 26., 28., 30., 33.,36., 39., 42., 45., 50., 55., 60., 65., 70., 80., 90., 100., 120., 140., 160., 180., 200., 220., 240., 260.,280.,300., 350., 400., 450., 500., 600., 700., 800., 900., 1000., ]
y_bins = np.linspace(1e3,1e6, 250)
re_1d = np.mean(PSD_dict['CTRL']['re'][:50,:,:,60:95], axis=(1,2)).flatten() # 8-9 km
D= np.ma.masked_less_equal(re_1d, 0.) # mask values < 0
N_1d = np.mean(PSD_dict['CTRL']['N'][:50,:,:,60:95], axis=(1,2)).flatten()
N = np.ma.masked_less_equal(N_1d, 0.)
x = np.digitize(D, bins = x_bins[1:])
y = np.digitize(N, bins = y_bins[1:])
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=50)
ax.set_xlim(0, 20)
ax.set_ylim(0,250)
plt.show()

X,Y = np.meshgrid(x, y)