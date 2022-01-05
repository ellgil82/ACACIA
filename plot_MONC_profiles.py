import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

filepath = '/storage/silver/acacia/ke923690/acacia/'
filename = 'Yang_test4b*_3600.nc'

# Load files
filename = input("Enter name of diagnostics file\n")
variable = input("Enter variable to plot\n")
timestep = input("Which timesteps do you need? (0-indexed)\n")
save = input("Do you want to save the figure?\n")

v = iris.load_cube(filepath + filename, variable)
z = pd.read_csv(filepath + 'Cirrus_vertical_grid.csv', header=0)
z = z['Model height'][:131].values

try:
    timestep = int(timestep)
    profile = v.data[timestep]
except ValueError:
    timestep = timestep
    profile = np.mean(v.data, axis=0)

unit_string = str(v.units)

fig, ax = plt.subplots(figsize=(8, 8))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
ax.set_ylim(0, max(z)/1000)
#[l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
ax.plot(profile, z/1000, color='k', linewidth=2.5)
ax.set_xlabel(variable + '\n(' + unit_string + ')', fontname='SegoeUI semibold', color='dimgrey', fontsize=28, labelpad=35)
ax.set_ylabel('Altitude \n(km)', rotation=0, fontname='SegoeUI semibold', fontsize=28, color='dimgrey', labelpad=80)
if profile.max() < 0.1: # If values are small, use scientific notation
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(24)
    ax.xaxis.get_offset_text().set_color('dimgrey')

plt.subplots_adjust(left = 0.35, right = 0.95, bottom = 0.25)
if save == 'yes' or save =='y' or save == 'Y':
    plt.savefig(filepath + variable + '_profile_at_timestep_' + str(timestep) + '.png')
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

df = pd.read_csv(filepath+'Initial_profiles_with_perturbations.csv')

for profile in df.keys():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
    ax.tick_params(axis='both', which='both', labelsize=24, color = 'dimgrey', length = 10, width = 3, labelcolor='dimgrey', pad=10)
    ax.set_ylim(0, max(df['z'])/1000)
    #[l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    ax.plot(df[profile], df['z']/1000, color='k', linewidth=2.5)
    ax.set_xlabel(profile + ' (' + unit_dict[profile] + ')', fontname='SegoeUI semibold', color='dimgrey', fontsize=28, labelpad=35)
    if profile == 'q':
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(24)
        ax.xaxis.get_offset_text().set_color('dimgrey')
    ax.set_ylabel('Altitude \n(km)', rotation=0, fontname='SegoeUI semibold', fontsize=28, color='dimgrey', labelpad=80)
   # if df[profile].max() < 0.1: # If values are small, use scientific notation
    plt.subplots_adjust(left = 0.35, right = 0.95, bottom = 0.25)
    plt.savefig(filepath + 'Initial_profile_' + profile + '.png')

plt.show()

filename = 'Yang_test4b_*d_14400.nc'

w = iris.load_cube(filepath+filename, 'w')
th = iris.load_cube(filepath+filename, 'th') #perturbation
th_init = iris.load_cube(filepath + filename, 'thinit')
theta = th+th_init
q = iris.load_cube(filepath+filename, 'q_vapour')
rh = iris.load_cube(filepath + filename, 'rh_mean')
iwp = iris.load_cube(filepath + filename, 'iwp')
ICNC = iris.load_cube(filepath+filename, 'q_ice_number')
rho = iris.load_cube(filepath+filename, 'rho')
ICNC_cm3 = (ICNC[-1].data.mean(axis=(0,1)) * rho[-1].data)/10e6
ice_mmr = iris.load_cube(filepath+filename, 'ice_mmr_mean')
sed = iris.load_cube(filepath+filename, 'psedi_mean')
dep = iris.load_cube(filepath+filename, 'pidep_mean')

z = rh.coord('zn').points