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
filename = 'Yang_GCSS_gw_forcing_28800.nc'

## Sources of water
proc_dict = {}
for name, v in zip(['cond', 'ice_nucl', 'homg_freezing', 'ice_dep', 'ice_sed', 'snow_sed', 'graupel_sed', 'ice_subm', 'snow_subm',
                    'graupel_subm', 'ice_melt', 'snow_melt', 'graupel_melt'], ['pcond_mean', 'pinuc_mean', 'phomc_mean',
                                                                               'pidep_mean', 'psedi_mean', 'pseds_mean',
                                                                               'psedg_mean', 'pisub_mean', 'pssub_mean',
                                                                               'pgsub_mean', 'pimlt_mean', 'psmlt_mean', 'pgmlt_mean']):
    #cubes = iris.load(filepath + filename, v)
    #proc_dict[name] = np.concatenate((cubes[0].data, cubes[1].data), axis=0)
    proc_dict[name] = iris.load_cube(filepath + filename, v)

prop_dict = {}
for name, v in zip(['ice_mmr', 'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean'], ['ice_mmr_mean',  'snow_mmr_mean', 'graupel_mmr_mean', 'vapour_mmr_mean']):
    prop_dict[name] = iris.load_cube(filepath + filename, v)

## Relevant met
met_dict = {}
for name, v in zip(['theta', 'w', 'rh'], ['theta_mean', 'w_wind_mean', 'rh_mean']):
    #cubes = iris.load(filepath + filename, v)
    #met_dict[name] = np.concatenate((cubes[0].data, cubes[1].data), axis=0)
    met_dict[name] = iris.load_cube(filepath + filename, v)

# Load model heights
model_hts = pd.read_csv(filepath + 'Cirrus_vertical_grid.csv')
model_hts = model_hts['Model height'][:116].values
model_hts = np.linspace(0,18000,191)

for t in range(proc_dict['ice_melt'].shape[0]):
    fig, ax = plt.subplots(figsize=(5, 6))
    ax2 = ax.twiny()
    color_list = ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal', 'brown', 'darkgrey', 'indigo', ]
    for c, i, in enumerate(proc_dict.keys()):
        ax.plot(proc_dict[i][t].data, model_hts/1000, label = i, color = color_list[c])
        ax2.plot(met_dict['w'][t].data, model_hts/1000, label = 'w', color = 'darkgrey', linestyle = '--')
    ax2.set_xlabel('w wind (m/s)')
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlim(-5e-8, 5e-8)
    ax2.set_xlim(-1.5e-3, 1.5e-3)
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('Process rate (kg/kg/s)')
    ax.legend(loc=3)
    #ax2.legend() sc
    ax.text(0., 1.1, s = 'timestep = '+ str(t), fontweight ='bold', transform = ax.transAxes)
    plt.subplots_adjust(left = 0.2)
    plt.savefig(filepath + 'figures/process_rates_timestep_perturbed_dz100m_GCSS_gravity_wave_' + str(t) + '.png')

#plt.show()
plt.close()

filenames = []
# create file name and append it to a list
for i in range(proc_dict['ice_melt'].shape[0]):
    fn = filepath + 'figures/' + f'process_rates_timestep_perturbed_dz100m_GCSS_{i}.png'
    filenames.append(fn)

with imageio.get_writer(filepath + 'figures/Animated_process_rates_perturbed_dz100m_GCSS_gravity_wave.gif', mode='I') as writer:
    for f in filenames:
        image = imageio.imread(f)
        writer.append_data(image)

# Calculate total water budget of cloud
ins = proc_dict['cond'] + proc_dict['ice_nucl'] + proc_dict['homg_freezing'] + proc_dict['ice_dep']
outs = proc_dict['ice_melt'] + proc_dict['graupel_melt'] + proc_dict['snow_melt'] + proc_dict['ice_subm'] + proc_dict['snow_subm'] + proc_dict['graupel_subm']
# Sink from individual layers - do we consider the 'cloud system', and if so, how do we define this?
movt = proc_dict['ice_sed'] + proc_dict['snow_sed'] + proc_dict['graupel_sed']


## Questions
# 1. is the budget for the entire cloud?
# 2. is it for specific layers?


fig, ax = plt.subplots(figsize=(5, 6))
for t, l in zip([4,10,16,19, 21, 24, 27, 33,  39, 43 ], ['60 mins', '120 mins', '180 mins', '210 mins', '240 mins', '270 mins', '300 mins', '360 mins', '420 mins', '470 mins']):
    color_list = ['red', 'orange', 'lightblue', 'purple', 'darkblue', 'magenta', 'green', 'lightgreen', 'pink', 'teal', 'brown', 'darkgrey', 'indigo', ]
    ax.plot(prop_dict['ice_mmr'][t][70:120].data, model_hts[70:120]/1000, label = l)#, color = color_list[t])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlim(1e-8, 1e-4)
    ax.set_xscale('log')
    ax.set_ylabel('Altitude (km)', rotation =90)
    ax.set_xlabel('Ice MMR (kg/kg)')
    ax.legend(loc=4)
    plt.subplots_adjust(left = 0.2)
    #plt.savefig(filepath + 'figures/ice_mmr_perturbed_dz100m.png')

plt.show()


# Define cloud layers
ts = 0
# find indices of array where CF == 1
cl_idx = np.where(ice_mmr[ts,:300].data>1e-8)[0]
cl_base = np.min(cl_idx)
cl_top = np.max(cl_idx)

water_budget = np.sum(ins[ts, cl_base:cl_top].data) - np.sum(outs[ts, cl_base:cl_top].data) - np.sum(movt[ts, cl_base].data)



no_F = iris.load_cube(filepath + 'Yang_casim_proc_pert_GCSS_28800.nc', 'theta_mean')
with_F = iris.load_cube(filepath + 'Yang_GCSS_gw_forcing_28800.nc', 'theta_mean')

plt.plot(np.mean(no_F.data, axis = 0), label = 'no forcing')
plt.plot(np.mean(with_F.data, axis = 0), label = 'with forcing')
plt.legend()
plt.show()



'''ice_mmr = iris.load_cube(filepath+filename, 'ice_mmr_mean')
snow_mmr = iris.load_cube(filepath+filename, 'snow_mmr_mean')
graupel_mmr = iris.load_cube(filepath+filename, 'graupel_mmr_mean')
dqi_mphys_mean = iris.load_cube(filepath+filename, 'dqi_mphys_mean') # is this the total change in ice concentration from mphys? Yes. tendencies.
dqv_mphys_mean = dqi_mphys_mean = iris.load_cube(filepath+filename, 'dqv_mphys_mean')
wv_mmr = iris.load_cube(filepath+filename, 'vapour_mmr_mean')
CF = iris.load_cube(filepath+filename, 'total_cloud_fraction')

# Condensation
proc_dict['cond'] = iris.load_cube(filepath + filename , 'pcond_mean') # source of cloud water and suspended aerosol particles
# heterogeneous ice nucleation rate (flux)
proc_dict['ice_nucl'] = iris.load_cube(filepath + filename , 'pinuc_mean')
# homogeneous ice nucleation rate (flux) <-- including freezing of aqueous aerosol droplets?
proc_dict['homg_freezing'] = iris.load_cube(filepath + filename , 'phomc_mean') # homogeneous freezing of cloud droplets -- assume negligible homog freezing of rain
# advection (?)
# Deposition
proc_dict['ice_dep'] = iris.load_cube(filepath + filename , 'pidep_mean')

## Sinks of water
# sedimentation -- assume negligible flux of cloud and rain particle sedimentation
proc_dict['ice_sed'] = iris.load_cube(filepath + filename , 'psedi_mean')
proc_dict['snow_sed'] = iris.load_cube(filepath + filename , 'pseds_mean')
proc_dict['graupel_sed'] = iris.load_cube(filepath + filename , 'psedg_mean')
# sublimation
proc_dict['ice_subm'] = iris.load_cube(filepath + filename , 'pisub_mean')
proc_dict['snow_subm'] = iris.load_cube(filepath + filename , 'pssub_mean')
proc_dict['graupel_subm'] = iris.load_cube(filepath + filename , 'pgsub_mean')
# Melting
proc_dict['ice_melt'] = iris.load_cube(filepath + filename , 'pimlt_mean')
proc_dict['snow_melt'] = iris.load_cube(filepath + filename , 'psmlt_mean')
proc_dict['graupel_melt'] = iris.load_cube(filepath + filename , 'pgmlt_mean')


# Precipitation

## Other processes
# Aggregation
# Accretion
# Hallet-Mossop (probably not important in expected temperature range)'''