import numpy as np
import netCDF4 as nc

filepath = '/storage/silver/acacia/ke923690/acacia/Yang_forcing/'
pecan_file = 'pecan60varanaPECANC1.c1.20150601.000000_reformed.nc'

fn = 'wsubs_time_dependent_forcing.nc'

ds = nc.Dataset(filepath + fn, 'w', format='NETCDF4')

time = ds.createDimension('time', None)
z = ds.createDimension('z', 191)

times = ds.createVariable('time', 'f4', ('time',))
times.units = 'seconds'
times.long_name = 'time'
times.standard_name = 'time'
zs = ds.createVariable('z', 'f4', ('z',))
zs.units ='m'
zs.standard_name = 'z'
zs.long_name = 'z'

value = ds.createVariable('wsubs', 'f4', ('time', 'z'))
value.units = 'm/s'
value.long_name = 'wsubs'
value.standard_name = 'wsubs'

wsubs = np.genfromtxt(filepath + 'gravity_wave_forcing.dat')
wsubs_profile = np.tile(wsubs[:,1], (191,1))
zs[:] = np.genfromtxt(filepath + 'Heights.dat')
times[:] = wsubs[:,0].astype(int)

for i, t in enumerate(times):
    value[i, :] = wsubs_profile[:,i]

ds.close()

## Create heating profile
fn = 'heating_profile_forcing.nc'

ds = nc.Dataset(filepath + fn, 'w', format='NETCDF4')

time = ds.createDimension('time', None)
z = ds.createDimension('z', 191)

times = ds.createVariable('time', 'f4', ('time',))
times.units = 'seconds'
times.long_name = 'time'
times.standard_name = 'time'

zs = ds.createVariable('z', 'f4', ('z',))
zs.units ='m'
zs.standard_name = 'z'
zs.long_name = 'z'

value = ds.createVariable('theta_tendency', 'f4', ('time', 'z'))
value.units = 'K/s'
value.long_name = 'theta_tendency'
value.standard_name = 'theta_tendency'

heating = np.loadtxt(filepath + 'heating_rate.dat')
heating = heating/(60*60*24) # Convert from K/day to K/s
heating_profile = np.array(np.split(np.array(np.ravel(heating)), 72))
ht = np.genfromtxt(filepath + 'Heights.dat')
zs[:] = ht
times[:] = np.arange(0, 21600, 300)

for i, t in enumerate(times):
    value[i, :] = heating_profile[i, :191]

ds.close()

## Create q forcing
fn = 'q_time_dependent_forcing.nc'

ds = nc.Dataset(filepath + fn, 'w', format='NETCDF4')

time = ds.createDimension('time', None)
z = ds.createDimension('z', 191)

times = ds.createVariable('time', 'f4', ('time',))
times.units = 'seconds'
times.long_name = 'time'
times.standard_name = 'time'

zs = ds.createVariable('z', 'f4', ('z',))
zs.units ='m'
zs.standard_name = 'z'
zs.long_name = 'z'

value = ds.createVariable('q_tendency', 'f4', ('time', 'z'))
value.units = 'kg/kg/s'
value.long_name = 'q_tendency'
value.standard_name = 'q_tendency'

q_force = np.loadtxt(filepath + 'q_forcing.csv', delimiter=',') # in kg/kg/s

ht = np.genfromtxt(filepath + 'Heights.dat')
zs[:] = ht
times[:] = np.arange(0, 21600, 300)

q_force_profile = np.tile(q_force[:, np.newaxis], 72)
q_force_profile = q_force_profile.transpose()

for i, t in enumerate(times):
    value[i, :] = q_force_profile[i, :191]

ds.close()

mean_heating_profile = heating_profile.mean(axis=0)


ds = nc.Dataset(filepath + 'pecan_edited.nc', 'w', format='NETCDF4')