## Xarray method
import xarray as xr

# Import from checkpoint
da = xr.open_dataset('checkpoint_files/Yang_2012_dump_1500.nc')
#da = xr.open_dataset('/storage/silver/acacia/ke923690/acacia/Yang_2012_dump_1400.nc')

# Multiply INP at altitudes 9-10 km by 10
#for n in ['q_coarse_dust_number', 'q_coarse_dust_mass', 'zq_coarse_dust_number', 'zq_coarse_dust_mass', 'olzqbar_coarse_dust_number', 'olzqbar_coarse_dust_mass', 'olqbar_coarse_dust_number', 'olqbar_coarse_dust_mass']:
#	try:
#		da[n][:, :, :, 77:85] = da[n][:, :, :, 77:85] * 0.5
#	except IndexError: 
#		da[n][:, 77:85] = da[n][:,77:85] * 0.5

# Multiply INP at altitudes 8-9 km by 10
#for n in ['q_coarse_dust_number', 'q_coarse_dust_mass', 'zq_coarse_dust_number', 'zq_coarse_dust_mass', 'olzqbar_coarse_dust_number', 'olzqbar_coarse_dust_mass', 'olqbar_coarse_dust_number', 'olqbar_coarse_dust_mass']:
#	try:
#		#da[n][:, :, :, 68:76] = da[n][:, :, :, 68:76] * 10 # uniform
#		da[n][:,  :, 25:34, 68:76] = da[n][:,:, 25:34, 68:76] * 10 # linear contrail, 1 km width north-south through domain (zonal)
#	except IndexError:
#		da[n][:, 68:76] = da[n][:,68:76] * 10

# Multiply ice at altitudes 8-9 km by 2
#for n in ['q_ice_number', 'q_ice_mass', 'zq_ice_number', 'zq_ice_mass', 'olzqbar_ice_number', 'olzqbar_ice_mass', 'olqbar_ice_number', 'olqbar_ice_mass']:#	da[n] = da[n] * 1.1#
#	try:
#		da[n][:, :, :, 68:76] = da[n][:, :, :, 68:76] * 2
#	except IndexError:
#		da[n][:, 68:76] = da[n][:, 68:76] * 2

# Multiply ice at altitudes 9-10 km by 2
#for n in ['q_ice_number', 'q_ice_mass', 'zq_ice_number', 'zq_ice_mass', 'olzqbar_ice_number', 'olzqbar_ice_mass', 'olqbar_ice_number', 'olqbar_ice_mass']:#	da[n] = da[n] * 1.1
#	try:
#		da[n][:, :, :, 77:85] = da[n][:, :, :, 77:85] * 1.1
#	except IndexError:
#		da[n][:, 77:85] = da[n][:,77:85] * 1.1

# Multiply ice NUMBER at altitudes 8-9 km by 2
for n in ['q_ice_number', 'zq_ice_number',  'olzqbar_ice_number',  'olqbar_ice_number', ]:#	da[n] = da[n] * 1.1
	try:
		da[n][:,  :,  :, 68:76] = da[n][:, :, :,  68:76] * 1.1 # N-S linear contrail 25:34
	except IndexError:
		da[n][:, 68:76] = da[n][:, 68:76] * 1.1

# Multiply ice NUMBER at altitudes 9-10 km by 2
#for n in ['q_ice_number', 'zq_ice_number',  'olzqbar_ice_number',  'olqbar_ice_number', ]:#
#	try:
#		da[n][:, :, :, 77:85] = da[n][:, :, :, 77:85] * 1.1 # N-S linear contrail
#	except IndexError:
#		da[n][:, 77:85] = da[n][:,77:85] * 1.1

# Save
da.to_netcdf('checkpoint_files/Yang_2012_dump_updated_1500.nc')
#da.to_netcdf('/storage/silver/acacia/ke923690/acacia/Yang_2012_dump_updated_1400.nc')