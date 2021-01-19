import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

filepath = '/storage/silver/acacia/ke923690/'

# Load files
filename = input("Enter name of diagnostics file\n")
variable = input("Enter variable to plot\n")
timestep = input("Which timesteps do you need? (0-indexed)\n")

## Sources of water
# Condensation
cond = iris.load_cube(filepath + filename + 'l_pcond') # source of cloud water and suspended aerosol particles
# heterogeneous ice nucleation rate (flux)
hetg_nucl = iris.load_cube(filepath + filename + 'l_pinuc')
# homogeneous ice nucleation rate (flux) <-- including freezing of aqueous aerosol droplets?
homg_nucl = iris.load_cube(filepath + filename + 'phomc') # homogeneous freezing of cloud droplets -- assume negligible homog freezing of rain
# advection (?)
# Deposition
ice_dep = iris.load_cube(filepath + filename + 'l_pidep')

## Sinks of water
# sedimentation -- assume negligible flux of cloud and rain particle sedimentation
ice_sed = iris.load_cube(filepath + filename + 'l_psedi')
snow_sed = iris.load_cube(filepath + filename + 'l_pseds')
graupel_sed = iris.load_cube(filepath + filename + 'l_psedg')
# sublimation
ice_subm = iris.load_cube(filepath + filename + 'l_pisub')
snow_subm = iris.load_cube(filepath + filename + 'l_pssub')
graupel_subm = iris.load_cube(filepath + filename + 'l_pgsub')
# Melting
ice_melt = iris.load_cube(filepath + filename + 'l_pimlt')
snow_melt = iris.load_cube(filepath + filename + 'l_psmlt')
graupel_melt = iris.load_cube(filepath + filename + 'l_pgmlt')

# Precipitation

## Other processes
# Aggregation
# Accretion
# Hallet-Mossop (probably not important in expected temperature range)

## Questions
# 1. is the budget for the entire cloud?
# 2. is it for specific layers?