#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:24:07 2017

@author: eereh
"""

import iris
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
##import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import netCDF4
import sys
#import rachel_dict as ra
import glob
import os
import matplotlib.gridspec as gridspec
import matplotlib 
from matplotlib.ticker import FormatStrFormatter
from math import gamma, pi

#sys.path.append('/home/d04/rahaw/monc_src/r7181_new_branch_for_number_diags/scripts/rachel_modules/')

#import rachel_lists as rl
#import rachel_dict as ra
#import colormaps as cmaps
#import UKCA_lib as ukl


def mean_in_cloud(variable,cloud_mass_total,threshold, height_axis_no):
  variable[cloud_mass_total<threshold]=np.nan
  mean = np.nanmean(variable, axis=height_axis_no)
  return mean


def calc_effective_radius(N,M,p1,p2,mu,c_x):
    m1 = M/c_x
    m2 = N
    j1 = 1.0/(p1-p2)
    lam = ((gamma(1.0+mu+p1)/gamma(1.0+mu+p2))*(m2/m1))**(j1)
    Moment2 = (N/(lam**2))*((gamma(1+mu+2))/gamma(1+mu))
    Moment3 = (N/(lam**3))*((gamma(1+mu+3))/gamma(1+mu))
    effective_radius = Moment3/Moment2
    return effective_radius

###ICE CRYSTALS, SNOW, GRAUPEL
p1l = [3.0,3.0,3.0]
p2l = [0.0,0.0,0.0]
mul = [0.0,2.5,2.5]
c_xl = [pi*200.0/6.0,pi*100.0/6.0,pi*250.0/6.0]

path = '/projects/iced/rahaw/MONC_EMULATOR/EM_store/'


monc_dir_list =[ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

pathnc = '/home/d04/rahaw/monc_src/r7351_new_branch__2for_one_off_and_emulation/'

def subset_to_in_cloud_rain(monc_data_path,variable,hydro_index):
      #ax = axes[l]
      numbers = np.arange(3600,12900,300)
      array = []
      for f in range(0,len(numbers)):
        if numbers[f]<1000:
          rad = monc_data_path+'random_fluxes_'+str(numbers[f])+'.0.nc'
          time_name = '00'+str(numbers[f])
        elif numbers[f]<10000:
          rad = monc_data_path+'random_fluxes_'+str(numbers[f])+'.0.nc'
          time_name = '0'+str(numbers[f])
        else:
          rad = monc_data_path+'random_fluxes_'+str(numbers[f])+'.0.nc'
          time_name = str(numbers[f])
        print(time_name)
        ncd = netCDF4.Dataset(rad)
        clthresh = 1e-6
        #icm = ncd.variables['q_ice_mass'][0,:,:,:]
        #sm = ncd.variables['q_snow_mass'][0,:,:,:]
        #gm = ncd.variables['q_graupel_mass'][0,:,:,:]
        #cm = ncd.variables['q_cloud_liquid_mass'][0,:,:,:]
        #rm = ncd.variables['q_rain_mass'][0,:,:,:]
        rho = ncd.variables['rho'][0,:]
        data_no = ncd.variables[variable+'_number'][0,:,:,:]*rho	
        data_mass = ncd.variables[variable+'_mass'][0,:,:,:]*rho
        cloud_threshold = 1e-6
        icm = ncd.variables['q_ice_mass'][0,:,:,:]
        sm = ncd.variables['q_snow_mass'][0,:,:,:]
        gm = ncd.variables['q_graupel_mass'][0,:,:,:]
        cm = ncd.variables['q_cloud_liquid_mass'][0,:,:,:]
        rm = ncd.variables['q_rain_mass'][0,:,:,:]
        cloud_mass = icm+sm+gm+cm+rm
        data_no[cloud_mass<cloud_threshold]=0#np.nan
        data_mass[cloud_mass<cloud_threshold]=0#np.nan
        data_no[data_no==0]=np.nan
        data_mass[data_no==0]=np.nan
        data_no = np.nanmean(data_no,axis=(0,1))
        data_mass = np.nanmean(data_mass,axis=(0,1))
        data = calc_effective_radius(data_no,data_mass,p1l[hydro_index],p2l[hydro_index],mul[hydro_index],c_xl[hydro_index])
        #data[data_no==0]=np.nan
        data = data*1e6 #convert to um
        mean = data
        #data[data_no==0]=np.nan
        #data[cloud_mass_rain>clthresh]=np.nan
        #data[clts!=5]=np.nan
        #data = np.prod((data,rho),axis=2)
        #mean = np.nanmean(data,axis=(0,1))
        #mean = mean*rho
        print(mean)
        array.append(mean)
        #print(array)
      final_np = np.asarray(array)
      final_dat = np.nanmean(final_np, axis=0)
      print('created variable')
      return final_np, final_dat


for i in range(0,len(monc_dir_list)):
    if monc_dir_list[i]=='none':
       continue
    print(monc_dir_list[i])
    monc_data_path=path+monc_dir_list[i]+'/diagnostic_files/'
    print(monc_data_path)
    ncfile1 = pathnc+'ncfiles/'+monc_dir_list[i]+'_effective_radius_in_cloud_profiles.nc'
    outfile=ncfile1
    exists = os.path.isfile(outfile)
    if exists:
        continue
    EM_name = monc_dir_list[i]
    #cloud_type_file = pathnc+'ncfiles/'+EM_name+'_low_mid_high_cloud_categorisation_file.nc'
    #cltf = netCDF4.Dataset(cloud_type_file)
    #rainnt, rainn = subset_to_in_cloud_rain(monc_data_path,'q_rain_number')
    #cloudnt,cloudn = subset_to_in_cloud_rain(monc_data_path,'q_cloud_liquid_number')
    icent,icen = subset_to_in_cloud_rain(monc_data_path,'q_ice',0)
    graupelnt,graupeln = subset_to_in_cloud_rain(monc_data_path,'q_graupel',2)
    snownt,snown = subset_to_in_cloud_rain(monc_data_path,'q_snow',1)
    #rainmt,rainm = subset_to_in_cloud_rain(monc_data_path,'q_rain_mass')
    #cloudmt,cloudm = subset_to_in_cloud_rain(monc_data_path,'q_cloud_liquid_mass')
    #icemt,icem = subset_to_in_cloud_rain(monc_data_path,'q_ice_mass')
    #graupelmt,graupelm = subset_to_in_cloud_rain(monc_data_path,'q_graupel_mass')
    #snowmt,snowm = subset_to_in_cloud_rain(monc_data_path,'q_snow_mass')
    names = ['ice','graupel','snow']#['rain','cloud','ice','graupel','snow']
    variables_number_time =[icent,graupelnt,snownt]# [rainnt,cloudnt,icent,graupelnt,snownt]
    variables_number = [icen,graupeln,snown]# [rainn,cloudn,icen,graupeln,snown]
    #variables_mass_time =[icemt,graupelmt,snowmt]#[rainmt,cloudmt,icemt,graupelmt,snowmt]
    #variables_mass =[icem,graupelm,snowm]# [rainm,cloudm,icem,graupelm,snowm]
    minutes = np.arange(3600,12900,300)
    mins = minutes/60
    ncfile1 = netCDF4.Dataset(ncfile1,mode='w',format='NETCDF4_CLASSIC')
    time = ncfile1.createDimension('time',len(mins))
    t_out = ncfile1.createVariable('Time', np.float64, ('time'))
    t_out.units = 'minutes'
    t_out[:] = mins
    sing_file = monc_data_path +'random_fluxes_10800.0.nc'
    ncd2 = netCDF4.Dataset(sing_file)
    z= ncd2.variables['zn'][:]
    z=z/1000
    height = ncfile1.createDimension('height',None)
    z_out = ncfile1.createVariable('height', np.float64, ('height'))
    z_out.units = 'km'
    z_out[:] = z
    for n in range(0,3):
        print(names[n])
        name = names[n]+'_effective_rad_time'
        var = ncfile1.createVariable(name, np.float32, ('time','height'))
        var[:,:] = variables_number_time[n]
        print(name)
        name = names[n]+'_effective_rad'
        var = ncfile1.createVariable(name, np.float32, ('height'))
        var[:] = variables_number[n]
        print(name)
        print('vars made')
    ncfile1.close()





#no_rain_rain, nrr = subset_to_in_cloud_no_rain(monc_data_path,'q_rain_number')

