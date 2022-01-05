import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
import imageio
from math import gamma, pi

filepath = '/storage/silver/acacia/ke923690/acacia/new_runs/'

# What do we want in the table?
# Mean T, w, in-cloud IWC, ICNC, RHice, re, cloud lifetime?

p1 = 3.0 # number
p2 = 0.0 # mass
mu = 2.5 # shape parameter (ice)
c_x = pi*997.0/6.0
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

tab_dict = {}
for fn, nm in zip([ '*22a*', '*22b*', '*22c*', '*22d*', '*22e*', '*22f*', '*22g*','*22h*', '*22i*','*22j*', '*22k*', '*22l*', '*22m*'],
                  [ 'CTRL', 'ICNCx2_8-9', 'ICNCx2_9-10', 'no_qF', 'ICNCx1.5_8-9', 'ICNCx1.5_9-10','ICNCx1.25_8-9','ICNCx1.25_9-10','ICNCx1.1_8-9','ICNCx1.1_9-10', 'ICNCx2_full', 'ICEx2_full', 'MASSx2_full' ]):
    tab_dict[nm] = {}
    try:
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'clbas')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'clbas')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'clbas')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'clbas')
            tab_dict[nm]['clbas'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            tab_dict[nm]['clbas'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        #tab_dict[nm]['clbas'][tab_dict[nm]['clbas'] == 0.] = np.nan
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'cltop')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'cltop')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'cltop')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'cltop')
            tab_dict[nm]['cltop'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
            tab_dict[nm]['time_srs_lowres'] = np.concatenate(
                (f1.coord('time_series_300_300').points, f2.coord('time_series_300_300').points,
                 f3.coord('time_series_300_300').points, f4.coord('time_series_300_300').points), axis=0)
        except:
            print('Not full length')
            tab_dict[nm]['cltop'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
            tab_dict[nm]['time_srs_lowres'] = np.concatenate(
                (f1.coord('time_series_300_300').points, f2.coord('time_series_300_300').points,
                 f3.coord('time_series_300_300').points), axis=0)
        #tab_dict[nm]['cltop'][tab_dict[nm]['cltop'] == 0.] = np.nan
#    except:
#        print(':\'(')
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'rho')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'rho')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'rho')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'rho')
            tab_dict[nm]['rho'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            tab_dict[nm]['rho'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'ice_mmr_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'ice_mmr_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'ice_mmr_mean')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'ice_mmr_mean')
            tab_dict[nm]['M'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            tab_dict[nm]['M'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        tab_dict[nm]['M'][tab_dict[nm]['M'] == 0.] = np.nan
        tab_dict[nm]['M'] = tab_dict[nm]['M'] * tab_dict[nm]['rho'] # convert to kg m-3
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'ice_nc_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'ice_nc_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'ice_nc_mean')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'ice_nc_mean')
            tab_dict[nm]['N'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            tab_dict[nm]['N'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        tab_dict[nm]['N'][tab_dict[nm]['N'] == 0.] = np.nan
        tab_dict[nm]['N'] = tab_dict[nm]['N'] * tab_dict[nm]['rho'] # convert to kg m-3
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'temperature')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'temperature')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'temperature')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'temperature')
            tab_dict[nm]['T'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            tab_dict[nm]['T'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        tab_dict[nm]['T'] = tab_dict[nm]['T'].mean(axis=(1,2))
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'w')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'w')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'w')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'w')
            tab_dict[nm]['w'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            tab_dict[nm]['w'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        #tab_dict[nm]['w'][tab_dict[nm]['w'] == 0.] = np.nan
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'rhi_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'rhi_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'rhi_mean')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'rhi_mean')
            tab_dict[nm]['RHi'] = (np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0) * 100. ) + 100
        except:
            tab_dict[nm]['RHi'] = (np.concatenate((f1.data, f2.data, f3.data), axis=0) * 100. ) + 100
        tab_dict[nm]['RHi'][tab_dict[nm]['RHi'] == 100] = np.nan # supersaturation
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'pidep_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'pidep_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'pidep_mean')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'pidep_mean')
            tab_dict[nm]['dep'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            tab_dict[nm]['dep'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        tab_dict[nm]['dep'][tab_dict[nm]['dep'] == 0.] = np.nan
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'psedi_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'psedi_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'psedi_mean')
        try:
            f4 = iris.load_cube(filepath + fn + '14400.nc', 'psedi_mean')
            tab_dict[nm]['sed'] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
        except:
            print('Not full length')
            tab_dict[nm]['sed'] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
        tab_dict[nm]['sed'][tab_dict[nm]['sed'] == 0.] = np.nan
        try:
            tab_dict[nm]['re'], tab_dict[nm]['n0'],tab_dict[nm]['lam'] = calc_effective_radius(tab_dict[nm]['N'], tab_dict[nm]['M'], p1, p2, mu, c_x)
            tab_dict[nm]['re'][tab_dict[nm]['re'] > 1000] = 0.
            tab_dict[nm]['re'][tab_dict[nm]['re'] < 0] = 0.
        except:
            print('Oops - forgot my keys')
    except:
        print('Goodness, how embarrassing. I can\'t seem to do that')
    try:
        tab_dict[nm]['time_srs'] = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points, f3.coord('time_series_10_30').points, f4.coord('time_series_10_30').points), axis=0)
    except iris.exceptions.CoordinateNotFoundError:
        tab_dict[nm]['time_srs'] = np.concatenate((f1.coord('time_series_100_100').points, f2.coord('time_series_100_100').points,
                        f3.coord('time_series_100_100').points, f4.coord('time_series_100_100').points), axis=0)

#nm = 'CTRL'
#cl_thickness = tab_dict[nm]['cltop']-tab_dict[nm]['clbas']
#dissipation_idx = np.min(np.where(np.nanmin(cl_thickness, axis = (1,2)) == 0.))
#lifetime = 14400 - tab_dict[nm]['time_srs'][dissipation_idx]


def make_table():
    df = pd.DataFrame()
    for nm in tab_dict.keys():
        mns = []
        for k in ['T', 'w', 'RHi', 'M', 'N', 're']:
            if k == 'w':
                w_p = np.copy(tab_dict[nm]['w'][4:, :, :, 60:95])  # discard spin-up, in-cloud only
                w_p[w_p <= 0] = np.nan
                w_mn = np.nanmean(w_p, axis = (1,2))
                #w_mn[(tab_dict[nm]['M']/tab_dict[nm]['rho'])< 1.e-6] = np.nan # average only in-cloud values
                mns.append(np.nanmean(w_mn))
            elif k == 'T':
                T = np.nanmean(tab_dict[nm][k][4:, 60:95])
                #T[(tab_dict[nm]['M'] / tab_dict[nm]['rho']) < 1.e-6] = np.nan
                mns.append(T)
            else:
                v = tab_dict[nm][k][56:, 60:95]
                v[(tab_dict[nm]['M'] / tab_dict[nm]['rho'])[56:, 60:95] < 1.e-6] = np.nan # in-cloud values only
                mns.append(np.nanmean(v))
        cl_thickness = tab_dict[nm]['cltop'] - tab_dict[nm]['clbas']
        try:
            #dissipation_idx = np.min(np.where(np.nanmin(cl_thickness, axis=(1, 2)) == 0.))
            ice_mmr = np.nanmean((tab_dict[nm]['M'] / tab_dict[nm]['rho'])[56:, 60:95], axis=1)
            ice_mmr = np.nan_to_num(ice_mmr)
            dissipation_idx = np.min(np.where(ice_mmr <= 1.e-6))
            lifetime = tab_dict[nm]['time_srs'][56:][dissipation_idx]
        except:
            lifetime = 14400
        mns.append(lifetime)
        df[nm] = pd.Series(mns, index = ['T', 'w', 'RHi', 'M', 'N', 're', 'lifetime'])
    return df

make_table()
df = make_table()
df.to_csv(filepath + 'table_of_values.csv')

nm = 'CTRL'
dissipation_idx = np.min(np.where(np.nanmean((tab_dict[nm]['M']/tab_dict[nm]['rho'])[56:, 60:95], axis=1) <= 1.e-7))
c = plt.contourf((tab_dict[nm]['M']/tab_dict[nm]['rho'])[56:, 60:95].transpose())
plt.contour((tab_dict[nm]['M']/tab_dict[nm]['rho'])[56:, 60:95].transpose(), levels = [1.e-7], colors = 'k')
plt.colorbar(c)
plt.show()