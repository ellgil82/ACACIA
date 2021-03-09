import pandas as pd
import numpy as np

filepath = '/storage/silver/acacia/ke923690/acacia/Yang_forcing/'

heights = np.genfromtxt(filepath + 'Heights.dat')
pressure = np.genfromtxt(filepath + 'Pressure.dat')
water_vapour = np.genfromtxt(filepath + 'q.dat')
u_v = np.genfromtxt(filepath + 'u_v.dat')
theta = np.genfromtxt(filepath + 'Theta.dat')
heating_rate = np.genfromtxt(filepath + 'heating_profile_initial.dat')

df = pd.DataFrame()
df['Heights'] = pd.Series(heights)
df['Pressure'] = pd.Series(np.ravel(pressure))
df['Water vapour'] = pd.Series(np.ravel(water_vapour))
df['u/v'] = pd.Series(np.ravel(u_v))
df['theta'] = pd.Series(np.ravel(theta))

#create range of random numbers
rand = np.empty(shape=(191,))
for l in range(len(df['theta'].values)):
    rand[l] = np.random.rand()

df['q_pert'] = pd.Series(((df['Water vapour'].values + (rand - 0.5) * 0.1) * df['Water vapour'].values) - df['Water vapour'].values)

#create range of random numbers
rand = np.empty(shape=(191,))
for l in range(len(df['theta'].values)):
    rand[l] = np.random.rand()

df['theta_pert'] = pd.Series((df['theta'].values + (rand - 0.5) * 0.2) - df['theta'].values)
for p in ['theta_pert', 'q_pert']:
    df[p].values[(df['Heights'].values < 6000)] = 0
    df[p].values[(df['Heights'].values > 10500)] = 0

df.to_csv(filepath + 'Initial_profiles_with_perturbations.csv')
df['Heights'].to_csv(filepath + 'Heights.csv')
df['theta'].to_csv(filepath + 'theta.csv')
df['Pressure'].to_csv(filepath + 'pressure.csv')
df['u/v'].to_csv(filepath + 'u_v.csv')
df['Water vapour'].to_csv(filepath + 'q.csv')
df['q_pert'].to_csv(filepath + 'q_pert.csv')
df['theta_pert'].to_csv(filepath + 'theta_pert.csv')


