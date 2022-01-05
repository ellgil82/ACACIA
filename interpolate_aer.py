import pandas as pd
from scipy.interpolate import interp2d
import numpy as np

df = pd.read_csv('/storage/silver/acacia/ke923690/acacia/aerosol_profiles_RH.csv', header = 1, )

ht = pd.read_csv('/storage/silver/acacia/ke923690/acacia/Heights.csv', header = 0, )
ht = ht['Heights']

df_interp = pd.DataFrame()

for h in df.keys()[1:]:
    df_interp[h] = pd.Series(np.interp(ht, df['z'], df[h]))

df_interp.to_csv('/storage/silver/acacia/ke923690/acacia/aerosol_profiles_interpolated.csv')