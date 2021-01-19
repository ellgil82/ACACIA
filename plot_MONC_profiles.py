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

v = iris.load_cube(filepath + filename, variable)
z = pd.read_csv(filepath + 'Cirrus_vertical_grid.csv', header=0)

try:
    timestep = int(timestep)
    profile = v.data[timestep]
except ValueError:
    timestep = timestep
    profile = np.mean(v.data, axis=0)

unit_string = str(g.units)

fig, ax = plt.subplots(figsize=(8, 8))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
ax.set_ylim(0, max(z['Model height'])/1000)
[l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
ax.plot(profile, z['Model height']/1000, color='k', linewidth=2.5)
ax.set_xlabel(variable + '\n(' + unit_string + ')', fontname='SegoeUI semibold', color='dimgrey', fontsize=28, labelpad=35)
ax.set_ylabel('Altitude \n(km)', rotation=0, fontname='SegoeUI semibold', fontsize=28, color='dimgrey', labelpad=80)
if profile.max() < 0.1: # If values are small, use scientific notation
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(24)
    ax.xaxis.get_offset_text().set_color('dimgrey')
plt.subplots_adjust(left = 0.35, right = 0.95, bottom = 0.25)
plt.show()
