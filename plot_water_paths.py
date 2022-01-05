import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

filepath = '/storage/silver/acacia/ke923690/acacia/new_runs/'

iwp_dict = {}
for fn, nm in zip(['*22a*', '*22b*', '*22c*', '*22d*', '*22e*', '*22f*', '*22g*', '*22h*', '*22i*', '*22j*', '*22k*', '*22l*', '*22m*'],['CTRL', 'ICNCx2_8-9', 'ICNCx2_9-10', 'no_qF', 'ICNCx1.5_8-9', 'ICNCx1.5_9-10','ICNCx1.25_8-9', 'ICNCx1.25_9-10','ICNCx1.1_8-9', 'ICNCx1.1_9-10', 'ICNCx2_full', 'ICEx2_full', 'MASSx2_full' ]):
#for fn, nm in zip(['*17_CTRL*', '*18a*', '*18b*', '*20k*', '*20l*','*20i*', '*20j*','*20m*', '*20n*', '*14m*?d_', '*14n*?d_', '*14o*?d_', '*14p*?d_',  '*14r*', '*14s*',  ],
#                  ['CTRL', 'ICNCx2_8-9', 'ICNCx2_9-10','ICNCx1.5_8-9', 'ICNCx1.5_9-10', 'ICNCx1.25_8-9', 'ICNCx1.25_9-10',  'ICNCx1.1_8-9', 'ICNCx1.1_9-10', 'ICEx1.1_8-9', 'ICEx1.1_9-10', 'ICEx2_8-9', 'ICEx2_9-10', 'ICEx0.5_8-9', 'ICEx0.5_9-10',  ]):
#for fn, nm in zip(['*17_CTRL*',  '*20o*','*20a*', '*20b*', '*20c*','*20d*','*20g*', '*20h*', '*20e*','*20f*', '*22a*', '*22b*', '*22c*', '*22d*',  ],
#                  ['CTRL', 'q150%','q95%', 'q90%', 'q85%', 'q80%', 'fast_uv_q125%', 'fast_uv_q110%', 'fast_uv_q100%', 'fast_uv_q75%', 'slow_uv', 'slow_uv_ICNCx2_8-9','slow_uv_ICNCx2_9-10', 'slow_uv_no_qF'  ]):
    f1 = iris.load_cube(filepath + fn + '3600.nc', 'IWP_mean')
    f2 = iris.load_cube(filepath + fn + '7200.nc', 'IWP_mean')
    f3 = iris.load_cube(filepath + fn + '10800.nc', 'IWP_mean')
    try:
        f4 = iris.load_cube(filepath + fn + '14400.nc', 'IWP_mean')
        iwp_dict[nm] = np.concatenate((f1.data, f2.data, f3.data, f4.data), axis=0)
    except:
        iwp_dict[nm] = np.concatenate((f1.data, f2.data, f3.data), axis=0)
    if nm == 'CTRL':
            time_srs_ctrl = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points, f3.coord('time_series_10_30').points), axis=0)
    else:
        try:
            time_srs_hires = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points,f3.coord('time_series_10_30').points,f4.coord('time_series_10_30').points), axis = 0) #, iwp4.coord('time_series_100_100').points
        except:
            time_srs_hires = np.concatenate((f1.coord('time_series_10_30').points, f2.coord('time_series_10_30').points,
                                             f3.coord('time_series_10_30').points), axis=0)  # , iwp4.coord('time_series_100_100').points

#except:
    #    time_srs = np.concatenate((f1.coord('time_series_100_100').points, f2.coord('time_series_100_100').points,f3.coord('time_series_100_100').points), axis = 0) #, iwp4.coord('time_series_100_100').points


# Load files

fig, ax = plt.subplots(figsize=(8, 5))
#for nm, col in zip(['CTRL', 'ICEx2_8-9', 'ICEx2_9-10', 'ICEx1.1_8-9', 'ICEx1.1_9-10', 'ICEx0.5_8-9',
#                    'ICEx0.5_9-10', 'ICNCx2_8-9', 'ICNCx2_9-10', 'ICNCx1.5_8-9', 'ICNCx1.5_9-10', 'ICNCx1.25_8-9', 'ICNCx1.25_9-10',
#                    'ICNCx1.1_8-9', 'ICNCx1.1_9-10', ], ['#222222','#013D21', '#086D3D', '#0D844C','#1DA766', '#32C580', '#35EB6B',
#                                                         '#6E3685', '#8E44AD', '#A569BD','#9B59B6', '#BB8FCE', '#AF7AC5', '#DA78E4', '#E8DAEF']):
#for nm in [ '17_CTRL',  'ICEx2_8-9_2000', 'ICEx2_9-10_2000',  'ICEx2_8-9_4000', 'ICEx2_9-10_4000','ICEx0.5_8-9_2000', 'ICEx0.5_9-10_2000',  'ICEx0.5_8-9_4000', 'ICEx0.5_9-10_4000', 'ICNCx2_8-9_2000', 'ICNCx2_9-10_2000','contrail-EW_9-10', 'contrail-NS_8-9']:
#for nm, col in zip(['CTRL',  'ICEx2_9-10',  'ICEx1.1_9-10',  'ICEx0.5_9-10', 'ICEx2_8-9','ICEx1.1_8-9','ICEx0.5_8-9',  'ICNCx2_9-10','ICNCx1.5_9-10','ICNCx1.25_9-10','ICNCx1.1_9-10','ICNCx2_8-9','ICNCx1.5_8-9','ICNCx1.25_8-9','ICNCx1.1_8-9',  'CTRL'],
#                   ['w', '#FF6347', '#FF8C00', '#F4A460',  '#DC143C', '#FF69B4', '#FFC0CB', '#006400', '#20B2AA', '#9ACD32',  '#00FF00', '#00008B', '#4169E1', '#1E90FF',  '#222222']):
for nm, col in zip(iwp_dict.keys(), ['k', '#20B2AA', '#00008B', '#F4A460', '#1E90FF', '#FFC0CB','#87CEEB','#9B59B6', '#BB8FCE', '#AF7AC5', '#87CEEB', '#DA78E4', '#E8DAEF'] ):
    if nm == 'ICEx2_full' or nm == 'ICNCx2_full' or nm == 'MASSx2_full':
        ax.plot(time_srs_hires, iwp_dict[nm], lw=3, ls='--', label=nm)
    #if nm == 'CTRL':
    #    ax.plot(time_srs_hires, iwp_dict[nm], lw=2, label=nm)
    else:
        #try:
        #ax.plot(time_srs_hires[150:], iwp_dict[nm][150:], lw=2, label = nm)
        ax.plot(time_srs_hires, iwp_dict[nm], lw=2, label=nm)
        print(nm)
        #except:
            #ax.plot(np.linspace(6000, 10800, 130), iwp_dict[nm][150:280], color=col, lw=2, label=nm)
            #ax.plot(np.linspace(0, 14400, 459), iwp_dict[nm][:459],  lw=2, label=nm)#color=col,

ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.set_xlim(000, 14400)
ax.set_ylim(0, 0.01)
#ax.set_ylim(0,0.005)
ax.set_xlabel('Time (s)', color='dimgrey', fontsize = 18)
ax.set_ylabel('IWP\n(kg/m$^{2}$)', rotation =0, labelpad=50, color='dimgrey', fontsize = 18)
plt.subplots_adjust(left = 0.25, bottom = 0.15, right=0.97)
plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='best',ncol=2)
ax.tick_params(axis='both', which='both', labelsize=14, labelcolor='dimgrey', pad=10,  color='dimgrey', length=5, width = 2.5)
plt.savefig(filepath + '../figures/IWP_comparison_uv_runs_14400s.png')
plt.show()

# Check if ICNCx2 full profile = ICNCx2_8-9 + ICNCx2_9-10
ICNC_sum = (iwp_dict['ICNCx2_8-9']-iwp_dict['CTRL']) + (iwp_dict['ICNCx2_9-10']-iwp_dict['CTRL'])
plt.plot(ICNC_sum, label = 'sum')
plt.plot((iwp_dict['ICNCx2_full']-iwp_dict['CTRL']), label = 'full')
plt.legend()
plt.show()



ICNC_dict = {}
for fn, nm in zip(['*17_CTRL*', '*22a*', '*22b*', '*22c*', '*22d*', '*22e*', '*22f*'],['CTRL', 'slower_uv', '_ICNCx2_8-9', '_ICNCx2_9-10', '_no_qF', '_ICNCx1.5_8-9', '_ICNCx1.5_9-10']):
    try:
        f1 = iris.load_cube(filepath + fn + '3600.nc', 'ice_nc_mean')
        f2 = iris.load_cube(filepath + fn + '7200.nc', 'ice_nc_mean')
        f3 = iris.load_cube(filepath + fn + '10800.nc', 'ice_nc_mean')
        ICNC_dict[nm] = np.concatenate((f1.data, f2.data, f3.data), axis = 0)
    except:
        ICNC_dict[nm] = np.zeros(())

model_hts = f3.coord('zn').points + 50

fig, ax = plt.subplots(figsize=(5,6))
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_xlim(1e3,1e8)
ax.set_ylim(model_hts[55]/1000, model_hts[90]/1000)
ax.yaxis.set_ticks([7, 8, 9, 10,])
ax.set_xscale('log')
ax.set_ylabel('Altitude (km)', rotation =90)
ax.set_xlabel('ICNC (cm$^{-3}$)')
ax.tick_params(which='both', axis='both', direction='in')
plt.subplots_adjust(left = 0.2)

#for d in ICNC_dict.keys():
#    if ICNC_dict[d].shape[0] > 282:
#        ax.plot(ICNC_dict[d][62, 60:95], model_hts[60:95]/1000, label=d)
#    else:
#        plt.plot(ICNC_dict[d][18, 60:95], model_hts[60:95]/1000, label = d)
for d, l in zip(['ICNCx1.1_8-9', 'ICNCx1.1_9-10'], ['dashed', 'solid']):
    for t in [48, 56, 64, 72, 90, 98, 106, 114, 122, 130]:
        ax.plot(ICNC_dict[d][t, 60:95], model_hts[60:95]/1000, label=t, linestyle=l)

plt.legend()
plt.show()


fig, ax = plt.subplots(figsize=(8, 5))
#for nm, col in zip(['CTRL', 'ICEx2_8-9', 'ICEx2_9-10', 'ICEx1.1_8-9', 'ICEx1.1_9-10', 'ICEx0.5_8-9',
#                    'ICEx0.5_9-10', 'ICNCx2_8-9', 'ICNCx2_9-10', 'ICNCx1.5_8-9', 'ICNCx1.5_9-10', 'ICNCx1.25_8-9', 'ICNCx1.25_9-10',
#                    'ICNCx1.1_8-9', 'ICNCx1.1_9-10', ], ['#222222','#013D21', '#086D3D', '#0D844C','#1DA766', '#32C580', '#35EB6B',
#                                                         '#6E3685', '#8E44AD', '#A569BD','#9B59B6', '#BB8FCE', '#AF7AC5', '#DA78E4', '#E8DAEF']):
#for nm in [ '17_CTRL',  'ICEx2_8-9_2000', 'ICEx2_9-10_2000',  'ICEx2_8-9_4000', 'ICEx2_9-10_4000','ICEx0.5_8-9_2000', 'ICEx0.5_9-10_2000',  'ICEx0.5_8-9_4000', 'ICEx0.5_9-10_4000', 'ICNCx2_8-9_2000', 'ICNCx2_9-10_2000','contrail-EW_9-10', 'contrail-NS_8-9']:
for nm, col in zip(['CTRL',  'ICEx2_9-10',  'ICEx1.1_9-10',  'ICEx0.5_9-10', 'ICEx2_8-9','ICEx1.1_8-9','ICEx0.5_8-9',  'ICNCx2_9-10','ICNCx1.5_9-10','ICNCx1.25_9-10','ICNCx1.1_9-10','ICNCx2_8-9','ICNCx1.5_8-9','ICNCx1.25_8-9','ICNCx1.1_8-9',  'CTRL'],
                   ['w', '#FF6347', '#FF8C00', '#F4A460',  '#DC143C', '#FF69B4', '#FFC0CB', '#006400', '#20B2AA', '#9ACD32',  '#00FF00', '#00008B', '#4169E1', '#1E90FF', '#87CEEB', '#222222']):
    try:
        #ax.plot(time_srs_hires[150:], iwp_dict[nm][150:], lw=2, label = nm)
        ax.plot(time_srs_hires, iwp_dict[nm], lw=2, label=nm)
    except:
        #ax.plot(np.linspace(6000, 10800, 130), iwp_dict[nm][150:280], color=col, lw=2, label=nm)
        ax.plot(np.linspace(0, 10800, 280), iwp_dict[nm][:280], color=col, lw=2, label=nm)

ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.set_xlim(000, 10800)
ax.set_ylim(0, 0.01)
#ax.set_ylim(0,0.005)
ax.set_xlabel('Time (s)', color='dimgrey', fontsize = 18)
ax.set_ylabel('IWP\n(kg/m$^{2}$)', rotation =0, labelpad=50, color='dimgrey', fontsize = 18)
plt.subplots_adjust(left = 0.25, bottom = 0.15, right=0.97)
plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='best',ncol=2)
ax.tick_params(axis='both', which='both', labelsize=14, labelcolor='dimgrey', pad=10,  color='dimgrey', length=5, width = 2.5)
plt.savefig(filepath + '../figures/IWP_comparison_uv_runs.png')
plt.show()