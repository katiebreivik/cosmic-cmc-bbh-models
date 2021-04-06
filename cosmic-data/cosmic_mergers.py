import pandas as pd

import pandas as pd
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value 
from astropy import units as u
import filters
import rate_functions
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt


mets=['0.000085', '0.00012673', '0.00018896', '0.00028173', '0.00042005', '0.00062629', '0.00093378', '0.00139224', '0.00207579', '0.00309496', '0.00461451', '0.00688012', '0.0102581', '0.01529458', '0.02280385', '0.034']

models=['alpha_0.25', 'alpha_0.5', 'alpha_0.75', 'alpha_1', 'alpha_2', 'alpha_3', 'alpha_4', 'alpha_5']

t_merge_list = np.linspace(1e-3,13700,100)

z_merge_list = []
for t in t_merge_list:
    z_merge_list.append(z_at_value(cosmo.lookback_time, t * u.Myr))
z_interp = interp1d(t_merge_list, z_merge_list)


zbins = np.linspace(0, 15, 100)
met_weights = np.zeros((len(mets),len(zbins)))
for met, ii in zip(mets, range(len(mets))):
    for zbin_low, zbin_high, jj in zip(zbins[::-1][1:], zbins[::-1][:-1], range(len(zbins))):
        midz = zbin_low + (zbin_high - zbin_low)/2
        met_weights[ii,jj] = rate_functions.metal_disp_z(midz, float(met))
        

for model,alpha in zip(models,[0.25,0.5,0.75,1,2,3,4,5]):
    merger_mets = []
    merger_rate_mets = []
    for met, ii in zip(mets, range(len(mets))):
        mass_stars = pd.read_hdf(model+'/BBH_mergers_{}.h5'.format(met), key='mass_stars')
        mass_stars = mass_stars.iloc[0][0]
        BBH = pd.read_hdf(model+'/BBH_mergers_{}.h5'.format(met), key='mergers')
        
        zbin_contribution = []
        mergers = []
        for zbin_low, zbin_high, jj in zip(zbins[::-1][1:], zbins[::-1][:-1], range(len(zbins))):
            midz = zbin_low + (zbin_high - zbin_low)/2
            sfr = rate_functions.sfr_z(midz) * u.M_sun * u.Mpc**(-3) * u.yr**(-1)
            E_z = (cosmo._Onu0*(1+midz)**4 + cosmo._Om0*(1+midz)**3 + cosmo._Ok0*(1+midz)**2 + cosmo._Ode0)**(1./2)
            
            t_delay_min = 0
            t_delay_max = cosmo.lookback_time(midz).to(u.Myr).value
            BBH_merge = BBH.loc[(BBH.tphys > t_delay_min) & (BBH.tphys < t_delay_max)]
            BBH_merge['t_form'] = cosmo.lookback_time(midz).to(u.Myr).value
            BBH_merge['t_merge'] = BBH_merge.t_form - BBH_merge.tphys
            BBH_merge = BBH_merge.loc[BBH_merge.t_merge > 1e-3]
            BBH_merge['z_merge'] = z_interp(BBH_merge.t_merge)
            merger_rate_per_mass = len(BBH_merge)/mass_stars
            
            merger_rate_per_z = ((sfr * (1/mass_stars) * met_weights[ii,jj] / ((1+midz) * E_z)) * (zbin_high-zbin_low)).value
            merger_rate_per_z = merger_rate_per_z * 1e9 # convert from Mpc^(-3) to Gpc^(-3)
            BBH_merge['weights'] = merger_rate_per_z
            BBH_merge['z_form'] = np.ones(len(BBH_merge)) * midz
            BBH_merge['met'] = np.ones(len(BBH_merge)) * np.float(met)
            
            BBH_merge = BBH_merge[['mass_1', 'mass_2', 't_form', 't_merge', 'z_form', 'z_merge', 'met', 'tphys', 'weights']]
            if len(mergers) == 0:
                mergers = BBH_merge
            else:
                mergers = mergers.append(BBH_merge)
    
            zbin_contribution.append(merger_rate_per_z * len(BBH_merge))
        if len(merger_mets) == 0:
            merger_mets = mergers
        else:
            merger_mets = merger_mets.append(mergers)
        merger_rate_mets.append(zbin_contribution)
    
    merger_mets.to_hdf('all_mergers_weighted_alpha_{}.h5'.format(alpha), key='mergers')
    np.save('comoving_merger_rates_{}.npy'.format(alpha), merger_rate_mets)

for alpha in [0.25,0.5,0.75,1,2,3,4,5]:
    merger_set = pd.read_hdf('all_mergers_weighted_alpha_{}.h5'.format(alpha), key='mergers')
    merger_set['z_digits'] = np.digitize(merger_set['z_merge'], zbins)
    merger_rate = merger_set.groupby('z_digits').sum()['weights']
    import pdb
    pdb.set_trace()
    cumulative_rate = np.cumsum(merger_rate * cosmo.comoving_volume(zbins).to(u.Gpc**(3)).value * 1/(1+zbins))
    rate_data = np.vstack([zbins, cumulative_rate]).T
    print(rate_data)
    np.save('cumulative_rates_{}'.format(alpha), rate_data)

fig = plt.figure(figsize=(14,3))
ax1 = plt.subplot(151)
ax2 = plt.subplot(152)
ax3 = plt.subplot(153)
ax4 = plt.subplot(154)
ax5 = plt.subplot(155)
for model, alpha in zip(models, [0.25,0.5,0.75,1,2,3,4,5]):
    merger_set = pd.read_hdf('all_mergers_weighted_alpha_{}.h5'.format(alpha), key='mergers')
    print(merger_set.columns)
    ax1.hist(merger_set.mass_1, label='{}'.format(model), density=False, histtype='step', lw=2, bins=np.linspace(0,125,20), weights=merger_set.weights)
    ax2.hist(merger_set.mass_2, density=False, histtype='step', lw=2, bins=np.linspace(0,90,20), weights=merger_set.weights)
    ax3.hist(merger_set.z_merge, density=False, histtype='step', lw=2, bins=np.linspace(0, 5, 20), weights=merger_set.weights)
    ax4.hist(merger_set.z_form, density=False, histtype='step', lw=2, bins=np.linspace(0, 5, 20), weights=merger_set.weights)
    ax5.hist(np.log10(merger_set.tphys), density=False, histtype='step', lw=2, bins=np.linspace(0, 4.1, 20), weights=merger_set.weights)
    
    ax1.set_xlabel('M1 [Msun]')
    ax2.set_xlabel('M2 [Msun]')
    ax3.set_xlabel('z merge')
    ax4.set_xlabel('z form')
    ax5.set_xlabel('Log(t delay/[Myr])')
    ax1.set_ylabel('weighted histogram')
ax1.legend()
plt.tight_layout()
plt.savefig('hist_compare.pdf'.format(model))
