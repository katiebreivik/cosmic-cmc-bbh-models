import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
import sfh
import mergers as m

import matplotlib.pyplot as plt

metallicities = ['0.000085', '0.00012673', '0.00018896', '0.00028173', '0.00042005',
                 '0.00062629', '0.00093378', '0.00139224', '0.00207579', '0.00309496',
                 '0.00461451', '0.00688012', '0.0102581', '0.01529458', '0.02280385', '0.034']
mets = [0.00008500, 0.00012673, 0.00018896, 0.00028173, 0.00042005,
        0.00062629, 0.00093378, 0.00139224, 0.00207579, 0.00309496,
        0.00461451, 0.00688012, 0.01025810, 0.01529458, 0.02280385, 0.03400000]

## Set up interpolator to go between redshift and lookback time
t_merge_list = np.linspace(1e-3, 13700, 500)

z_merge_list = []
for t in t_merge_list:
    z_merge_list.append(z_at_value(cosmo.lookback_time, t * u.Myr))

z_interp = interp1d(t_merge_list, z_merge_list)

# set up a linspace of redshift bins:
zbins = np.linspace(0, 15, 100)

# set up metallicity weights for the star formation history
met_weights = sfh.get_metallicity_weights(zbins, mets)

# set up differential comoving volume
dVdz = cosmo.differential_comoving_volume(zbins).to(u.Gpc**(3)*u.steradian**(-1)).value*(4*np.pi)

# decide to save data 
save_data = True

#decide to make plots
make_plots = True
# get mergers for each alpha
for alpha in [0.25, 0.5, 0.75, 1, 2, 3, 4, 5]:
    mergers = m.get_mergers(zbins, mets, metallicities, alpha=alpha, z_interp=z_interp, downsample=10)
    mergers['z_digits'] = np.digitize(mergers['z_merge'], zbins)

    if save_data:
        mergers.to_hdf('all_mergers_alpha_{}.h5'.format(alpha), key='mergers')
    
    if make_plots:
        comoving_rate = mergers.groupby('z_digits').sum().reset_index()
        fig = plt.figure()
        plt.plot(zbins[comoving_rate['z_digits']], comoving_rate['dN_dVdtm_det'], lw=2)
        plt.ylabel('Comoving merger rate [N/(Gpc$^{3}$yr)]', size=14)
        plt.xlabel('redshift', size=14)
        plt.tight_layout()
        plt.show()
        plt.close()

        fig = plt.figure()
        plt.plot(zbins[comoving_rate['z_digits']][1:],
                 np.cumsum(comoving_rate['dN_dVdtm_det'][1:] * dVdz[comoving_rate['z_digits']][1:]) *
                 (zbins[comoving_rate['z_digits']][1:] - zbins[comoving_rate['z_digits']][:-1]), lw=2)
        plt.ylabel('Volumetric rate [N/yr]', size=14)
        plt.xlabel('redshift', size=14)
        plt.tight_layout()
        plt.show()
        plt.close()

    