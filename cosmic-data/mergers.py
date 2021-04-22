"""collection of methods for generating merger populations and rates"""

import utils
import sfh
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u


def get_mergers(zbins, mets, metallicities, alpha, downsample):
    met_weights = sfh.get_metallicity_weights(zbins, mets)
    mergers_tot = []
    for met_read, met, ii in zip(metallicities, mets, range(len(metallicities))):
        print(met)
        BBH, mass_stars = utils.get_cosmic_data(alpha=alpha, met=met, met_read=met_read)
        mergers = []
        for zbin_low, zbin_high, jj in zip(zbins[1:], zbins[:-1], range(len(zbins))):
            # get the midpoint of the zbin
            midz = zbin_low + (zbin_high - zbin_low) / 2
            # get the star formation rate from Madau & Fragos (2017)
            sfr = sfh.madau_17(midz) * u.Msun * u.yr ** (-1) * u.Mpc ** (-3)

            # we want *anything* that merges between the formation and today!
            t_delay_min = 0
            t_delay_max = cosmo.lookback_time(midz).to(u.Myr).value
            BBH_merge = BBH.loc[(BBH.tphys > t_delay_min) & (BBH.tphys < t_delay_max)].copy()
            if len(BBH_merge) > 0:
                # log the formation and merger times
                BBH_merge['t_form'] = cosmo.lookback_time(midz).to(u.Myr).value
                BBH_merge['t_merge'] = BBH_merge.t_form - BBH_merge.tphys
                # filter just to be safe
                BBH_merge = BBH_merge.loc[BBH_merge.t_merge > 1e-3].copy()

                # log the merger redshift
                BBH_merge['z_merge'] = z_interp(BBH_merge.t_merge)

                # log the formation redshift
                BBH_merge['z_form'] = np.ones(len(BBH_merge)) * midz

                # downsample because we have too much data
                BBH_merge = BBH_merge.sample(int(BBH_merge) / downsample)

                # calculate the number of mergers per unit mass formed
                merger_rate_per_mass = len(BBH_merge) / (mass_stars / downsample)

                # calculate the total amount of mass formed at redshift bin midz
                SFR_met_weighted = (sfr * met_weights[ii, jj]).to(u.Msun * u.Gpc ** (-3) * u.yr ** (-1))

                # number of merging BBH formed per comoving volume per detector time
                BBH_merge['dN_dVdtf_source'] = SFR_met_weighted / (mass_stars * u.Msun).value

                dt_f_dt_m = (1 + BBH_merge['z_merge']) * cosmo.H(BBH_merge['z_merge']) / \
                            ((1 + BBH_merge['z_form']) * cosmo.H(BBH_merge['z_form']))

                BBH_merge['dN_dVdtm_source'] = BBH_merge['dN_dVdtf_source'] * dt_f_dt_m
                BBH_merge['dN_dVdtm_det'] = BBH_merge['dN_dVdtm_source'] * 1 / (1 + BBH_merge['z_merge'])
                # differential comoving volume at merger redshift
                BBH_merge['dV_dz'] = cosmo.differential_comoving_volume(np.array(BBH_merge['z_merge'])).to(
                    u.Gpc ** (3) * u.steradian ** (-1)).value * (4 * np.pi)
                if len(mergers) == 0:
                    mergers = BBH_merge
                else:
                    mergers = mergers.append(BBH_merge)
            else:
                continue
        if len(mergers_tot) == 0:
            mergers_tot = mergers
        else:
            mergers_tot = mergers_tot.append(mergers)
    if len(mergers_tot) > 0:
        return mergers_tot

    else:
        return []
